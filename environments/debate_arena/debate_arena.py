import logging
import re
from typing import Optional

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

CONCESSION_PATTERNS = [
    r"\bi concede\b",
    r"\byou win\b",
    r"\bi agree with your position\b",
    r"\byou(?:'ve| have) convinced me\b",
    r"\bi(?:'m| am) convinced\b",
    r"\bi surrender\b",
    r"\bi accept your argument\b",
    r"\bi was wrong\b",
    r"\byou(?:'re| are) right\b",
    r"\bi lose\b",
]

DEBATE_TOPICS = [
    {
        "topic": "AI will create more jobs than it destroys",
        "pro": "AI will create more jobs than it destroys",
        "con": "AI will destroy more jobs than it creates",
    },
    {
        "topic": "Social media does more harm than good",
        "pro": "Social media does more harm than good",
        "con": "Social media does more good than harm",
    },
    {
        "topic": "Space exploration is worth the investment",
        "pro": "Space exploration is worth the investment",
        "con": "Space exploration is not worth the investment",
    },
    {
        "topic": "Universal basic income should be implemented",
        "pro": "Universal basic income should be implemented",
        "con": "Universal basic income should not be implemented",
    },
    {
        "topic": "Nuclear energy is the best solution to climate change",
        "pro": "Nuclear energy is the best solution to climate change",
        "con": "Nuclear energy is not the best solution to climate change",
    },
]


def _has_concession(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in CONCESSION_PATTERNS)


PRO_SYSTEM_PROMPT = (
    "You are a skilled debater. You are arguing IN FAVOR of the following position:\n\n"
    "\"{pro_position}\"\n\n"
    "Argue persuasively and respond to your opponent's points. "
    "If you become genuinely convinced that your opponent is right, you may say "
    "'I concede' to end the debate. Do not concede easily -- only if truly persuaded."
)

CON_SYSTEM_PROMPT = (
    "You are a skilled debater. You are arguing AGAINST the following position:\n\n"
    "\"{pro_position}\"\n\n"
    "Your position is: \"{con_position}\"\n\n"
    "Argue persuasively and respond to your opponent's points. "
    "If you become genuinely convinced that your opponent is right, you may say "
    "'I concede' to end the debate. Do not concede easily -- only if truly persuaded."
)


def build_dataset() -> Dataset:
    data = []
    for t in DEBATE_TOPICS:
        system_msg = PRO_SYSTEM_PROMPT.format(pro_position=t["pro"])
        user_msg = (
            f"The debate topic is: \"{t['topic']}\"\n\n"
            f"You are arguing FOR: \"{t['pro']}\"\n\n"
            "Please present your opening argument."
        )
        data.append({
            "prompt": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "answer": t["topic"],
            "info": {
                "topic": t["topic"],
                "pro_position": t["pro"],
                "con_position": t["con"],
            },
        })
    return Dataset.from_list(data)


def load_environment(
    opponent_model: str = "anthropic/claude-4.5-sonnet",
    opponent_base_url: str = "https://api.pinference.ai/api/v1",
    opponent_api_key: Optional[str] = None,
    max_turns: int = 10,
    **kwargs,
) -> vf.Environment:
    """
    Load the debate arena environment.

    Args:
        opponent_model: Model name for the opponent (con side).
        opponent_base_url: Base URL for the opponent's API. Defaults to Prime Intellect.
        opponent_api_key: API key for the opponent. Defaults to PRIME_API_KEY env var.
        max_turns: Max turns per side (total messages = 2 * max_turns).
    """
    import os
    api_key = opponent_api_key or os.getenv("PRIME_API_KEY", "EMPTY")
    opponent_client = AsyncOpenAI(
        base_url=opponent_base_url,
        api_key=api_key,
    )

    class DebateEnv(vf.MultiTurnEnv):
        async def setup_state(self, state: State) -> State:
            state["conceded_by"] = None  # "pro" or "con" or None
            return state

        @vf.stop
        async def concession_detected(self, state: State) -> bool:
            return state.get("conceded_by") is not None

        async def env_response(
            self, messages: Messages, state: State, **kwargs
        ) -> Messages:
            # Check if the pro model (assistant) just conceded
            last_assistant = None
            for m in reversed(messages):
                if m["role"] == "assistant":
                    last_assistant = m
                    break

            if last_assistant and _has_concession(last_assistant.get("content", "")):
                state["conceded_by"] = "pro"
                state["final_env_response"] = [
                    {"role": "user", "content": "[DEBATE ENDED: Pro side conceded.]"}
                ]
                return state["final_env_response"]

            # Build opponent conversation with con system prompt
            con_system = CON_SYSTEM_PROMPT.format(
                pro_position=state["info"]["pro_position"],
                con_position=state["info"]["con_position"],
            )

            # Convert the conversation: assistant messages become "user" for opponent,
            # and the original "user" messages become "assistant" for opponent.
            opponent_messages = [{"role": "system", "content": con_system}]
            for m in messages:
                if m["role"] == "system":
                    continue
                elif m["role"] == "assistant":
                    opponent_messages.append({"role": "user", "content": m["content"]})
                elif m["role"] == "user":
                    # Skip the initial framing message, map others as opponent's prior turns
                    if len(opponent_messages) == 1:
                        # First user message is the debate prompt — rephrase for opponent
                        opponent_messages.append({
                            "role": "user",
                            "content": (
                                f"The debate topic is: \"{state['info']['topic']}\"\n\n"
                                f"You are arguing AGAINST: \"{state['info']['pro_position']}\"\n\n"
                                "Your opponent has just presented their opening argument. "
                                "Here it is — please respond."
                            ),
                        })
                    else:
                        opponent_messages.append({"role": "assistant", "content": m["content"]})

            response = await opponent_client.chat.completions.create(
                model=opponent_model,
                messages=opponent_messages,
                temperature=0.7,
                max_tokens=1024,
            )
            con_reply = response.choices[0].message.content

            # Check if opponent conceded
            if _has_concession(con_reply or ""):
                state["conceded_by"] = "con"
                state["final_env_response"] = [
                    {"role": "user", "content": f"{con_reply}\n\n[DEBATE ENDED: Con side conceded.]"}
                ]
                return state["final_env_response"]

            return [{"role": "user", "content": con_reply}]

    def winner_reward(completion: Messages, state: State, **kwargs) -> float:
        conceded_by = state.get("conceded_by")
        if conceded_by == "con":
            return 1.0  # pro (rollout model) won
        elif conceded_by == "pro":
            return 0.0  # pro lost
        else:
            return 0.5  # draw (max turns)

    def concession_metric(state: State, **kwargs) -> float:
        conceded_by = state.get("conceded_by")
        if conceded_by == "con":
            return 1.0  # opponent conceded
        elif conceded_by == "pro":
            return -1.0  # we conceded
        return 0.0  # draw

    rubric = vf.Rubric(funcs=[winner_reward], weights=[1.0])
    rubric.add_metric(concession_metric)

    dataset = build_dataset()

    return DebateEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
    )

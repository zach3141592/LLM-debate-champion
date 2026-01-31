"""
Round-robin debate tournament across 5 models.

Each pair debates every topic once. Model A argues pro, Model B argues con.
To keep it fair, each pair debates in both directions (A=pro/B=con AND B=pro/A=con).

Usage:
    python run_tournament.py
"""

import asyncio
import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

from openai import AsyncOpenAI

from debate_arena import load_environment

PRIME_API_BASE = "https://api.pinference.ai/api/v1"

# --- Configure your 5 models here (all via Prime Intellect API) ---
MODELS = [
    {
        "name": "claude-sonnet",
        "model": "anthropic/claude-sonnet-4.5",
    },
    {
        "name": "gemini-2.5-flash",
        "model": "google/gemini-2.5-flash",
    },
    {
        "name": "qwen3-235b",
        "model": "qwen/qwen3-235b-a22b-instruct-2507",
    },
    {
        "name": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b",
    },
    {
        "name": "kimi-k2",
        "model": "moonshotai/kimi-k2-0905",
    },
]

MAX_TURNS = 10  # per side


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=PRIME_API_BASE,
        api_key=os.getenv("PRIME_API_KEY", "EMPTY"),
    )


async def run_matchup(
    pro_cfg: dict, con_cfg: dict
) -> list[dict]:
    """Run one matchup: pro_cfg model argues pro, con_cfg model argues con, across all topics."""
    prime_api_key = os.getenv("PRIME_API_KEY", "EMPTY")
    env = load_environment(
        opponent_model=con_cfg["model"],
        opponent_base_url=PRIME_API_BASE,
        opponent_api_key=prime_api_key,
        max_turns=MAX_TURNS,
    )
    pro_client = get_client()

    results = await env.evaluate(
        client=pro_client,
        model=pro_cfg["model"],
        num_examples=-1,  # all topics
        rollouts_per_example=1,
        max_concurrent=4,
        use_tqdm=True,
    )

    matchup_results = []
    for output in results["outputs"]:
        matchup_results.append({
            "pro_model": pro_cfg["name"],
            "con_model": con_cfg["name"],
            "topic": output.get("answer", ""),
            "reward": output["reward"],
            "num_turns": output.get("metrics", {}).get("num_turns", 0),
            "concession_metric": output.get("metrics", {}).get("concession_metric", 0),
        })
    return matchup_results


async def run_tournament():
    all_results = []
    pairings = list(combinations(range(len(MODELS)), 2))

    print(f"Running tournament: {len(MODELS)} models, {len(pairings)} pairings (x2 directions)")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print()

    # Build all matchup tasks (both directions per pairing)
    tasks = []
    for a, b in pairings:
        model_a = MODELS[a]
        model_b = MODELS[b]
        print(f"  Queuing: {model_a['name']} (pro) vs {model_b['name']} (con)")
        tasks.append(run_matchup(model_a, model_b))
        print(f"  Queuing: {model_b['name']} (pro) vs {model_a['name']} (con)")
        tasks.append(run_matchup(model_b, model_a))

    print(f"\nRunning {len(tasks)} debates in parallel...")
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results_list:
        if isinstance(result, Exception):
            print(f"  Matchup failed: {result}")
        else:
            all_results.extend(result)

    # Aggregate leaderboard
    stats = {}
    for m in MODELS:
        stats[m["name"]] = {"wins": 0, "losses": 0, "draws": 0, "total_reward": 0.0, "debates": 0}

    for r in all_results:
        pro = r["pro_model"]
        con = r["con_model"]
        reward = r["reward"]

        stats[pro]["debates"] += 1
        stats[con]["debates"] += 1
        stats[pro]["total_reward"] += reward
        stats[con]["total_reward"] += (1.0 - reward)

        if reward == 1.0:
            stats[pro]["wins"] += 1
            stats[con]["losses"] += 1
        elif reward == 0.0:
            stats[pro]["losses"] += 1
            stats[con]["wins"] += 1
        else:
            stats[pro]["draws"] += 1
            stats[con]["draws"] += 1

    # Print leaderboard
    print("\n" + "=" * 60)
    print("DEBATE TOURNAMENT LEADERBOARD")
    print("=" * 60)

    sorted_models = sorted(stats.items(), key=lambda x: x[1]["total_reward"], reverse=True)
    for rank, (name, s) in enumerate(sorted_models, 1):
        win_rate = s["total_reward"] / s["debates"] if s["debates"] > 0 else 0
        print(
            f"  {rank}. {name:20s}  "
            f"W:{s['wins']:3d}  L:{s['losses']:3d}  D:{s['draws']:3d}  "
            f"Score: {s['total_reward']:.1f}/{s['debates']}  "
            f"WinRate: {win_rate:.1%}"
        )

    print("=" * 60)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"tournament_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({"results": all_results, "leaderboard": stats}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_tournament())

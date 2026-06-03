from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_ran_slicing.zip"
N_EPISODES = 30
BASE_SEED = 10_000

STEP_INFO_KEYS = [
    "alpha_A",
    "prb_ambulance",
    "prb_ordinary",
    "ambulance_arrival_mbit",
    "ordinary_arrival_mbit",
    "ambulance_capacity_mbps",
    "ordinary_capacity_mbps",
    "ambulance_queue_mbit",
    "ordinary_queue_mbit",
    "ambulance_latency_s",
    "ambulance_sla_violation",
    "ordinary_throughput_mbps",
    "prb_utilization",
    "reward_total",
    "n_ambulance",
    "n_ordinary_vehicles",
    "n_embb_users",
    "ambulance_emergency",
    "embb_surge_active",
]


def evaluate_dqn() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the trained DQN and evaluate it with deterministic actions."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained DQN model not found: {MODEL_PATH}. "
            "Run agents/train_dqn.py before evaluation."
        )

    env = RANSlicingEnv(CONFIG_PATH)
    model = DQN.load(str(MODEL_PATH), env=env)

    step_logs = []
    episode_summaries = []

    for episode in range(N_EPISODES):
        seed = BASE_SEED + episode
        obs, _ = env.reset(seed=seed)

        total_reward = 0.0
        action_counts = np.zeros(env.action_space.n, dtype=int)
        ambulance_latencies_ms = []
        ambulance_sla_violations = []
        ordinary_throughputs = []
        ordinary_deficits = []
        prb_utilizations = []

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            # Use deterministic DQN inference for evaluation.
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            latency_ms = float(info["ambulance_latency_s"]) * 1000.0
            sla_violation = int(info["ambulance_sla_violation"])
            ordinary_throughput = float(info["ordinary_throughput_mbps"])
            prb_utilization = float(info["prb_utilization"])

            total_reward += float(reward)
            action_counts[action] += 1
            ambulance_latencies_ms.append(latency_ms)
            ambulance_sla_violations.append(sla_violation)
            ordinary_throughputs.append(ordinary_throughput)
            ordinary_deficits.append(
                int(ordinary_throughput < env.ordinary_throughput_target)
            )
            prb_utilizations.append(prb_utilization)

            # Keep step logs to the exact fields requested for DQN evaluation.
            step_log = {
                "episode": episode,
                "step": step,
                "action": action,
            }
            for key in STEP_INFO_KEYS:
                step_log[key] = info[key]
            step_logs.append(step_log)

            step += 1

        episode_summaries.append(
            build_episode_summary(
                episode=episode,
                seed=seed,
                total_reward=total_reward,
                ambulance_latencies_ms=ambulance_latencies_ms,
                ambulance_sla_violations=ambulance_sla_violations,
                ordinary_throughputs=ordinary_throughputs,
                ordinary_deficits=ordinary_deficits,
                prb_utilizations=prb_utilizations,
                action_counts=action_counts,
            )
        )

    return step_logs, episode_summaries


def build_episode_summary(
    episode: int,
    seed: int,
    total_reward: float,
    ambulance_latencies_ms: list[float],
    ambulance_sla_violations: list[int],
    ordinary_throughputs: list[float],
    ordinary_deficits: list[int],
    prb_utilizations: list[float],
    action_counts: np.ndarray,
) -> dict[str, Any]:
    """Aggregate one DQN evaluation episode into report metrics."""
    sla_violation_rate = float(np.mean(ambulance_sla_violations))
    summary = {
        "episode": episode,
        "seed": seed,
        "total_reward": float(total_reward),
        "avg_ambulance_latency_ms": float(np.mean(ambulance_latencies_ms)),
        "p95_ambulance_latency_ms": float(
            np.percentile(ambulance_latencies_ms, 95)
        ),
        "ambulance_sla_violation_rate": sla_violation_rate,
        "ambulance_qos_satisfaction_rate": 1.0 - sla_violation_rate,
        "avg_ordinary_throughput_mbps": float(np.mean(ordinary_throughputs)),
        "ordinary_throughput_deficit_rate": float(np.mean(ordinary_deficits)),
        "avg_prb_utilization": float(np.mean(prb_utilizations)),
    }

    for action in range(8):
        summary[f"action_{action}_count"] = int(action_counts[action])

    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_compact_summary(summaries: list[dict[str, Any]]) -> None:
    """Print the compact aggregate DQN summary requested by the spec."""
    action_totals = np.zeros(8, dtype=int)
    for row in summaries:
        for action in range(8):
            action_totals[action] += int(row[f"action_{action}_count"])

    print("\nDQN summary")
    print(
        "mean total reward: "
        f"{np.mean([row['total_reward'] for row in summaries]):.6f}"
    )
    print(
        "mean ambulance latency: "
        f"{np.mean([row['avg_ambulance_latency_ms'] for row in summaries]):.6f} ms"
    )
    print(
        "mean SLA violation rate: "
        f"{np.mean([row['ambulance_sla_violation_rate'] for row in summaries]):.6f}"
    )
    print(
        "mean ordinary throughput: "
        f"{np.mean([row['avg_ordinary_throughput_mbps'] for row in summaries]):.6f} Mbps"
    )
    print(
        "mean PRB utilization: "
        f"{np.mean([row['avg_prb_utilization'] for row in summaries]):.6f}"
    )
    print(f"action distribution: {action_totals.tolist()}")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    steps_path = PROJECT_ROOT / "results" / f"dqn_steps_{timestamp}.csv"
    summary_path = PROJECT_ROOT / "results" / f"dqn_summary_{timestamp}.csv"

    step_logs, episode_summaries = evaluate_dqn()
    write_csv(steps_path, step_logs)
    write_csv(summary_path, episode_summaries)
    print_compact_summary(episode_summaries)
    print(f"\nSaved step logs to {steps_path}")
    print(f"Saved episode summary to {summary_path}")


if __name__ == "__main__":
    main()

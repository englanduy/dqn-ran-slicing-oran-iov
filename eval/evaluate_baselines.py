from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.baseline_policies import (  # noqa: E402
    GreedySLAPolicy,
    LoadBasedPolicy,
    PriorityPolicy,
    RandomPolicy,
    StaticPolicy,
)
from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402


CONFIG_PATH = "configs/default_config.yaml"
N_EPISODES = 30
BASE_SEED = 42


def make_policies() -> dict[str, Any]:
    """Create one instance of each baseline policy."""
    return {
        "StaticPolicy": StaticPolicy(CONFIG_PATH),
        "PriorityPolicy": PriorityPolicy(CONFIG_PATH),
        "LoadBasedPolicy": LoadBasedPolicy(CONFIG_PATH),
        "GreedySLAPolicy": GreedySLAPolicy(CONFIG_PATH),
        "RandomPolicy": RandomPolicy(CONFIG_PATH, seed=BASE_SEED),
    }


def evaluate_policy(
    policy_name: str,
    policy: Any,
    env: RANSlicingEnv,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one policy for N_EPISODES and return step logs plus summaries."""
    step_logs = []
    episode_summaries = []

    for episode in range(N_EPISODES):
        seed = BASE_SEED + seed_offset + episode
        obs, _ = env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()

        total_reward = 0.0
        action_counts = np.zeros(env.action_space.n, dtype=int)
        ambulance_latencies_ms = []
        ambulance_sla_violations = []
        ordinary_throughputs = []
        ordinary_deficits = []
        prb_utilizations = []

        last_info = None
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            # Select actions causally from the current observation and previous info.
            action = int(policy.select_action(obs, last_info))
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info

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

            step_log = {
                "policy": policy_name,
                "episode": episode,
                "seed": seed,
                "step": step,
                "action": action,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
            }
            step_log.update(info)
            step_logs.append(step_log)

            step += 1

        summary = build_episode_summary(
            policy_name=policy_name,
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
        episode_summaries.append(summary)

    return step_logs, episode_summaries


def build_episode_summary(
    policy_name: str,
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
    """Aggregate one episode into the metrics required by the spec."""
    sla_violation_rate = float(np.mean(ambulance_sla_violations))
    summary = {
        "policy": policy_name,
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
    """Write dictionaries to CSV while preserving first-seen field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_compact_summary(summaries: list[dict[str, Any]]) -> None:
    """Print mean metrics grouped by policy."""
    print("\nBaseline summary")
    print(
        "policy,total_reward,ambulance_latency_ms,"
        "sla_violation_rate,ordinary_throughput_mbps,prb_utilization"
    )

    policy_names = list(dict.fromkeys(row["policy"] for row in summaries))
    for policy_name in policy_names:
        rows = [row for row in summaries if row["policy"] == policy_name]
        print(
            f"{policy_name},"
            f"{np.mean([row['total_reward'] for row in rows]):.6f},"
            f"{np.mean([row['avg_ambulance_latency_ms'] for row in rows]):.6f},"
            f"{np.mean([row['ambulance_sla_violation_rate'] for row in rows]):.6f},"
            f"{np.mean([row['avg_ordinary_throughput_mbps'] for row in rows]):.6f},"
            f"{np.mean([row['avg_prb_utilization'] for row in rows]):.6f}"
        )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    steps_path = PROJECT_ROOT / "results" / f"baseline_steps_{timestamp}.csv"
    summary_path = PROJECT_ROOT / "results" / f"baseline_summary_{timestamp}.csv"

    env = RANSlicingEnv(CONFIG_PATH)
    all_step_logs = []
    all_episode_summaries = []

    for policy_index, (policy_name, policy) in enumerate(make_policies().items()):
        step_logs, episode_summaries = evaluate_policy(
            policy_name=policy_name,
            policy=policy,
            env=env,
            seed_offset=policy_index * N_EPISODES,
        )
        all_step_logs.extend(step_logs)
        all_episode_summaries.extend(episode_summaries)

    write_csv(steps_path, all_step_logs)
    write_csv(summary_path, all_episode_summaries)
    print_compact_summary(all_episode_summaries)
    print(f"\nSaved step logs to {steps_path}")
    print(f"Saved episode summary to {summary_path}")


if __name__ == "__main__":
    main()

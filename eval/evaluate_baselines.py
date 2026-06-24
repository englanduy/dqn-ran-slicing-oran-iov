from __future__ import annotations

import argparse
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
    EmergencyPriorityPolicy,
    GreedySLAPolicy,
    GuaranteedLoadBasedPolicy,
    LoadBasedPolicy,
    PriorityPolicy,
    RandomPolicy,
    StaticPolicy,
)
from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402


CONFIG_PATH = "configs/default_config.yaml"
N_EPISODES = 30
BASE_SEED = 42

POLICY_REGISTRY = {
    "static": ("StaticPolicy", StaticPolicy),
    "emergency_priority": ("EmergencyPriorityPolicy", EmergencyPriorityPolicy),
    "priority": ("PriorityPolicy", PriorityPolicy),
    "load_based": ("LoadBasedPolicy", LoadBasedPolicy),
    "guaranteed_load_based": (
        "GuaranteedLoadBasedPolicy",
        GuaranteedLoadBasedPolicy,
    ),
    "greedy_sla": ("GreedySLAPolicy", GreedySLAPolicy),
    "random": ("RandomPolicy", RandomPolicy),
}
DEFAULT_POLICY_KEYS = ["static", "priority", "load_based", "greedy_sla", "random"]


def parse_args() -> argparse.Namespace:
    """Parse baseline evaluation controls."""
    parser = argparse.ArgumentParser(description="Evaluate baseline RAN slicing policies.")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICY_REGISTRY),
        default=DEFAULT_POLICY_KEYS,
        help="Policy keys to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional prefix for output CSV filenames.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes. Defaults to 30.",
    )
    return parser.parse_args()


def make_policies(policy_keys: list[str]) -> dict[str, Any]:
    """Create the selected baseline policy instances."""
    policies = {}
    for key in policy_keys:
        policy_name, policy_cls = POLICY_REGISTRY[key]
        if key == "random":
            policies[policy_name] = policy_cls(CONFIG_PATH, seed=BASE_SEED)
        else:
            policies[policy_name] = policy_cls(CONFIG_PATH)
    return policies


def evaluate_policy(
    policy_name: str,
    policy: Any,
    env: RANSlicingEnv,
    seed_offset: int,
    n_episodes: int = N_EPISODES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one policy for the requested episodes using causal action selection."""
    step_logs = []
    episode_summaries = []

    for episode in range(n_episodes):
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
            # Policies only receive the current observation and previous step info.
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

        episode_summaries.append(
            build_episode_summary(
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
        )

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
    """Aggregate one episode into the required report metrics."""
    sla_violation_rate = float(np.mean(ambulance_sla_violations))
    qos_satisfaction_rate = 1.0 - sla_violation_rate

    summary = {
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "total_reward": float(total_reward),
        "avg_ambulance_latency_ms": float(np.mean(ambulance_latencies_ms)),
        "p95_ambulance_latency_ms": float(
            np.percentile(ambulance_latencies_ms, 95)
        ),
        "sla_violation_rate": sla_violation_rate,
        "qos_satisfaction_rate": qos_satisfaction_rate,
        "ambulance_sla_violation_rate": sla_violation_rate,
        "ambulance_qos_satisfaction_rate": qos_satisfaction_rate,
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
            f"{np.mean([row['sla_violation_rate'] for row in rows]):.6f},"
            f"{np.mean([row['avg_ordinary_throughput_mbps'] for row in rows]):.6f},"
            f"{np.mean([row['avg_prb_utilization'] for row in rows]):.6f}"
        )


def output_paths(run_name: str | None) -> tuple[Path, Path]:
    """Resolve timestamped output paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        return (
            PROJECT_ROOT / "results" / f"{run_name}_steps_{timestamp}.csv",
            PROJECT_ROOT / "results" / f"{run_name}_summary_{timestamp}.csv",
        )
    return (
        PROJECT_ROOT / "results" / f"baseline_steps_{timestamp}.csv",
        PROJECT_ROOT / "results" / f"baseline_summary_{timestamp}.csv",
    )


def main() -> None:
    args = parse_args()
    n_episodes = args.episodes if args.episodes is not None else N_EPISODES
    if n_episodes <= 0:
        raise ValueError("--episodes must be a positive integer.")
    steps_path, summary_path = output_paths(args.run_name)

    env = RANSlicingEnv(CONFIG_PATH)
    all_step_logs = []
    all_episode_summaries = []

    for policy_index, (policy_name, policy) in enumerate(make_policies(args.policies).items()):
        step_logs, episode_summaries = evaluate_policy(
            policy_name=policy_name,
            policy=policy,
            env=env,
            seed_offset=policy_index * n_episodes,
            n_episodes=n_episodes,
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

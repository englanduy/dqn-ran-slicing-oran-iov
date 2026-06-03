import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv


N_EPISODES = 5
CONFIG_PATH = "configs/default_config.yaml"
STEP_LOG_PATH = PROJECT_ROOT / "results" / "random_env_check_steps.csv"
SUMMARY_LOG_PATH = PROJECT_ROOT / "results" / "random_env_check_summary.csv"


def main() -> None:
    env = RANSlicingEnv(CONFIG_PATH)
    env.action_space.seed(42)

    step_logs = []
    summary_logs = []

    for episode in range(N_EPISODES):
        env.reset(seed=42 + episode)

        total_reward = 0.0
        action_counts = np.zeros(env.action_space.n, dtype=int)
        ambulance_latencies_ms = []
        ambulance_sla_violations = []
        ordinary_throughputs = []
        prb_utilizations = []

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            action = int(env.action_space.sample())
            _, reward, terminated, truncated, info = env.step(action)

            latency_ms = info["ambulance_latency_s"] * 1000.0
            total_reward += reward
            action_counts[action] += 1
            ambulance_latencies_ms.append(latency_ms)
            ambulance_sla_violations.append(info["ambulance_sla_violation"])
            ordinary_throughputs.append(info["ordinary_throughput_mbps"])
            prb_utilizations.append(info["prb_utilization"])

            step_log = {
                "episode": episode,
                "step": step,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
            step_log.update(info)
            step_logs.append(step_log)

            step += 1

        summary = {
            "episode": episode,
            "total_reward": total_reward,
            "avg_ambulance_latency_ms": float(np.mean(ambulance_latencies_ms)),
            "p95_ambulance_latency_ms": float(
                np.percentile(ambulance_latencies_ms, 95)
            ),
            "ambulance_sla_violation_rate": float(np.mean(ambulance_sla_violations)),
            "avg_ordinary_throughput_mbps": float(np.mean(ordinary_throughputs)),
            "avg_prb_utilization": float(np.mean(prb_utilizations)),
            "action_counts": action_counts.tolist(),
        }
        summary_logs.append(summary)

        print(f"Episode {episode}")
        print(f"  total_reward: {summary['total_reward']:.6f}")
        print(
            "  average ambulance latency: "
            f"{summary['avg_ambulance_latency_ms']:.6f} ms"
        )
        print(
            "  95th percentile ambulance latency: "
            f"{summary['p95_ambulance_latency_ms']:.6f} ms"
        )
        print(
            "  ambulance SLA violation rate: "
            f"{summary['ambulance_sla_violation_rate']:.6f}"
        )
        print(
            "  average ordinary throughput: "
            f"{summary['avg_ordinary_throughput_mbps']:.6f} Mbps"
        )
        print(f"  average PRB utilization: {summary['avg_prb_utilization']:.6f}")
        print(f"  action counts: {summary['action_counts']}")

    STEP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_csv(STEP_LOG_PATH, step_logs)
    write_csv(SUMMARY_LOG_PATH, summary_logs)
    print(f"Saved step-level logs to {STEP_LOG_PATH}")
    print(f"Saved per-episode summary to {SUMMARY_LOG_PATH}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

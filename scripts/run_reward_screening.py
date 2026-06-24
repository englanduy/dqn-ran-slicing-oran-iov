from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default_config.yaml"
CONFIGURATIONS = {
    "R0": (0.0, 0.0),
    "R1": (1.0, 10.0),
    "R2": (2.0, 10.0),
    "R3": (1.0, 20.0),
}
REWARD_COMPONENTS = (
    "reward_latency_excess_sum",
    "reward_sla_violation_sum",
    "reward_ordinary_throughput_sum",
    "reward_resource_waste_sum",
    "reward_action_change_sum",
    "reward_ordinary_queue_sum",
    "reward_ordinary_overflow_sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen queue/overflow reward weights.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def create_config(root: Path, label: str, queue_weight: float, overflow_weight: float) -> Path:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["reward"]["w_ordinary_queue"] = queue_weight
    config["reward"]["w_ordinary_overflow"] = overflow_weight
    path = root / label / "config_used.yaml"
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def train_and_evaluate(root: Path, label: str, config_path: Path) -> None:
    directory = root / label
    run_name = f"no_embb_queue_penalty_{label.lower()}_30k_{root.name.split('_')[-1]}"
    model_path = directory / "model" / f"{run_name}.zip"
    run(
        [
            sys.executable,
            "agents/train_dqn.py",
            "--config-path",
            str(config_path),
            "--total-timesteps",
            "30000",
            "--seed",
            "42",
            "--run-name",
            run_name,
            "--model-path",
            str(model_path),
            "--tensorboard-log",
            str(directory / "tensorboard"),
            "--summary-path",
            str(directory / "training_summary.json"),
            "--monitor-path",
            str(directory / "training.monitor.csv"),
            "--episode-metrics-path",
            str(directory / "training_episode_metrics.csv"),
            "--git-diff-path",
            str(directory / "source_diff.patch"),
            "--fail-if-exists",
        ]
    )
    run(
        [
            sys.executable,
            "eval/evaluate_no_embb_experiment.py",
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
            "--output-dir",
            str(directory / "evaluation"),
            "--episodes",
            "30",
            "--base-seed",
            "10000",
            "--monitor-path",
            str(directory / "training.monitor.csv"),
            "--quiet",
        ]
    )


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for label, (queue_weight, overflow_weight) in CONFIGURATIONS.items():
        directory = root / label / "evaluation"
        episodes = pd.read_csv(directory / "evaluation_episodes.csv")
        steps = pd.read_csv(directory / "evaluation_steps.csv")
        dqn_episodes = episodes[episodes["policy"] == "DQN"]
        dqn_steps = steps[steps["policy"] == "DQN"]
        guaranteed = episodes[episodes["policy"] == "Guaranteed Load-based"]
        total_offered = float(dqn_steps["ordinary_arrival_mbit"].sum())
        total_overflow = float(dqn_steps["ordinary_overflow_mbit"].sum())
        target_eligible = int(dqn_steps["ordinary_target_eligible"].sum())
        target_violations = int(dqn_steps["ordinary_target_violation"].sum())
        action_rates = dqn_steps["action"].value_counts(normalize=True)
        row: dict[str, Any] = {
            "configuration": label,
            "w_ordinary_queue": queue_weight,
            "w_ordinary_overflow": overflow_weight,
            "episode_reward_mean": float(dqn_episodes["episode_reward"].mean()),
            "episode_reward_std": float(dqn_episodes["episode_reward"].std(ddof=1)),
            "ambulance_latency_ms_mean": float(
                dqn_episodes["avg_ambulance_latency_ms"].mean()
            ),
            "ambulance_latency_ms_std": float(
                dqn_episodes["avg_ambulance_latency_ms"].std(ddof=1)
            ),
            "ambulance_p95_ms_mean": float(
                dqn_episodes["p95_ambulance_latency_ms"].mean()
            ),
            "ambulance_p95_ms_std": float(
                dqn_episodes["p95_ambulance_latency_ms"].std(ddof=1)
            ),
            "ambulance_sla_mean": float(
                dqn_episodes["ambulance_sla_violation_rate"].mean()
            ),
            "ambulance_sla_std": float(
                dqn_episodes["ambulance_sla_violation_rate"].std(ddof=1)
            ),
            "ordinary_demand_satisfaction": float(
                dqn_steps["ordinary_demand_satisfaction"].mean()
            ),
            "ordinary_conditional_target_violation": (
                target_violations / target_eligible if target_eligible else 0.0
            ),
            "ordinary_queue_mean_mbit": float(
                dqn_episodes["avg_ordinary_queue_mbit"].mean()
            ),
            "ordinary_queue_p95_mbit": float(
                dqn_episodes["p95_ordinary_queue_mbit"].mean()
            ),
            "ordinary_queue_max_mbit": float(
                dqn_episodes["max_ordinary_queue_mbit"].max()
            ),
            "ordinary_cap_hit_rate": float(
                dqn_steps["ordinary_queue_cap_hit"].mean()
            ),
            "ordinary_overflow_mbit": total_overflow,
            "ordinary_offered_mbit": total_offered,
            "ordinary_overflow_fraction": (
                total_overflow / total_offered if total_offered else 0.0
            ),
            "ordinary_throughput_emergency_mbps": float(
                dqn_steps.loc[
                    dqn_steps["ambulance_emergency"].astype(bool),
                    "ordinary_throughput_mbps",
                ].mean()
            ),
            "guaranteed_p95_ambulance_ms": float(
                guaranteed["p95_ambulance_latency_ms"].mean()
            ),
            "guaranteed_ambulance_sla": float(
                guaranteed["ambulance_sla_violation_rate"].mean()
            ),
            "unique_actions": int(dqn_steps["action"].nunique()),
            "max_action_rate": float(action_rates.max()),
        }
        for component in REWARD_COMPONENTS:
            row[f"{component}_mean"] = float(dqn_episodes[component].mean())
            row[f"{component}_std"] = float(dqn_episodes[component].std(ddof=1))
        row["passes_overflow"] = row["ordinary_overflow_fraction"] <= 0.0005
        row["passes_cap_hit"] = row["ordinary_cap_hit_rate"] <= 0.001
        row["passes_demand_satisfaction"] = (
            row["ordinary_demand_satisfaction"] >= 0.99
        )
        row["passes_target_violation"] = (
            row["ordinary_conditional_target_violation"] <= 0.05
        )
        row["passes_ambulance_p95"] = (
            row["ambulance_p95_ms_mean"] < row["guaranteed_p95_ambulance_ms"]
        )
        row["passes_ambulance_sla"] = (
            row["ambulance_sla_mean"] <= row["guaranteed_ambulance_sla"]
        )
        row["passes_action_diversity"] = (
            row["unique_actions"] > 1 and row["max_action_rate"] < 0.99
        )
        row["passes_all"] = all(
            row[key]
            for key in row
            if key.startswith("passes_") and key != "passes_all"
        )
        rows.append(row)

        for emergency, group in dqn_steps.groupby(
            dqn_steps["ambulance_emergency"].astype(bool)
        ):
            for action in range(8):
                action_rows.append(
                    {
                        "configuration": label,
                        "emergency": bool(emergency),
                        "action": action,
                        "ambulance_prb_fraction": (action + 1) / 10.0,
                        "rate": float((group["action"] == action).mean()),
                        "count": int((group["action"] == action).sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        root = PROJECT_ROOT / "results" / f"reward_screening_no_embb_{stamp}"
    else:
        root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    for label, weights in CONFIGURATIONS.items():
        config_path = create_config(root, label, *weights)
        train_and_evaluate(root, label, config_path)

    summary, actions = summarize(root)
    summary.to_csv(root / "screening_summary.csv", index=False)
    actions.to_csv(root / "screening_action_distribution.csv", index=False)
    passing = summary[summary["passes_all"]].copy()
    recommendation = None
    if not passing.empty:
        priority = {"R1": 0, "R2": 1, "R3": 2, "R0": 3}
        passing["penalty_weight_sum"] = (
            passing["w_ordinary_queue"] + passing["w_ordinary_overflow"]
        )
        passing["priority"] = passing["configuration"].map(priority)
        recommendation = str(
            passing.sort_values(["penalty_weight_sum", "priority"])
            .iloc[0]["configuration"]
        )
    payload = {
        "output_dir": str(root),
        "timesteps_per_configuration": 30_000,
        "training_seed": 42,
        "evaluation_seeds": list(range(10_000, 10_030)),
        "recommended_configuration": recommendation,
        "rows": summary.to_dict(orient="records"),
    }
    (root / "screening_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print("RECOMMENDATION", recommendation, flush=True)


if __name__ == "__main__":
    main()

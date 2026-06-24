from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHECKPOINTS = (25_000, 50_000, 75_000, 100_000)
COMPONENTS = (
    "reward_latency_excess_sum",
    "reward_sla_violation_sum",
    "reward_ordinary_throughput_sum",
    "reward_resource_waste_sum",
    "reward_action_change_sum",
    "reward_ordinary_queue_sum",
    "reward_ordinary_overflow_sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze demand-aware DQN checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoint_evaluations" / f"step_{step:06d}"


def normalized_action_entropy(step_df: pd.DataFrame) -> float:
    probabilities = (
        step_df["action"].value_counts().reindex(range(8), fill_value=0).to_numpy(float)
    )
    probabilities /= probabilities.sum()
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)) / np.log(8.0))


def longest_true_run(values: pd.Series) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def build_checkpoint_summary(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step in CHECKPOINTS:
        directory = checkpoint_dir(run_dir, step)
        episodes = pd.read_csv(directory / "evaluation_episodes.csv")
        steps = pd.read_csv(directory / "evaluation_steps.csv")
        dqn_episodes = episodes[episodes["policy"] == "DQN"]
        dqn_steps = steps[steps["policy"] == "DQN"]
        row: dict[str, Any] = {"checkpoint_timesteps": step, "n_episodes": len(dqn_episodes)}
        metrics = [
            "episode_reward",
            *COMPONENTS,
            "avg_ambulance_latency_ms",
            "p95_ambulance_latency_ms",
            "ambulance_sla_violation_rate",
            "avg_ordinary_demand_satisfaction",
            "avg_ordinary_throughput_emergency_mbps",
            "avg_ordinary_throughput_non_emergency_mbps",
            "ordinary_conditional_target_violation_rate",
            "ordinary_capacity_limited_rate",
            "avg_ordinary_queue_mbit",
            "p95_ordinary_queue_mbit",
            "max_ordinary_queue_mbit",
            "avg_prb_utilization",
            "avg_ambulance_prb_fraction_emergency",
            "avg_ambulance_prb_fraction_non_emergency",
            "action_switch_rate",
        ]
        for metric in metrics:
            row[f"{metric}_mean"] = float(dqn_episodes[metric].mean())
            row[f"{metric}_std"] = float(dqn_episodes[metric].std(ddof=1))
        total_offered = float(dqn_steps["ordinary_arrival_mbit"].sum())
        total_overflow = float(dqn_steps["ordinary_overflow_mbit"].sum())
        row["ordinary_cap_hit_rate"] = float(
            dqn_steps["ordinary_queue_cap_hit"].sum() / len(dqn_steps)
        )
        row["ordinary_overflow_mbit_total"] = total_overflow
        row["ordinary_offered_mbit_total"] = total_offered
        row["ordinary_overflow_fraction"] = (
            total_overflow / total_offered if total_offered else 0.0
        )
        row["episodes_with_overflow"] = int(
            (dqn_episodes["ordinary_overflow_mbit"] > 0.0).sum()
        )
        row["unique_actions"] = int(dqn_steps["action"].nunique())
        row["normalized_action_entropy"] = normalized_action_entropy(dqn_steps)
        for action in range(8):
            row[f"action_{action}_rate"] = float((dqn_steps["action"] == action).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def overflow_episode_analysis(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = run_dir / "final_model_evaluation"
    if not directory.exists():
        directory = checkpoint_dir(run_dir, 100_000)
    steps = pd.read_csv(directory / "evaluation_steps.csv")
    dqn = steps[steps["policy"] == "DQN"].copy()
    overflow_steps = dqn[dqn["ordinary_overflow_mbit"] > 0.0].copy()
    arrival_p99 = float(dqn["ordinary_offered_load_mbps"].quantile(0.99))
    rows = []
    overflow_episode_ids = sorted(overflow_steps["episode"].unique())
    for episode in overflow_episode_ids:
        episode_df = dqn[dqn["episode"] == episode]
        overflow_df = episode_df[episode_df["ordinary_overflow_mbit"] > 0.0]
        emergency = episode_df["ambulance_emergency"].astype(bool)
        row = {
            "episode": int(episode),
            "seed": int(episode_df["seed"].iloc[0]),
            "emergency_steps": int(emergency.sum()),
            "action_70_steps": int((episode_df["action"] == 6).sum()),
            "action_80_steps": int((episode_df["action"] == 7).sum()),
            "action_60_steps": int((episode_df["action"] == 5).sum()),
            "action_70_or_80_emergency_steps": int(
                ((episode_df["action"] >= 6) & emergency).sum()
            ),
            "ordinary_offered_load_mbps_mean": float(
                episode_df["ordinary_offered_load_mbps"].mean()
            ),
            "ordinary_offered_load_mbps_p95": float(
                episode_df["ordinary_offered_load_mbps"].quantile(0.95)
            ),
            "ordinary_offered_load_mbps_max": float(
                episode_df["ordinary_offered_load_mbps"].max()
            ),
            "ordinary_capacity_mbps_mean": float(
                episode_df["ordinary_capacity_mbps"].mean()
            ),
            "ordinary_capacity_mbps_emergency_mean": float(
                episode_df.loc[emergency, "ordinary_capacity_mbps"].mean()
            ),
            "ordinary_capacity_mbps_min": float(
                episode_df["ordinary_capacity_mbps"].min()
            ),
            "ordinary_backlog_before_mbit_mean": float(
                episode_df["ordinary_queue_before_mbit"].mean()
            ),
            "ordinary_backlog_before_mbit_p95": float(
                episode_df["ordinary_queue_before_mbit"].quantile(0.95)
            ),
            "ordinary_backlog_before_mbit_max": float(
                episode_df["ordinary_queue_before_mbit"].max()
            ),
            "ordinary_demand_mbps_mean": float(episode_df["ordinary_demand_mbps"].mean()),
            "ordinary_demand_mbps_max": float(episode_df["ordinary_demand_mbps"].max()),
            "capacity_limited_steps": int(episode_df["ordinary_capacity_limited"].sum()),
            "cap_hit_steps": int(episode_df["ordinary_queue_cap_hit"].sum()),
            "overflow_steps": int(len(overflow_df)),
            "longest_action_60_70_80_run": longest_true_run(episode_df["action"] >= 5),
            "longest_nonzero_queue_run": longest_true_run(
                episode_df["ordinary_queue_after_mbit"] > 0.0
            ),
            "overflow_mbit": float(episode_df["ordinary_overflow_mbit"].sum()),
            "offered_mbit": float(episode_df["ordinary_arrival_mbit"].sum()),
            "overflow_fraction": float(
                episode_df["ordinary_overflow_mbit"].sum()
                / episode_df["ordinary_arrival_mbit"].sum()
            ),
            "overflow_steps_in_emergency_rate": float(
                overflow_df["ambulance_emergency"].astype(bool).mean()
            ),
            "overflow_steps_action_70_or_80_rate": float(
                (overflow_df["action"] >= 6).mean()
            ),
            "overflow_steps_extreme_arrival_rate": float(
                (overflow_df["ordinary_offered_load_mbps"] >= arrival_p99).mean()
            ),
            "global_arrival_p99_mbps": arrival_p99,
        }
        rows.append(row)
    trajectories = dqn[dqn["episode"].isin(overflow_episode_ids)].copy()
    return pd.DataFrame(rows), overflow_steps, trajectories


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = build_checkpoint_summary(run_dir)
    overflow_episodes, overflow_steps, overflow_trajectories = (
        overflow_episode_analysis(run_dir)
    )
    summary.to_csv(run_dir / "checkpoint_trend_summary.csv", index=False)
    overflow_episodes.to_csv(run_dir / "final_overflow_episode_analysis.csv", index=False)
    overflow_steps.to_csv(run_dir / "final_overflow_steps.csv", index=False)
    overflow_trajectories.to_csv(
        run_dir / "final_overflow_episode_trajectories.csv", index=False
    )
    payload = {
        "main_checkpoint": 100_000,
        "checkpoint_selection_used": False,
        "checkpoint_rows": summary.to_dict(orient="records"),
        "overflow_episode_rows": overflow_episodes.to_dict(orient="records"),
    }
    (run_dir / "checkpoint_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nFinal overflow episodes")
    print(overflow_episodes.to_string(index=False) if not overflow_episodes.empty else "None")


if __name__ == "__main__":
    main()

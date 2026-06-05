from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
AGGREGATE_METRICS = [
    "total_reward",
    "avg_ambulance_latency_ms",
    "p95_ambulance_latency_ms",
    "sla_violation_rate",
    "avg_ordinary_throughput_mbps",
    "ordinary_throughput_deficit_rate",
    "avg_prb_utilization",
]


def parse_args() -> argparse.Namespace:
    """Parse optional comparison inputs."""
    parser = argparse.ArgumentParser(
        description="Compare baseline summaries with a selected DQN summary."
    )
    parser.add_argument(
        "--dqn-summary-pattern",
        default="results/dqn_summary_*.csv",
        help=(
            "Glob pattern for DQN summary files. The latest matching file is used. "
            'Example: "results/dqn_300k_summary_*.csv".'
        ),
    )
    return parser.parse_args()


def latest_csv(pattern: str) -> Path:
    """Return the newest CSV matching pattern."""
    pattern_path = Path(pattern)
    search_pattern = pattern if pattern_path.is_absolute() else str(PROJECT_ROOT / pattern)
    matches = sorted(
        (Path(path) for path in glob.glob(search_pattern)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return matches[0]


def load_episode_summaries(dqn_summary_pattern: str) -> pd.DataFrame:
    """Load the newest baseline and DQN summaries and combine them."""
    baseline_path = latest_csv("results/baseline_summary_*.csv")
    dqn_path = latest_csv(dqn_summary_pattern)

    baseline_df = pd.read_csv(baseline_path)
    dqn_df = pd.read_csv(dqn_path)

    # The selected DQN run is labeled explicitly for comparison tables.
    dqn_df["policy"] = "DQN-300k"

    combined_df = pd.concat([baseline_df, dqn_df], ignore_index=True, sort=False)
    combined_df = normalize_metric_columns(combined_df)

    print(f"Loaded baseline summary: {baseline_path}")
    print(f"Loaded DQN summary: {dqn_path}")
    return combined_df


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact aggregate metric names expected by this comparison."""
    if (
        "sla_violation_rate" not in df.columns
        and "ambulance_sla_violation_rate" in df.columns
    ):
        df["sla_violation_rate"] = df["ambulance_sla_violation_rate"]
    return df


def aggregate_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation for each metric by policy."""
    missing = [metric for metric in AGGREGATE_METRICS if metric not in df.columns]
    if missing:
        raise KeyError(f"Missing required metric columns: {missing}")

    aggregate = df.groupby("policy", sort=True)[AGGREGATE_METRICS].agg(["mean", "std"])
    aggregate.columns = [
        f"{metric}_{stat}" for metric, stat in aggregate.columns.to_flat_index()
    ]
    return aggregate.reset_index()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = RESULTS_DIR / f"combined_episode_results_{timestamp}.csv"
    aggregate_path = RESULTS_DIR / f"aggregate_comparison_{timestamp}.csv"

    combined_df = load_episode_summaries(args.dqn_summary_pattern)
    aggregate_df = aggregate_by_policy(combined_df)

    combined_df.to_csv(combined_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("\nAggregate comparison")
    print(aggregate_df.to_string(index=False))
    print(f"\nSaved combined episode results to {combined_path}")
    print(f"Saved aggregate comparison to {aggregate_path}")


if __name__ == "__main__":
    main()

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
POLICY_ORDER = ["Static", "Guaranteed Load-based", "DQN"]
AGGREGATE_METRICS = [
    "total_reward",
    "avg_ambulance_latency_ms",
    "p95_ambulance_latency_ms",
    "sla_violation_rate",
    "qos_satisfaction_rate",
    "avg_ordinary_throughput_mbps",
    "ordinary_throughput_deficit_rate",
    "avg_prb_utilization",
]
POLICY_NAME_MAP = {
    "StaticPolicy": "Static",
    "static": "Static",
    "Static": "Static",
    "GuaranteedLoadBasedPolicy": "Guaranteed Load-based",
    "guaranteed_load_based": "Guaranteed Load-based",
    "Guaranteed Load-based": "Guaranteed Load-based",
    "LoadBasedPolicy": "Load-based",
    "load_based": "Load-based",
    "Load-based": "Load-based",
    "DQN": "DQN",
}


def parse_args() -> argparse.Namespace:
    """Parse input summary patterns for the final comparison table."""
    parser = argparse.ArgumentParser(
        description="Compare final DQN results with report baselines."
    )
    parser.add_argument(
        "--baseline-summary-pattern",
        default="results/report_baselines_emergency_obs_summary_*.csv",
        help="Glob pattern for baseline summary CSV files.",
    )
    parser.add_argument(
        "--dqn-summary-pattern",
        default="results/dqn_summary_*.csv",
        help="Glob pattern for DQN summary CSV files.",
    )
    parser.add_argument(
        "--dqn-policy-name",
        default="DQN",
        help="Policy label assigned to the selected DQN summary.",
    )
    return parser.parse_args()


def latest_csv(pattern: str) -> Path:
    """Return the latest file matching a relative or absolute glob pattern."""
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


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric aliases produced by older evaluators."""
    normalized_df = df.copy()
    if "ambulance_sla_violation_rate" in normalized_df.columns:
        if "sla_violation_rate" not in normalized_df.columns:
            normalized_df["sla_violation_rate"] = normalized_df[
                "ambulance_sla_violation_rate"
            ]
        else:
            normalized_df["sla_violation_rate"] = normalized_df[
                "sla_violation_rate"
            ].fillna(normalized_df["ambulance_sla_violation_rate"])

    if "ambulance_qos_satisfaction_rate" in normalized_df.columns:
        if "qos_satisfaction_rate" not in normalized_df.columns:
            normalized_df["qos_satisfaction_rate"] = normalized_df[
                "ambulance_qos_satisfaction_rate"
            ]
        else:
            normalized_df["qos_satisfaction_rate"] = normalized_df[
                "qos_satisfaction_rate"
            ].fillna(normalized_df["ambulance_qos_satisfaction_rate"])
    return normalized_df


def normalize_baseline_policy_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map baseline implementation names to report display names."""
    normalized_df = df.copy()
    if "policy" not in normalized_df.columns:
        raise KeyError("Baseline summary must contain a policy column.")
    normalized_df["policy"] = normalized_df["policy"].astype(str).map(
        lambda name: POLICY_NAME_MAP.get(name, name)
    )
    return normalized_df


def load_final_results(
    baseline_summary_pattern: str,
    dqn_summary_pattern: str,
    dqn_policy_name: str,
) -> pd.DataFrame:
    """Load latest baseline and DQN summaries for the final comparison."""
    baseline_path = latest_csv(baseline_summary_pattern)
    dqn_path = latest_csv(dqn_summary_pattern)

    baseline_df_raw = pd.read_csv(baseline_path)
    dqn_df = pd.read_csv(dqn_path)
    print(f"Loaded baseline summary: {baseline_path}")
    print(f"Loaded DQN summary: {dqn_path}")
    print(
        "Baseline policy names before normalization:",
        sorted(baseline_df_raw["policy"].astype(str).unique())
        if "policy" in baseline_df_raw.columns
        else "missing policy column",
    )
    print(
        "DQN policy names before normalization:",
        sorted(dqn_df["policy"].astype(str).unique())
        if "policy" in dqn_df.columns
        else "missing policy column",
    )

    baseline_df = normalize_metric_columns(
        normalize_baseline_policy_names(baseline_df_raw)
    )
    dqn_df["policy"] = dqn_policy_name
    dqn_df["policy"] = dqn_df["policy"].astype(str).map(
        lambda name: POLICY_NAME_MAP.get(name, name)
    )
    dqn_df = normalize_metric_columns(dqn_df)

    combined_df = pd.concat([baseline_df, dqn_df], ignore_index=True, sort=False)
    combined_df = normalize_metric_columns(combined_df)
    combined_df["policy"] = combined_df["policy"].astype(str)
    combined_df = combined_df[combined_df["policy"].isin(POLICY_ORDER)].copy()
    combined_df["policy"] = pd.Categorical(
        combined_df["policy"],
        categories=POLICY_ORDER,
        ordered=True,
    )
    combined_df = combined_df.sort_values(["policy", "episode"]).reset_index(drop=True)
    combined_df["policy"] = combined_df["policy"].astype(str)

    print("Final policy list after filtering:", list(combined_df["policy"].unique()))
    return combined_df


def aggregate_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation for final report metrics."""
    missing = [metric for metric in AGGREGATE_METRICS if metric not in df.columns]
    if missing:
        available = ", ".join(str(column) for column in df.columns)
        raise KeyError(f"Missing required metric columns: {missing}. Available: {available}")

    numeric_df = df.copy()
    for metric in AGGREGATE_METRICS:
        numeric_df[metric] = pd.to_numeric(numeric_df[metric], errors="coerce")

    aggregate = numeric_df.groupby("policy", sort=False)[AGGREGATE_METRICS].agg(
        ["mean", "std"]
    )
    aggregate.columns = [
        f"{metric}_{stat}" for metric, stat in aggregate.columns.to_flat_index()
    ]
    return aggregate.reset_index()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = RESULTS_DIR / f"final_combined_episode_results_{timestamp}.csv"
    aggregate_path = RESULTS_DIR / f"final_aggregate_comparison_{timestamp}.csv"

    combined_df = load_final_results(
        baseline_summary_pattern=args.baseline_summary_pattern,
        dqn_summary_pattern=args.dqn_summary_pattern,
        dqn_policy_name=args.dqn_policy_name,
    )
    aggregate_df = aggregate_by_policy(combined_df)

    combined_df.to_csv(combined_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("\nFinal aggregate comparison")
    print(aggregate_df.to_string(index=False))
    print(f"\nSaved combined episode results to {combined_path}")
    print(f"Saved aggregate comparison to {aggregate_path}")


if __name__ == "__main__":
    main()

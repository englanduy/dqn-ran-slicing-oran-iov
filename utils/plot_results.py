from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures_report"
POLICY_DISPLAY_NAMES = {
    "StaticPolicy": "Static",
    "PriorityPolicy": "Priority",
    "LoadBasedPolicy": "Load-based",
    "GreedySLAPolicy": "Greedy SLA",
    "RandomPolicy": "Random",
    "DQN": "DQN",
}
POLICY_ORDER = ["Static", "Priority", "Load-based", "Greedy SLA", "DQN", "Random"]

FIGURE_SPECS = [
    {
        "filename": "fig_sla_violation_rate.png",
        "mean_column": "sla_violation_rate_mean",
        "std_column": "sla_violation_rate_std",
        "ylabel": "SLA violation rate (%)",
        "title": "Ambulance SLA violation rate",
        "percent": True,
    },
    {
        "filename": "fig_avg_ambulance_latency.png",
        "mean_column": "avg_ambulance_latency_ms_mean",
        "std_column": "avg_ambulance_latency_ms_std",
        "ylabel": "Latency (ms)",
        "title": "Average ambulance latency",
        "sla_line": True,
    },
    {
        "filename": "fig_p95_ambulance_latency.png",
        "mean_column": "p95_ambulance_latency_ms_mean",
        "std_column": "p95_ambulance_latency_ms_std",
        "ylabel": "Latency (ms)",
        "title": "95th percentile ambulance latency",
        "sla_line": True,
    },
    {
        "filename": "fig_ordinary_throughput.png",
        "mean_column": "avg_ordinary_throughput_mbps_mean",
        "std_column": "avg_ordinary_throughput_mbps_std",
        "ylabel": "Throughput (Mbps)",
        "title": "Average ordinary throughput",
    },
    {
        "filename": "fig_ordinary_throughput_deficit_rate.png",
        "mean_column": "ordinary_throughput_deficit_rate_mean",
        "std_column": "ordinary_throughput_deficit_rate_std",
        "ylabel": "Throughput deficit rate (%)",
        "title": "Ordinary throughput deficit rate",
        "percent": True,
    },
    {
        "filename": "fig_total_reward.png",
        "mean_column": "total_reward_mean",
        "std_column": "total_reward_std",
        "ylabel": "Total reward",
        "title": "Episode total reward",
    },
    {
        "filename": "fig_prb_utilization.png",
        "mean_column": "avg_prb_utilization_mean",
        "std_column": "avg_prb_utilization_std",
        "ylabel": "PRB utilization",
        "title": "Average PRB utilization",
    },
]


def latest_csv(pattern: str) -> Path:
    """Return the newest CSV matching pattern in results/."""
    matches = sorted(
        RESULTS_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No files found for results/{pattern}")
    return matches[0]


def readable_policy_names(policies: pd.Series) -> pd.Series:
    """Make policy labels easier to read on chart axes."""
    return policies.astype(str).map(lambda policy: POLICY_DISPLAY_NAMES.get(policy, policy))


def order_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Apply report display names and keep policies in a consistent order."""
    ordered_df = df.copy()
    ordered_df["policy"] = readable_policy_names(ordered_df["policy"])
    ordered_df["policy"] = pd.Categorical(
        ordered_df["policy"],
        categories=POLICY_ORDER,
        ordered=True,
    )
    return ordered_df.sort_values("policy").reset_index(drop=True)


def plot_metric(aggregate_df: pd.DataFrame, spec: dict[str, str]) -> None:
    """Create and save one metric bar chart."""
    mean_column = spec["mean_column"]
    std_column = spec["std_column"]
    if mean_column not in aggregate_df.columns:
        raise KeyError(f"Missing required aggregate column: {mean_column}")

    policies = aggregate_df["policy"].astype(str)
    scale = 100.0 if spec.get("percent", False) else 1.0
    means = aggregate_df[mean_column] * scale
    yerr = aggregate_df[std_column] * scale if std_column in aggregate_df.columns else None

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(policies, means, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set_xlabel("Policy")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    if spec.get("sla_line", False):
        ax.axhline(100.0, linestyle="--", linewidth=1.2, label="SLA threshold")
        ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    output_path = FIGURES_DIR / spec["filename"]
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    aggregate_path = latest_csv("aggregate_comparison_*.csv")
    combined_path = latest_csv("combined_episode_results_*.csv")

    aggregate_df = order_policies(pd.read_csv(aggregate_path))
    # Loaded for traceability alongside the aggregate input used by the figures.
    combined_df = pd.read_csv(combined_path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded aggregate comparison: {aggregate_path}")
    print(f"Loaded combined episode results: {combined_path} ({len(combined_df)} rows)")

    for spec in FIGURE_SPECS:
        plot_metric(aggregate_df, spec)


if __name__ == "__main__":
    main()

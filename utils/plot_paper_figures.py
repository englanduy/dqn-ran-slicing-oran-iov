from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = RESULTS_DIR / "figures_paper"

POLICY_DISPLAY_NAMES = {
    "StaticPolicy": "Static",
    "PriorityPolicy": "Priority",
    "LoadBasedPolicy": "Load-based",
    "GreedySLAPolicy": "Greedy SLA",
    "DQN": "DQN",
    "DQN-300k": "DQN",
    "RandomPolicy": "Random",
}
POLICY_ORDER = ["Static", "Priority", "Load-based", "Greedy SLA", "DQN", "Random"]
ACTION_INDICES = np.arange(8)
ALPHA_LABELS = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%"]


def configure_matplotlib() -> None:
    """Apply a consistent publication-style matplotlib layout."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.size": 11,
        }
    )


def latest_file(pattern: str, root: Path = RESULTS_DIR) -> Path:
    """Return the newest file matching a glob pattern."""
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files found for {root / pattern}")
    return matches[0]


def clean_axes(ax: plt.Axes) -> None:
    """Use clean report-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a high-resolution paper figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def smooth_series(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Smooth a reward curve with a centered rolling mean."""
    if len(values) < 2:
        return values
    window = min(window, len(values))
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def find_tensorboard_scalar() -> tuple[np.ndarray, np.ndarray]:
    """Read DQN reward scalars from the latest TensorBoard event logs."""
    event_files = sorted(
        LOGS_DIR.rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under {LOGS_DIR}. "
            "Run DQN training before plotting the learning curve."
        )

    preferred_tags = [
        "rollout/ep_rew_mean",
        "episode/reward",
        "episode_reward",
        "train/episode_reward",
    ]

    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        tag = next((name for name in preferred_tags if name in scalar_tags), None)
        if tag is None and scalar_tags:
            tag = scalar_tags[0]
        if tag is None:
            continue

        events = accumulator.Scalars(tag)
        steps = np.asarray([event.step for event in events], dtype=float)
        rewards = np.asarray([event.value for event in events], dtype=float)
        if len(steps) > 0:
            print(f"Using TensorBoard scalar '{tag}' from {event_file}")
            return steps, rewards

    raise ValueError("No scalar data found in TensorBoard event files.")


def plot_learning_curve() -> None:
    """Plot raw and smoothed DQN episode reward against training timesteps."""
    steps, rewards = find_tensorboard_scalar()
    smoothed_rewards = smooth_series(rewards)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(steps, rewards, linewidth=0.8, alpha=0.35, label="Raw episode reward")
    ax.plot(steps, smoothed_rewards, linewidth=2.0, label="Smoothed episode reward")
    ax.axvline(300000, linestyle="--", linewidth=1.2, label="300000 timesteps")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Learning curve of DQN-based RAN slicing")
    ax.legend()
    clean_axes(ax)
    save_figure(fig, "fig_learning_curve_dqn.png")


def load_aggregate_results() -> pd.DataFrame:
    """Load and normalize the latest aggregate comparison CSV."""
    aggregate_path = latest_file("aggregate_comparison_*.csv")
    df = pd.read_csv(aggregate_path)
    print(f"Loaded aggregate comparison: {aggregate_path}")

    df["policy"] = df["policy"].astype(str).map(lambda name: POLICY_DISPLAY_NAMES.get(name, name))
    df = df[df["policy"].isin(POLICY_ORDER)].copy()
    df["_policy_rank"] = df["policy"].map({name: idx for idx, name in enumerate(POLICY_ORDER)})
    df = df.sort_values("_policy_rank").drop(columns="_policy_rank")

    metric_columns = [
        "avg_ambulance_latency_ms_mean",
        "avg_ambulance_latency_ms_std",
        "p95_ambulance_latency_ms_mean",
        "p95_ambulance_latency_ms_std",
        "sla_violation_rate_mean",
        "sla_violation_rate_std",
        "avg_ordinary_throughput_mbps_mean",
        "avg_ordinary_throughput_mbps_std",
    ]
    for column in metric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a readable error if a required aggregate column is missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(str(column) for column in df.columns)
        raise KeyError(f"Missing required columns {missing}. Available columns: {available}")


def plot_kpi_bar(
    df: pd.DataFrame,
    filename: str,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    percent: bool = False,
    reference_line: float | None = None,
    reference_label: str | None = None,
) -> None:
    """Plot one KPI comparison bar chart with error bars."""
    require_columns(df, [mean_column])
    plot_df = df.dropna(subset=[mean_column]).copy()
    scale = 100.0 if percent else 1.0
    policies = plot_df["policy"].to_numpy()
    means = (plot_df[mean_column] * scale).to_numpy(dtype=float)
    yerr = None
    if std_column in plot_df.columns:
        yerr = (plot_df[std_column] * scale).to_numpy(dtype=float)
        yerr = np.nan_to_num(yerr, nan=0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(policies, means, yerr=yerr, capsize=4)
    if reference_line is not None:
        ax.axhline(reference_line, linestyle="--", linewidth=1.2, label=reference_label)
        ax.legend()
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    clean_axes(ax)
    save_figure(fig, filename)


def plot_kpi_figures() -> None:
    """Generate paper KPI comparison figures from aggregate results."""
    df = load_aggregate_results()
    plot_kpi_bar(
        df,
        "fig_avg_ambulance_latency.png",
        "avg_ambulance_latency_ms_mean",
        "avg_ambulance_latency_ms_std",
        "Average ambulance latency (ms)",
        "Average ambulance latency",
        reference_line=100.0,
        reference_label="SLA threshold",
    )
    plot_kpi_bar(
        df,
        "fig_p95_ambulance_latency.png",
        "p95_ambulance_latency_ms_mean",
        "p95_ambulance_latency_ms_std",
        "95th percentile latency (ms)",
        "95th percentile ambulance latency",
        reference_line=100.0,
        reference_label="SLA threshold",
    )
    plot_kpi_bar(
        df,
        "fig_sla_violation_rate.png",
        "sla_violation_rate_mean",
        "sla_violation_rate_std",
        "SLA violation rate (%)",
        "Ambulance SLA violation rate",
        percent=True,
    )
    plot_kpi_bar(
        df,
        "fig_avg_ordinary_throughput.png",
        "avg_ordinary_throughput_mbps_mean",
        "avg_ordinary_throughput_mbps_std",
        "Average ordinary throughput (Mbps)",
        "Average ordinary throughput",
        reference_line=8.0,
        reference_label="Throughput target",
    )


def action_counts(df: pd.DataFrame) -> pd.Series:
    """Count actions 0 through 7, including unselected actions."""
    return df["action"].value_counts().reindex(ACTION_INDICES, fill_value=0)


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Parse boolean CSV values without treating every non-empty string as True."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def plot_dqn_action_by_emergency() -> None:
    """Plot DQN action distribution under emergency and non-emergency states."""
    steps_path = latest_file("dqn_300k_steps_*.csv")
    df = pd.read_csv(steps_path)
    print(f"Loaded DQN 300k steps: {steps_path}")
    require_columns(df, ["action", "ambulance_emergency"])

    df["ambulance_emergency"] = parse_boolean_series(df["ambulance_emergency"])
    non_emergency_counts = action_counts(df[~df["ambulance_emergency"]]).to_numpy()
    emergency_counts = action_counts(df[df["ambulance_emergency"]]).to_numpy()

    labels = [f"{idx}\n{alpha}" for idx, alpha in zip(ACTION_INDICES, ALPHA_LABELS)]
    width = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(ACTION_INDICES - width / 2, non_emergency_counts, width=width, label="Non-emergency")
    ax.bar(ACTION_INDICES + width / 2, emergency_counts, width=width, label="Emergency")
    ax.set_xticks(ACTION_INDICES)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action index\nAmbulance PRB ratio")
    ax.set_ylabel("Action count")
    ax.set_title("DQN action distribution by ambulance emergency state")
    ax.legend()
    clean_axes(ax)
    save_figure(fig, "fig_dqn_action_by_emergency.png")


def main() -> None:
    configure_matplotlib()
    plot_learning_curve()
    plot_kpi_figures()
    plot_dqn_action_by_emergency()


if __name__ == "__main__":
    main()

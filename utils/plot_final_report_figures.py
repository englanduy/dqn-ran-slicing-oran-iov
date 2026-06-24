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
CLIPPED_DQN_RUN_NAME = "dqn_300k_emergency_obs_clipped_reward"
LOG_DIR = PROJECT_ROOT / "logs" / CLIPPED_DQN_RUN_NAME
FIGURES_DIR = RESULTS_DIR / "final_figures_report_clipped_reward"
POLICY_ORDER = ["Static", "Guaranteed Load-based", "DQN"]
POLICY_NAME_MAP = {
    "StaticPolicy": "Static",
    "static": "Static",
    "Static": "Static",
    "GuaranteedLoadBasedPolicy": "Guaranteed Load-based",
    "guaranteed_load_based": "Guaranteed Load-based",
    "Guaranteed Load-based": "Guaranteed Load-based",
    "DQN": "DQN",
}
ACTION_INDICES = np.arange(8)
ALPHA_LABELS = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%"]
TB_REWARD_TAG = "rollout/ep_rew_mean"
SMOOTHING_WINDOW = 10


def configure_matplotlib() -> None:
    """Apply a clean publication-style matplotlib theme."""
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
    """Return the newest file matching pattern."""
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files found for {root / pattern}")
    return matches[0]


def clean_axes(ax: plt.Axes) -> None:
    """Use clean report axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure at 300 dpi."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def load_final_aggregate() -> pd.DataFrame:
    """Load the latest final aggregate comparison table."""
    aggregate_path = latest_file("final_aggregate_comparison_*.csv")
    combined_path = latest_file("final_combined_episode_results_*.csv")
    aggregate_df = pd.read_csv(aggregate_path)
    combined_df = pd.read_csv(combined_path)

    aggregate_df["policy"] = aggregate_df["policy"].astype(str).map(
        lambda name: POLICY_NAME_MAP.get(name, name)
    )
    aggregate_df = aggregate_df[aggregate_df["policy"].isin(POLICY_ORDER)].copy()
    aggregate_df["policy"] = pd.Categorical(
        aggregate_df["policy"],
        categories=POLICY_ORDER,
        ordered=True,
    )
    aggregate_df = aggregate_df.sort_values("policy").reset_index(drop=True)
    aggregate_df["policy"] = aggregate_df["policy"].astype(str)

    for column in aggregate_df.columns:
        if column != "policy":
            aggregate_df[column] = pd.to_numeric(aggregate_df[column], errors="coerce")

    print(f"Loaded final aggregate comparison: {aggregate_path}")
    print(f"Loaded final combined episode results: {combined_path} ({len(combined_df)} rows)")
    return aggregate_df


def require_column(df: pd.DataFrame, column: str) -> None:
    """Raise a readable error if a needed column is missing."""
    if column not in df.columns:
        available = ", ".join(str(name) for name in df.columns)
        raise KeyError(f"Missing required column: {column}. Available columns: {available}")


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
) -> Path:
    """Plot a KPI comparison bar chart with error bars."""
    require_column(df, mean_column)
    plot_df = df.dropna(subset=[mean_column]).copy()
    scale = 100.0 if percent else 1.0
    policies = (
        plot_df["policy"]
        .astype(str)
        .replace({"Guaranteed Load-based": "Guaranteed\nLoad-based"})
        .to_numpy()
    )
    means = (plot_df[mean_column] * scale).to_numpy(dtype=float)
    yerr = None
    if std_column in plot_df.columns:
        yerr = (plot_df[std_column] * scale).to_numpy(dtype=float)
        yerr = np.nan_to_num(yerr, nan=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.bar(policies, means, yerr=yerr, capsize=4)
    if reference_line is not None:
        ax.axhline(reference_line, linestyle="--", linewidth=1.2, label=reference_label)
        ax.legend()
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    clean_axes(ax)
    return save_figure(fig, filename)


def plot_kpi_figures() -> list[Path]:
    """Generate the final KPI comparison figures."""
    aggregate_df = load_final_aggregate()
    output_paths = []
    output_paths.append(
        plot_kpi_bar(
            aggregate_df,
            "fig_avg_ambulance_latency.png",
            "avg_ambulance_latency_ms_mean",
            "avg_ambulance_latency_ms_std",
            "Average ambulance latency (ms)",
            "Average ambulance latency",
            reference_line=100.0,
            reference_label="SLA threshold",
        )
    )
    output_paths.append(
        plot_kpi_bar(
            aggregate_df,
            "fig_p95_ambulance_latency.png",
            "p95_ambulance_latency_ms_mean",
            "p95_ambulance_latency_ms_std",
            "95th percentile latency (ms)",
            "95th percentile ambulance latency",
            reference_line=100.0,
            reference_label="SLA threshold",
        )
    )
    output_paths.append(
        plot_kpi_bar(
            aggregate_df,
            "fig_sla_violation_rate.png",
            "sla_violation_rate_mean",
            "sla_violation_rate_std",
            "SLA violation rate (%)",
            "Ambulance SLA violation rate",
            percent=True,
        )
    )
    output_paths.append(
        plot_kpi_bar(
            aggregate_df,
            "fig_avg_ordinary_throughput.png",
            "avg_ordinary_throughput_mbps_mean",
            "avg_ordinary_throughput_mbps_std",
            "Average ordinary throughput (Mbps)",
            "Average ordinary throughput",
            reference_line=8.0,
            reference_label="Throughput target",
        )
    )
    return output_paths


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse CSV boolean-like values robustly."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def action_counts(df: pd.DataFrame) -> pd.Series:
    """Count all DQN actions, including unselected actions."""
    return df["action"].value_counts().reindex(ACTION_INDICES, fill_value=0)


def plot_dqn_action_by_emergency() -> Path:
    """Plot DQN action distribution under non-emergency and emergency states."""
    steps_path = latest_file(f"{CLIPPED_DQN_RUN_NAME}_steps_*.csv")
    steps_df = pd.read_csv(steps_path)
    require_column(steps_df, "action")
    require_column(steps_df, "ambulance_emergency")
    steps_df["action"] = pd.to_numeric(steps_df["action"], errors="coerce")
    steps_df["ambulance_emergency"] = parse_bool_series(steps_df["ambulance_emergency"])

    non_emergency_counts = action_counts(steps_df[~steps_df["ambulance_emergency"]]).to_numpy()
    emergency_counts = action_counts(steps_df[steps_df["ambulance_emergency"]]).to_numpy()

    labels = [f"{idx}\n{alpha}" for idx, alpha in zip(ACTION_INDICES, ALPHA_LABELS)]
    width = 0.4
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(ACTION_INDICES - width / 2, non_emergency_counts, width=width, label="Non-emergency")
    ax.bar(ACTION_INDICES + width / 2, emergency_counts, width=width, label="Emergency")
    ax.set_xticks(ACTION_INDICES)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action index\nAmbulance PRB ratio")
    ax.set_ylabel("Action count")
    ax.legend()
    clean_axes(ax)
    print(f"Loaded clipped-reward DQN step CSV: {steps_path}")
    return save_figure(fig, "fig_dqn_action_by_emergency.png")


def load_tensorboard_rewards() -> tuple[pd.DataFrame, list[Path]]:
    """Read rollout/ep_rew_mean from TensorBoard logs."""
    event_files = sorted(LOG_DIR.rglob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {LOG_DIR}")

    rows = []
    available_tags = set()
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        available_tags.update(scalar_tags)
        if TB_REWARD_TAG not in scalar_tags:
            continue
        for event in accumulator.Scalars(TB_REWARD_TAG):
            rows.append({"step": event.step, "reward": event.value})

    if not rows:
        tags = ", ".join(sorted(available_tags))
        raise KeyError(f"Missing TensorBoard scalar '{TB_REWARD_TAG}'. Available tags: {tags}")

    rewards_df = pd.DataFrame(rows).sort_values("step")
    rewards_df = rewards_df.drop_duplicates(subset="step", keep="last").reset_index(drop=True)
    rewards_df["reward_smooth"] = (
        rewards_df["reward"]
        .rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True)
        .mean()
    )
    print(f"Loaded TensorBoard rewards from {LOG_DIR} ({len(rewards_df)} points)")
    print("TensorBoard event files used:")
    for event_file in event_files:
        print(f"- {event_file}")
    return rewards_df, event_files


def plot_learning_curve() -> Path:
    """Plot smoothed DQN episode reward only."""
    rewards_df, _ = load_tensorboard_rewards()
    display_reward = rewards_df["reward_smooth"]
    y_min = float(display_reward.quantile(0.02))
    y_max = float(display_reward.quantile(0.98))
    margin = max(5.0, 0.1 * (y_max - y_min))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(rewards_df["step"], display_reward, linewidth=2.2, label="Smoothed episode reward")
    ax.set_xlim(0, 300000)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.axvline(300000, linestyle="--", linewidth=1.2, label="300000 timesteps")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.legend()
    clean_axes(ax)
    return save_figure(fig, "fig_learning_curve_dqn.png")


def main() -> None:
    configure_matplotlib()
    print(f"Using clipped-reward DQN run: {CLIPPED_DQN_RUN_NAME}")
    print(f"Using TensorBoard log directory: {LOG_DIR}")
    output_paths = []
    output_paths.extend(plot_kpi_figures())
    output_paths.append(plot_dqn_action_by_emergency())
    output_paths.append(plot_learning_curve())
    print("\nGenerated figure paths:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

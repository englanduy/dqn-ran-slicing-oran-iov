from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures_report"
ACTION_INDICES = np.arange(8)
ALPHA_LABELS = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%"]


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


def action_counts(df: pd.DataFrame) -> pd.Series:
    """Count actions 0 through 7, including actions with zero selections."""
    return df["action"].value_counts().reindex(ACTION_INDICES, fill_value=0)


def set_action_axis(ax: plt.Axes) -> None:
    """Label each action by index and corresponding Ambulance PRB ratio."""
    labels = [f"{idx}\n{alpha}" for idx, alpha in zip(ACTION_INDICES, ALPHA_LABELS)]
    ax.set_xticks(ACTION_INDICES)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action index\nAmbulance PRB ratio")
    ax.set_ylabel("Count")


def plot_overall_distribution(df: pd.DataFrame) -> None:
    """Plot the overall deterministic DQN action distribution."""
    counts = action_counts(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ACTION_INDICES, counts.to_numpy())
    set_action_axis(ax)
    ax.set_title("DQN action distribution")
    fig.tight_layout()

    output_path = FIGURES_DIR / "fig_dqn_action_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_distribution_by_emergency(df: pd.DataFrame) -> None:
    """Plot DQN action counts for non-emergency and emergency states."""
    non_emergency = df[~df["ambulance_emergency"].astype(bool)]
    emergency = df[df["ambulance_emergency"].astype(bool)]

    non_emergency_counts = action_counts(non_emergency).to_numpy()
    emergency_counts = action_counts(emergency).to_numpy()

    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ACTION_INDICES - width / 2, non_emergency_counts, width=width, label="Non-emergency")
    ax.bar(ACTION_INDICES + width / 2, emergency_counts, width=width, label="Emergency")
    set_action_axis(ax)
    ax.set_title("DQN action distribution by ambulance emergency state")
    ax.legend()
    fig.tight_layout()

    output_path = FIGURES_DIR / "fig_dqn_action_distribution_by_emergency.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    dqn_steps_path = latest_csv("dqn_steps_*.csv")
    df = pd.read_csv(dqn_steps_path)

    if "action" not in df.columns:
        raise KeyError(f"Missing required column 'action' in {dqn_steps_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded DQN steps: {dqn_steps_path}")
    plot_overall_distribution(df)

    if "ambulance_emergency" in df.columns:
        plot_distribution_by_emergency(df)


if __name__ == "__main__":
    main()

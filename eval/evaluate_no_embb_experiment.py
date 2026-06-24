from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.baseline_policies import (  # noqa: E402
    GuaranteedLoadBasedPolicy,
    StaticPolicy,
)
from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402


POLICY_ORDER = ("DQN", "Static", "Guaranteed Load-based")
POLICY_COLORS = {
    "DQN": "#0072B2",
    "Static": "#999999",
    "Guaranteed Load-based": "#E69F00",
}
DISPLAY_NAMES = {
    "DQN": "DQN",
    "Static": "Static",
    "Guaranteed Load-based": "Guaranteed\nLoad-based",
}
METRIC_UNITS = {
    "episode_reward": "reward/episode",
    "mean_step_reward": "reward/step",
    "reward_latency_excess_sum": "reward/episode",
    "reward_sla_violation_sum": "reward/episode",
    "reward_ordinary_throughput_sum": "reward/episode",
    "reward_resource_waste_sum": "reward/episode",
    "reward_action_change_sum": "reward/episode",
    "reward_ordinary_queue_sum": "reward/episode",
    "reward_ordinary_overflow_sum": "reward/episode",
    "reward_component_error": "reward/episode",
    "avg_ambulance_latency_ms": "ms",
    "p95_ambulance_latency_ms": "ms",
    "ambulance_sla_violation_rate": "fraction",
    "ambulance_sla_violation_rate_present_or_emergency": "fraction",
    "ambulance_sla_violation_rate_when_present": "fraction",
    "ambulance_sla_violation_rate_when_emergency": "fraction",
    "avg_ambulance_throughput_mbps": "Mbit/s",
    "avg_ambulance_queue_mbit": "Mbit",
    "max_ambulance_queue_mbit": "Mbit",
    "ambulance_queue_cap_hits": "steps/episode",
    "ambulance_overflow_mbit": "Mbit/episode",
    "avg_ordinary_throughput_mbps": "Mbit/s",
    "avg_ordinary_throughput_emergency_mbps": "Mbit/s",
    "avg_ordinary_throughput_non_emergency_mbps": "Mbit/s",
    "avg_ordinary_demand_satisfaction": "fraction",
    "ordinary_conditional_target_violation_rate": "fraction",
    "ordinary_capacity_limited_rate": "fraction",
    "avg_ordinary_queue_mbit": "Mbit",
    "p95_ordinary_queue_mbit": "Mbit",
    "max_ordinary_queue_mbit": "Mbit",
    "ordinary_throughput_target_attainment_rate": "fraction",
    "ordinary_queue_cap_hits": "steps/episode",
    "ordinary_overflow_mbit": "Mbit/episode",
    "ordinary_overflow_fraction": "fraction",
    "avg_ordinary_queue_term": "fraction",
    "avg_ordinary_overflow_term": "fraction",
    "avg_prb_utilization": "fraction",
    "avg_ambulance_prb_fraction": "fraction",
    "avg_ambulance_prb_fraction_emergency": "fraction",
    "avg_ambulance_prb_fraction_non_emergency": "fraction",
    "action_switches": "switches/episode",
    "action_switch_rate": "fraction",
    "total_queue_cap_hits": "steps/episode",
}
for action_index in range(8):
    METRIC_UNITS[f"action_{action_index}_count"] = "selections/episode"
    METRIC_UNITS[f"action_{action_index}_rate"] = "fraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair no-eMBB evaluation for DQN and report baselines."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default_config.yaml",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--monitor-path", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def conditional_mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def validate_no_embb_config(config: dict[str, Any]) -> None:
    forbidden = ("embb", "surge")
    offending = [
        f"{section}.{key}"
        for section, values in config.items()
        if isinstance(values, dict)
        for key in values
        if any(term in key.lower() for term in forbidden)
    ]
    if offending:
        raise ValueError(f"Forbidden traffic keys found: {offending}")


def evaluate_episode(
    policy_name: str,
    policy: Any,
    env: RANSlicingEnv,
    episode: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    traffic_cfg = env.config["traffic"]
    packet_size = float(traffic_cfg["vehicle_packet_size_mbit"])
    target = env.ordinary_throughput_target
    step_rows: list[dict[str, Any]] = []
    ambulance_latencies_ms: list[float] = []
    ambulance_violations: list[int] = []
    violations_present_or_emergency: list[int] = []
    violations_when_present: list[int] = []
    violations_when_emergency: list[int] = []
    ambulance_throughputs: list[float] = []
    ambulance_queues: list[float] = []
    ordinary_throughputs: list[float] = []
    ordinary_throughputs_emergency: list[float] = []
    ordinary_throughputs_non_emergency: list[float] = []
    ordinary_demand_satisfaction: list[float] = []
    ordinary_queues: list[float] = []
    ordinary_target_hits: list[int] = []
    prb_utilizations: list[float] = []
    ambulance_prb_fractions: list[float] = []
    ambulance_prb_emergency: list[float] = []
    ambulance_prb_non_emergency: list[float] = []
    rewards: list[float] = []
    reward_components = {
        key: 0.0
        for key in (
            "reward_latency_excess",
            "reward_sla_violation",
            "reward_ordinary_throughput",
            "reward_resource_waste",
            "reward_action_change",
            "reward_ordinary_queue",
            "reward_ordinary_overflow",
        )
    }
    ordinary_capacity_limited_steps = 0
    ordinary_target_eligible_steps = 0
    ordinary_target_violations = 0
    ordinary_offered_mbit_total = 0.0
    action_counts = np.zeros(8, dtype=int)
    action_switches = 0
    previous_action: int | None = None
    ambulance_cap_hits = 0
    ordinary_cap_hits = 0
    ambulance_overflow_total = 0.0
    ordinary_overflow_total = 0.0
    ordinary_queue_terms: list[float] = []
    ordinary_overflow_terms: list[float] = []
    last_info: dict[str, Any] | None = None

    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        if policy_name == "DQN":
            action_array, _ = policy.predict(obs, deterministic=True)
            action = int(action_array)
        else:
            action = int(policy.select_action(obs, last_info))

        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

        if info["prb_ambulance"] + info["prb_ordinary"] != env.total_prb:
            raise AssertionError("Allocated PRBs do not sum to total_prb.")
        if not isinstance(info["prb_ambulance"], int) or not isinstance(
            info["prb_ordinary"], int
        ):
            raise AssertionError("PRB allocations must be integers.")

        finite_values = [
            *np.asarray(obs, dtype=float).tolist(),
            reward,
            info["ambulance_latency_s"],
            info["ambulance_queue_mbit"],
            info["ordinary_queue_mbit"],
            info["ambulance_capacity_mbps"],
            info["ordinary_capacity_mbps"],
            info["ordinary_throughput_mbps"],
            info["prb_utilization"],
        ]
        if not np.isfinite(finite_values).all():
            raise FloatingPointError(
                f"Non-finite evaluation value for {policy_name}, seed={seed}, step={step}."
            )

        packet_count = float(info["ordinary_arrival_mbit"]) / packet_size
        if not np.isclose(packet_count, round(packet_count), atol=1e-9):
            raise AssertionError("Ordinary arrival is not an integer number of vehicle packets.")

        ambulance_raw_queue = max(
            0.0,
            float(info["ambulance_queue_before_mbit"])
            + float(info["ambulance_arrival_mbit"])
            - float(info["ambulance_capacity_mbps"]) * env.delta_t,
        )
        ordinary_raw_queue = max(
            0.0,
            float(info["ordinary_queue_before_mbit"])
            + float(info["ordinary_arrival_mbit"])
            - float(info["ordinary_capacity_mbps"]) * env.delta_t,
        )
        ambulance_overflow = max(0.0, ambulance_raw_queue - env.q_ambulance_max)
        ordinary_overflow = max(0.0, ordinary_raw_queue - env.q_ordinary_max)
        ambulance_cap_hit = int(ambulance_overflow > 0.0)
        ordinary_cap_hit = int(ordinary_overflow > 0.0)
        ambulance_cap_hits += ambulance_cap_hit
        ordinary_cap_hits += ordinary_cap_hit
        ambulance_overflow_total += ambulance_overflow
        ordinary_overflow_total += ordinary_overflow
        ordinary_queue_terms.append(float(info["ordinary_queue_term"]))
        ordinary_overflow_terms.append(float(info["ordinary_overflow_term"]))

        offered_ambulance_mbit = (
            float(info["ambulance_queue_before_mbit"])
            + float(info["ambulance_arrival_mbit"])
        )
        ambulance_served_mbit = min(
            offered_ambulance_mbit,
            float(info["ambulance_capacity_mbps"]) * env.delta_t,
        )
        ambulance_throughput_mbps = ambulance_served_mbit / env.delta_t

        latency_ms = float(info["ambulance_latency_s"]) * 1000.0
        violation = int(info["ambulance_sla_violation"])
        ambulance_present = int(info["n_ambulance"]) > 0
        emergency = bool(info["ambulance_emergency"])
        ordinary_throughput = float(info["ordinary_throughput_mbps"])
        step_component_sum = sum(float(info[key]) for key in reward_components)
        if abs(step_component_sum - float(reward)) >= 1e-8:
            raise AssertionError("Reward components do not sum to reward_total.")

        ambulance_latencies_ms.append(latency_ms)
        ambulance_violations.append(violation)
        if ambulance_present or emergency:
            violations_present_or_emergency.append(violation)
        if ambulance_present:
            violations_when_present.append(violation)
        if emergency:
            violations_when_emergency.append(violation)
        ambulance_throughputs.append(ambulance_throughput_mbps)
        ambulance_queues.append(float(info["ambulance_queue_mbit"]))
        ordinary_throughputs.append(ordinary_throughput)
        if emergency:
            ordinary_throughputs_emergency.append(ordinary_throughput)
            ambulance_prb_emergency.append(float(info["alpha_A"]))
        else:
            ordinary_throughputs_non_emergency.append(ordinary_throughput)
            ambulance_prb_non_emergency.append(float(info["alpha_A"]))
        ordinary_demand_satisfaction.append(
            float(info["ordinary_demand_satisfaction"])
        )
        ordinary_queues.append(float(info["ordinary_queue_mbit"]))
        ordinary_target_hits.append(int(ordinary_throughput >= target))
        prb_utilizations.append(float(info["prb_utilization"]))
        ambulance_prb_fractions.append(float(info["alpha_A"]))
        rewards.append(float(reward))
        for key in reward_components:
            reward_components[key] += float(info[key])
        ordinary_capacity_limited_steps += int(info["ordinary_capacity_limited"])
        ordinary_target_eligible_steps += int(info["ordinary_target_eligible"])
        ordinary_target_violations += int(info["ordinary_target_violation"])
        ordinary_offered_mbit_total += float(info["ordinary_arrival_mbit"])
        action_counts[action] += 1
        if previous_action is not None and action != previous_action:
            action_switches += 1
        previous_action = action

        step_row = {
            "policy": policy_name,
            "episode": episode,
            "seed": seed,
            "step": step,
            "action": action,
            "ambulance_throughput_mbps": ambulance_throughput_mbps,
            "ambulance_queue_cap_hit": ambulance_cap_hit,
            "ordinary_queue_cap_hit": ordinary_cap_hit,
            "ambulance_overflow_mbit": ambulance_overflow,
            "ordinary_overflow_mbit": ordinary_overflow,
        }
        step_row.update(info)
        step_rows.append(step_row)
        step += 1

    episode_reward = float(np.sum(rewards))
    component_sum = float(sum(reward_components.values()))
    episode_row: dict[str, Any] = {
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "steps": step,
        "episode_reward": episode_reward,
        "mean_step_reward": float(np.mean(rewards)),
        **{f"{key}_sum": value for key, value in reward_components.items()},
        "reward_component_error": component_sum - episode_reward,
        "avg_ambulance_latency_ms": float(np.mean(ambulance_latencies_ms)),
        "p95_ambulance_latency_ms": float(np.percentile(ambulance_latencies_ms, 95)),
        "ambulance_sla_violation_rate": float(np.mean(ambulance_violations)),
        "ambulance_sla_violation_rate_present_or_emergency": conditional_mean(
            violations_present_or_emergency
        ),
        "ambulance_sla_violation_rate_when_present": conditional_mean(
            violations_when_present
        ),
        "ambulance_sla_violation_rate_when_emergency": conditional_mean(
            violations_when_emergency
        ),
        "avg_ambulance_throughput_mbps": float(np.mean(ambulance_throughputs)),
        "avg_ambulance_queue_mbit": float(np.mean(ambulance_queues)),
        "max_ambulance_queue_mbit": float(np.max(ambulance_queues)),
        "ambulance_queue_cap_hits": ambulance_cap_hits,
        "ambulance_overflow_mbit": ambulance_overflow_total,
        "avg_ordinary_throughput_mbps": float(np.mean(ordinary_throughputs)),
        "avg_ordinary_throughput_emergency_mbps": conditional_mean(
            ordinary_throughputs_emergency
        ),
        "avg_ordinary_throughput_non_emergency_mbps": float(
            np.mean(ordinary_throughputs_non_emergency)
        ),
        "avg_ordinary_demand_satisfaction": float(
            np.mean(ordinary_demand_satisfaction)
        ),
        "ordinary_conditional_target_violation_rate": (
            ordinary_target_violations / ordinary_target_eligible_steps
            if ordinary_target_eligible_steps
            else 0.0
        ),
        "ordinary_capacity_limited_rate": ordinary_capacity_limited_steps / step,
        "avg_ordinary_queue_mbit": float(np.mean(ordinary_queues)),
        "p95_ordinary_queue_mbit": float(np.percentile(ordinary_queues, 95)),
        "max_ordinary_queue_mbit": float(np.max(ordinary_queues)),
        "ordinary_throughput_target_attainment_rate": float(
            np.mean(ordinary_target_hits)
        ),
        "ordinary_queue_cap_hits": ordinary_cap_hits,
        "ordinary_overflow_mbit": ordinary_overflow_total,
        "ordinary_overflow_fraction": (
            ordinary_overflow_total / ordinary_offered_mbit_total
            if ordinary_offered_mbit_total
            else 0.0
        ),
        "avg_ordinary_queue_term": float(np.mean(ordinary_queue_terms)),
        "avg_ordinary_overflow_term": float(np.mean(ordinary_overflow_terms)),
        "avg_prb_utilization": float(np.mean(prb_utilizations)),
        "avg_ambulance_prb_fraction": float(np.mean(ambulance_prb_fractions)),
        "avg_ambulance_prb_fraction_emergency": conditional_mean(
            ambulance_prb_emergency
        ),
        "avg_ambulance_prb_fraction_non_emergency": float(
            np.mean(ambulance_prb_non_emergency)
        ),
        "action_switches": action_switches,
        "action_switch_rate": action_switches / max(1, step - 1),
        "total_queue_cap_hits": ambulance_cap_hits + ordinary_cap_hits,
    }
    for action_index in range(8):
        episode_row[f"action_{action_index}_count"] = int(action_counts[action_index])
        episode_row[f"action_{action_index}_rate"] = float(
            action_counts[action_index] / step
        )
    return step_rows, episode_row


def aggregate_episodes(episodes_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRIC_UNITS if metric in episodes_df.columns]
    rows = []
    for policy_name in POLICY_ORDER:
        policy_df = episodes_df[episodes_df["policy"] == policy_name]
        row: dict[str, Any] = {
            "policy": policy_name,
            "n_episodes": int(len(policy_df)),
        }
        for metric in metrics:
            values = pd.to_numeric(policy_df[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_unit"] = METRIC_UNITS[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    episodes_df: pd.DataFrame,
    figures_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    percent: bool = False,
) -> None:
    means = []
    stds = []
    for policy in POLICY_ORDER:
        values = episodes_df.loc[episodes_df["policy"] == policy, metric].astype(float)
        scale = 100.0 if percent else 1.0
        means.append(float(values.mean()) * scale)
        stds.append(float(values.std(ddof=1)) * scale)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    positions = np.arange(len(POLICY_ORDER))
    ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=3,
        color=[POLICY_COLORS[name] for name in POLICY_ORDER],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(positions, [DISPLAY_NAMES[name] for name in POLICY_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, figures_dir, stem)


def plot_action_distribution(episodes_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(8)
    width = 0.25
    for policy_index, policy in enumerate(POLICY_ORDER):
        policy_df = episodes_df[episodes_df["policy"] == policy]
        means = [policy_df[f"action_{idx}_rate"].mean() * 100.0 for idx in range(8)]
        stds = [policy_df[f"action_{idx}_rate"].std(ddof=1) * 100.0 for idx in range(8)]
        ax.bar(
            x + (policy_index - 1) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=policy,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x, [f"{idx}\n{10 * (idx + 1)}%" for idx in range(8)])
    ax.set_xlabel("Action index and Ambulance Slice PRB allocation")
    ax.set_ylabel("Action share (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    save_figure(fig, figures_dir, "fig_action_distribution")


def plot_dqn_action_by_emergency(steps_df: pd.DataFrame, figures_dir: Path) -> None:
    dqn_df = steps_df[steps_df["policy"] == "DQN"].copy()
    dqn_df["ambulance_emergency"] = dqn_df["ambulance_emergency"].astype(bool)
    state_specs = (
        (False, "Non-emergency", "#56B4E9"),
        (True, "Emergency", "#D55E00"),
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(8)
    width = 0.36
    for state_index, (state, label, color) in enumerate(state_specs):
        episode_rates = []
        for _, episode_df in dqn_df.groupby("episode"):
            state_df = episode_df[episode_df["ambulance_emergency"] == state]
            if state_df.empty:
                continue
            counts = state_df["action"].value_counts().reindex(range(8), fill_value=0)
            episode_rates.append(counts.to_numpy(dtype=float) / len(state_df) * 100.0)
        rates = np.asarray(episode_rates, dtype=float)
        means = rates.mean(axis=0)
        stds = rates.std(axis=0, ddof=1)
        ax.bar(
            x + (state_index - 0.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x, [f"{idx}\n{10 * (idx + 1)}%" for idx in range(8)])
    ax.set_xlabel("Action index and Ambulance Slice PRB allocation")
    ax.set_ylabel("DQN action share (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    save_figure(fig, figures_dir, "fig_dqn_action_distribution_by_emergency")


def plot_learning_curve(monitor_path: Path, figures_dir: Path) -> None:
    monitor_df = pd.read_csv(monitor_path, comment="#")
    if not {"r", "l"}.issubset(monitor_df.columns):
        raise KeyError(f"Monitor file lacks r/l columns: {monitor_path}")
    monitor_df["timesteps"] = monitor_df["l"].cumsum()
    smoothing_window = min(50, max(1, len(monitor_df) // 10))
    monitor_df["reward_smooth"] = (
        monitor_df["r"].rolling(smoothing_window, min_periods=1).mean()
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        monitor_df["timesteps"],
        monitor_df["r"],
        color="#9ECAE1",
        linewidth=0.7,
        alpha=0.45,
        label="Episode reward",
    )
    ax.plot(
        monitor_df["timesteps"],
        monitor_df["reward_smooth"],
        color=POLICY_COLORS["DQN"],
        linewidth=1.6,
        label=f"Rolling mean ({smoothing_window} episodes)",
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save_figure(fig, figures_dir, "fig_learning_curve")


def create_figures(
    episodes_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    figures_dir: Path,
    monitor_path: Path | None,
) -> None:
    configure_matplotlib()
    if monitor_path is not None:
        plot_learning_curve(monitor_path, figures_dir)
    plot_metric(
        episodes_df,
        figures_dir,
        "avg_ambulance_latency_ms",
        "Average Ambulance latency (ms)",
        "fig_avg_ambulance_latency",
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "p95_ambulance_latency_ms",
        "P95 Ambulance latency (ms)",
        "fig_p95_ambulance_latency",
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "ambulance_sla_violation_rate",
        "SLA violation rate (%)",
        "fig_sla_violation_rate",
        percent=True,
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "avg_ordinary_throughput_mbps",
        "Average Ordinary throughput (Mbit/s)",
        "fig_ordinary_throughput",
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "avg_ordinary_throughput_emergency_mbps",
        "Ordinary throughput in emergency (Mbit/s)",
        "fig_ordinary_throughput_emergency",
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "avg_ordinary_demand_satisfaction",
        "Demand satisfaction (%)",
        "fig_ordinary_demand_satisfaction",
        percent=True,
    )
    plot_metric(
        episodes_df,
        figures_dir,
        "p95_ordinary_queue_mbit",
        "P95 Ordinary queue (Mbit)",
        "fig_p95_ordinary_queue",
    )
    plot_action_distribution(episodes_df, figures_dir)
    plot_dqn_action_by_emergency(steps_df, figures_dir)


def main() -> None:
    args = parse_args()
    if args.episodes <= 1:
        raise ValueError("--episodes must be greater than one to report sample SD.")
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config_path)
    monitor_path = resolve_path(args.monitor_path) if args.monitor_path else None
    if output_dir.exists() and any(output_dir.glob("evaluation_*")):
        raise FileExistsError(f"Refusing to overwrite evaluation artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_env = RANSlicingEnv(config_path)
    validate_no_embb_config(probe_env.config)
    if probe_env.observation_space.shape != (13,):
        raise AssertionError("Observation shape must remain (13,).")
    if probe_env.action_space.n != 8 or probe_env.episode_length != 200:
        raise AssertionError("Expected 8 actions and 200-step episodes.")

    model = DQN.load(str(model_path))
    policies = {
        "DQN": model,
        "Static": StaticPolicy(config_path),
        "Guaranteed Load-based": GuaranteedLoadBasedPolicy(config_path),
    }
    seeds = [args.base_seed + episode for episode in range(args.episodes)]
    all_steps: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for policy_name in POLICY_ORDER:
        env = RANSlicingEnv(config_path)
        policy = policies[policy_name]
        for episode, seed in enumerate(seeds):
            step_rows, episode_row = evaluate_episode(
                policy_name=policy_name,
                policy=policy,
                env=env,
                episode=episode,
                seed=seed,
            )
            all_steps.extend(step_rows)
            all_episodes.append(episode_row)

    steps_df = pd.DataFrame(all_steps)
    episodes_df = pd.DataFrame(all_episodes)
    summary_df = aggregate_episodes(episodes_df)
    steps_path = output_dir / "evaluation_steps.csv"
    episodes_path = output_dir / "evaluation_episodes.csv"
    comparison_path = output_dir / "evaluation_comparison.csv"
    json_path = output_dir / "evaluation_summary.json"
    steps_df.to_csv(steps_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)
    summary_df.to_csv(comparison_path, index=False)

    summary_payload = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "n_episodes_per_policy": args.episodes,
        "episode_length": probe_env.episode_length,
        "base_seed": args.base_seed,
        "seeds": seeds,
        "deterministic_dqn": True,
        "same_seeds_for_all_policies": True,
        "metric_units": METRIC_UNITS,
        "validation": {
            "non_finite_values": 0,
            "prb_sum_violations": 0,
            "non_integer_prb_violations": 0,
            "ordinary_packet_model_violations": 0,
            "forbidden_traffic_config_keys": 0,
        },
        "results": summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    create_figures(episodes_df, steps_df, output_dir / "figures", monitor_path)
    if not args.quiet:
        print(summary_df.to_string(index=False))
        print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()

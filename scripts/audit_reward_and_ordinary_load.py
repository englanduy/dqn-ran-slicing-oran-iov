from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.baseline_policies import (  # noqa: E402
    GuaranteedLoadBasedPolicy,
    StaticPolicy,
)
from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402
from utils.config import load_config  # noqa: E402


REWARD_COMPONENTS = (
    "reward_latency_excess",
    "reward_sla_violation",
    "reward_ordinary_throughput",
    "reward_resource_waste",
    "reward_action_change",
)
POLICIES = {
    "Static": StaticPolicy,
    "Guaranteed Load-based": GuaranteedLoadBasedPolicy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit no-eMBB rewards and sweep Ordinary vehicle load."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "no_embb_20260620_194602",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default_config.yaml",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[2, 4, 6, 8, 10, 12],
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.zeros_like(numerator, dtype=float)
    positive_denominator = denominator > 0.0
    ratio[positive_denominator] = (
        numerator[positive_denominator] / denominator[positive_denominator]
    )
    ratio[(~positive_denominator) & (numerator > 0.0)] = np.inf
    return ratio


def audit_selected_dqn_episodes(
    run_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes_df = pd.read_csv(run_dir / "evaluation_episodes.csv")
    steps_df = pd.read_csv(run_dir / "evaluation_steps.csv")
    dqn_episodes = episodes_df[episodes_df["policy"] == "DQN"].copy()
    dqn_steps = steps_df[steps_df["policy"] == "DQN"].copy()

    median_reward = float(dqn_episodes["episode_reward"].median())
    selected = {
        "minimum": dqn_episodes.loc[dqn_episodes["episode_reward"].idxmin()],
        "near_median": dqn_episodes.loc[
            (dqn_episodes["episode_reward"] - median_reward).abs().idxmin()
        ],
        "maximum": dqn_episodes.loc[dqn_episodes["episode_reward"].idxmax()],
    }

    summary_rows: list[dict[str, Any]] = []
    selected_step_frames: list[pd.DataFrame] = []
    for selection, episode_row in selected.items():
        episode = int(episode_row["episode"])
        seed = int(episode_row["seed"])
        episode_steps = dqn_steps[
            (dqn_steps["episode"] == episode) & (dqn_steps["seed"] == seed)
        ].sort_values("step").copy()
        episode_steps.insert(0, "selection", selection)
        episode_steps["cumulative_reward"] = episode_steps["reward_total"].cumsum()
        component_step_sum = episode_steps[list(REWARD_COMPONENTS)].sum(axis=1)
        episode_steps["component_sum"] = component_step_sum
        episode_steps["component_sum_error"] = (
            component_step_sum - episode_steps["reward_total"]
        )

        component_total = float(episode_steps[list(REWARD_COMPONENTS)].to_numpy().sum())
        logged_step_total = float(episode_steps["reward_total"].sum())
        episode_csv_total = float(episode_row["episode_reward"])
        summary: dict[str, Any] = {
            "selection": selection,
            "episode": episode,
            "seed": seed,
            "median_reward_all_dqn_episodes": median_reward,
            "episode_reward": episode_csv_total,
            "reward_total_from_steps": logged_step_total,
            "reward_total_from_components": component_total,
            "components_minus_episode_reward": component_total - episode_csv_total,
            "steps": int(len(episode_steps)),
            "emergency_steps": int(episode_steps["ambulance_emergency"].astype(bool).sum()),
            "ambulance_present_steps": int((episode_steps["n_ambulance"] > 0).sum()),
            "avg_ambulance_latency_ms": float(
                episode_steps["ambulance_latency_s"].mean() * 1000.0
            ),
            "p95_ambulance_latency_ms": float(
                episode_steps["ambulance_latency_s"].quantile(0.95) * 1000.0
            ),
            "max_ambulance_latency_ms": float(
                episode_steps["ambulance_latency_s"].max() * 1000.0
            ),
            "sla_violation_steps": int(episode_steps["ambulance_sla_violation"].sum()),
            "sla_violation_rate": float(
                episode_steps["ambulance_sla_violation"].mean()
            ),
            "max_absolute_step_component_error": float(
                episode_steps["component_sum_error"].abs().max()
            ),
        }
        for component in REWARD_COMPONENTS:
            summary[f"{component}_sum"] = float(episode_steps[component].sum())
            summary[f"{component}_mean"] = float(episode_steps[component].mean())
        for action in range(8):
            summary[f"action_{action}_count"] = int((episode_steps["action"] == action).sum())
        summary_rows.append(summary)

        selected_columns = [
            "selection",
            "episode",
            "seed",
            "step",
            "ambulance_emergency",
            "n_ambulance",
            "action",
            "alpha_A",
            "ambulance_latency_s",
            "ambulance_sla_violation",
            *REWARD_COMPONENTS,
            "component_sum",
            "reward_total",
            "component_sum_error",
            "cumulative_reward",
        ]
        selected_step_frames.append(episode_steps[selected_columns])

    summary_df = pd.DataFrame(summary_rows)
    selected_steps_df = pd.concat(selected_step_frames, ignore_index=True)
    summary_df.to_csv(output_dir / "reward_episode_decomposition.csv", index=False)
    selected_steps_df.to_csv(output_dir / "reward_selected_episode_steps.csv", index=False)
    return summary_df, selected_steps_df


def current_throughput_audit(
    run_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    steps_df = pd.read_csv(run_dir / "evaluation_steps.csv")
    delta_t = float(config["simulation"]["delta_t"])
    target = float(config["qos"]["ordinary_throughput_target_mbps"])
    rows: list[dict[str, Any]] = []
    for policy, policy_df in steps_df.groupby("policy", sort=False):
        row: dict[str, Any] = {"policy": policy, "steps": int(len(policy_df))}
        for slice_name in ("ambulance", "ordinary"):
            arrivals = policy_df[f"{slice_name}_arrival_mbit"].to_numpy(float)
            queue_before = policy_df[f"{slice_name}_queue_before_mbit"].to_numpy(float)
            capacities = policy_df[f"{slice_name}_capacity_mbps"].to_numpy(float)
            arrival_load = arrivals / delta_t
            service_demand = (queue_before + arrivals) / delta_t
            served = np.minimum(service_demand, capacities)
            ratios = safe_ratio(service_demand, capacities)
            finite_ratios = ratios[np.isfinite(ratios)]
            capacity_limited = service_demand > capacities + 1e-12
            queue_nonzero = policy_df[f"{slice_name}_queue_after_mbit"].to_numpy(float) > 0

            prefix = slice_name
            row[f"{prefix}_arrival_offered_mbps_mean"] = float(arrival_load.mean())
            row[f"{prefix}_arrival_offered_mbps_p95"] = float(
                np.percentile(arrival_load, 95)
            )
            row[f"{prefix}_service_demand_mbps_mean"] = float(service_demand.mean())
            row[f"{prefix}_served_mbps_mean"] = float(served.mean())
            row[f"{prefix}_capacity_mbps_mean"] = float(capacities.mean())
            row[f"{prefix}_offered_capacity_ratio_mean"] = float(finite_ratios.mean())
            row[f"{prefix}_offered_capacity_ratio_p95"] = float(
                np.percentile(finite_ratios, 95)
            )
            row[f"{prefix}_capacity_limited_rate"] = float(capacity_limited.mean())
            row[f"{prefix}_queue_nonzero_rate"] = float(queue_nonzero.mean())
        row["ordinary_target_attainment_rate"] = float(
            (policy_df["ordinary_throughput_mbps"] >= target).mean()
        )
        rows.append(row)
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "current_throughput_audit.csv", index=False)
    return audit_df


def evaluate_baseline_episode(
    policy_name: str,
    policy: Any,
    env: RANSlicingEnv,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    delta_t = env.delta_t
    arrivals: list[float] = []
    demands: list[float] = []
    throughputs: list[float] = []
    capacities: list[float] = []
    ambulance_latencies_ms: list[float] = []
    sla_violations: list[int] = []
    queues: list[float] = []
    utilizations: list[float] = []
    capacity_limited: list[int] = []
    queue_cap_hits = 0
    overflow_total = 0.0
    last_info: dict[str, Any] | None = None
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(policy.select_action(obs, last_info))
        obs, _, terminated, truncated, info = env.step(action)
        last_info = info
        arrival_load = float(info["ordinary_arrival_mbit"]) / delta_t
        service_demand = (
            float(info["ordinary_queue_before_mbit"])
            + float(info["ordinary_arrival_mbit"])
        ) / delta_t
        capacity = float(info["ordinary_capacity_mbps"])
        raw_queue = max(
            0.0,
            float(info["ordinary_queue_before_mbit"])
            + float(info["ordinary_arrival_mbit"])
            - capacity * delta_t,
        )
        overflow = max(0.0, raw_queue - env.q_ordinary_max)
        queue_cap_hits += int(overflow > 0.0)
        overflow_total += overflow
        arrivals.append(arrival_load)
        demands.append(service_demand)
        throughputs.append(float(info["ordinary_throughput_mbps"]))
        capacities.append(capacity)
        ambulance_latencies_ms.append(float(info["ambulance_latency_s"]) * 1000.0)
        sla_violations.append(int(info["ambulance_sla_violation"]))
        queues.append(float(info["ordinary_queue_after_mbit"]))
        utilizations.append(float(info["prb_utilization"]))
        capacity_limited.append(int(service_demand > capacity + 1e-12))

    return {
        "lambda_O_packets_per_ue_s": float(env.config["traffic"]["lambda_vehicle"]),
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "ordinary_offered_load_mbps_mean": float(np.mean(arrivals)),
        "ordinary_offered_load_mbps_p95": float(np.percentile(arrivals, 95)),
        "ordinary_service_demand_mbps_mean": float(np.mean(demands)),
        "ordinary_throughput_mbps_mean": float(np.mean(throughputs)),
        "ordinary_capacity_mbps_mean": float(np.mean(capacities)),
        "ambulance_latency_ms_mean": float(np.mean(ambulance_latencies_ms)),
        "ambulance_latency_ms_p95": float(np.percentile(ambulance_latencies_ms, 95)),
        "ambulance_sla_violation_rate": float(np.mean(sla_violations)),
        "ordinary_queue_mbit_mean": float(np.mean(queues)),
        "ordinary_queue_mbit_p95": float(np.percentile(queues, 95)),
        "ordinary_queue_mbit_max": float(np.max(queues)),
        "prb_utilization_mean": float(np.mean(utilizations)),
        "ordinary_capacity_limited_rate": float(np.mean(capacity_limited)),
        "ordinary_queue_nonzero_rate": float(np.mean(np.asarray(queues) > 0.0)),
        "ordinary_queue_cap_hits": queue_cap_hits,
        "ordinary_overflow_mbit": overflow_total,
        "ordinary_target_attainment_rate": float(
            np.mean(np.asarray(throughputs) >= env.ordinary_throughput_target)
        ),
    }


def aggregate_survey(episode_df: pd.DataFrame) -> pd.DataFrame:
    id_columns = {
        "lambda_O_packets_per_ue_s",
        "policy",
        "episode",
        "seed",
    }
    metrics = [column for column in episode_df.columns if column not in id_columns]
    rows: list[dict[str, Any]] = []
    for (lambda_o, policy), group in episode_df.groupby(
        ["lambda_O_packets_per_ue_s", "policy"], sort=True
    ):
        row: dict[str, Any] = {
            "lambda_O_packets_per_ue_s": lambda_o,
            "policy": policy,
            "n_episodes": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def policy_differences(summary_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in summary_df.columns
        if column not in {"lambda_O_packets_per_ue_s", "policy", "n_episodes"}
    ]
    rows = []
    for lambda_o, group in summary_df.groupby("lambda_O_packets_per_ue_s"):
        indexed = group.set_index("policy")
        static = indexed.loc["Static"]
        guaranteed = indexed.loc["Guaranteed Load-based"]
        row: dict[str, Any] = {
            "lambda_O_packets_per_ue_s": lambda_o,
            "delta_definition": "Guaranteed Load-based minus Static",
        }
        for column in value_columns:
            row[f"delta_{column}"] = float(guaranteed[column] - static[column])
        rows.append(row)
    return pd.DataFrame(rows)


def run_load_survey(
    base_config: dict[str, Any],
    lambdas: list[float],
    episodes: int,
    base_seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_dir = output_dir / "survey_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    for lambda_o in lambdas:
        config = deepcopy(base_config)
        config["traffic"]["lambda_vehicle"] = float(lambda_o)
        config_path = config_dir / f"lambda_O_{lambda_o:g}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        seeds = [base_seed + episode for episode in range(episodes)]
        for policy_name, policy_class in POLICIES.items():
            env = RANSlicingEnv(config_path)
            policy = policy_class(config_path)
            for episode, seed in enumerate(seeds):
                episode_rows.append(
                    evaluate_baseline_episode(
                        policy_name=policy_name,
                        policy=policy,
                        env=env,
                        episode=episode,
                        seed=seed,
                    )
                )

    episode_df = pd.DataFrame(episode_rows)
    summary_df = aggregate_survey(episode_df)
    differences_df = policy_differences(summary_df)
    episode_df.to_csv(output_dir / "load_survey_episode_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "load_survey_summary.csv", index=False)
    differences_df.to_csv(output_dir / "load_survey_policy_differences.csv", index=False)
    return episode_df, summary_df, differences_df


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config_hash_before = sha256(config_path)
    env_hash_before = sha256(PROJECT_ROOT / "envs" / "ran_slicing_env.py")
    reward_summary, _ = audit_selected_dqn_episodes(run_dir, output_dir)
    throughput_summary = current_throughput_audit(run_dir, config, output_dir)
    _, survey_summary, differences = run_load_survey(
        base_config=config,
        lambdas=args.lambdas,
        episodes=args.episodes,
        base_seed=args.base_seed,
        output_dir=output_dir,
    )
    config_hash_after = sha256(config_path)
    env_hash_after = sha256(PROJECT_ROOT / "envs" / "ran_slicing_env.py")
    if config_hash_before != config_hash_after or env_hash_before != env_hash_after:
        raise AssertionError("Default config or environment changed during read-only audit.")

    metadata = {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "base_seed": args.base_seed,
        "evaluation_seeds": [args.base_seed + idx for idx in range(args.episodes)],
        "n_episodes_per_policy_per_lambda": args.episodes,
        "lambdas_packets_per_ue_s": args.lambdas,
        "policies": list(POLICIES),
        "dqn_training_performed": False,
        "default_config_modified": False,
        "reward_modified": False,
        "model_modified": False,
        "config_sha256_before": config_hash_before,
        "config_sha256_after": config_hash_after,
        "env_sha256_before": env_hash_before,
        "env_sha256_after": env_hash_after,
        "definitions": {
            "ordinary_offered_load_mbps": "ordinary_arrival_mbit / delta_t",
            "ordinary_service_demand_mbps": (
                "(ordinary_queue_before_mbit + ordinary_arrival_mbit) / delta_t"
            ),
            "capacity_limited": "ordinary_service_demand_mbps > ordinary_capacity_mbps",
            "policy_difference": "Guaranteed Load-based minus Static",
            "reward_components": list(REWARD_COMPONENTS),
        },
    }
    (output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Reward episode decomposition")
    print(reward_summary.to_string(index=False))
    print("\nCurrent throughput audit")
    print(throughput_summary.to_string(index=False))
    print("\nLoad survey summary")
    print(survey_summary.to_string(index=False))
    print("\nPolicy differences")
    print(differences.to_string(index=False))
    print(f"\nSaved audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()

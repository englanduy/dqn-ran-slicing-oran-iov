from __future__ import annotations

import argparse
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


POLICIES = {
    "Static": StaticPolicy,
    "Guaranteed Load-based": GuaranteedLoadBasedPolicy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demand-aware Ordinary load sweep.")
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
        default=[10.0, 10.5, 11.0, 11.5],
    )
    return parser.parse_args()


def run_episode(
    env: RANSlicingEnv,
    policy: Any,
    policy_name: str,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    values: dict[str, list[float]] = {
        "demand": [],
        "throughput": [],
        "throughput_emergency": [],
        "throughput_non_emergency": [],
        "demand_satisfaction": [],
        "queue": [],
        "ambulance_latency_ms": [],
        "prb_utilization": [],
    }
    capacity_limited = 0
    queue_nonzero = 0
    target_eligible = 0
    target_violations = 0
    sla_violations = 0
    cap_hits = 0
    overflow_mbit = 0.0
    offered_mbit = 0.0
    last_info: dict[str, Any] | None = None
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy.select_action(obs, last_info))
        obs, _, terminated, truncated, info = env.step(action)
        last_info = info
        throughput = float(info["ordinary_served_throughput_mbps"])
        emergency = bool(info["ambulance_emergency"])
        values["demand"].append(float(info["ordinary_demand_mbps"]))
        values["throughput"].append(throughput)
        values[
            "throughput_emergency" if emergency else "throughput_non_emergency"
        ].append(throughput)
        values["demand_satisfaction"].append(
            float(info["ordinary_demand_satisfaction"])
        )
        values["queue"].append(float(info["ordinary_queue_after_mbit"]))
        values["ambulance_latency_ms"].append(
            float(info["ambulance_latency_s"]) * 1000.0
        )
        values["prb_utilization"].append(float(info["prb_utilization"]))
        capacity_limited += int(info["ordinary_capacity_limited"])
        queue_nonzero += int(float(info["ordinary_queue_after_mbit"]) > 0.0)
        target_eligible += int(info["ordinary_target_eligible"])
        target_violations += int(info["ordinary_target_violation"])
        sla_violations += int(info["ambulance_sla_violation"])
        cap_hits += int(info["ordinary_queue_cap_hit"])
        overflow_mbit += float(info["ordinary_overflow_mbit"])
        offered_mbit += float(info["ordinary_arrival_mbit"])

    steps = len(values["demand"])
    return {
        "lambda_O_packets_per_ue_s": float(env.config["traffic"]["lambda_vehicle"]),
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "steps": steps,
        "ordinary_demand_mbps_mean": float(np.mean(values["demand"])),
        "ordinary_demand_mbps_p95": float(np.percentile(values["demand"], 95)),
        "ordinary_throughput_mbps_mean": float(np.mean(values["throughput"])),
        "ordinary_throughput_emergency_mbps_mean": float(
            np.mean(values["throughput_emergency"])
            if values["throughput_emergency"]
            else np.nan
        ),
        "ordinary_throughput_non_emergency_mbps_mean": float(
            np.mean(values["throughput_non_emergency"])
        ),
        "ordinary_demand_satisfaction_mean": float(
            np.mean(values["demand_satisfaction"])
        ),
        "ordinary_target_eligible_steps": target_eligible,
        "ordinary_target_violations": target_violations,
        "ordinary_conditional_target_violation_rate": (
            target_violations / target_eligible if target_eligible else 0.0
        ),
        "ordinary_capacity_limited_rate": capacity_limited / steps,
        "ordinary_queue_nonzero_rate": queue_nonzero / steps,
        "ordinary_queue_mbit_mean": float(np.mean(values["queue"])),
        "ordinary_queue_mbit_p95": float(np.percentile(values["queue"], 95)),
        "ordinary_queue_mbit_max": float(np.max(values["queue"])),
        "ordinary_queue_cap_hits": cap_hits,
        "ordinary_overflow_mbit": overflow_mbit,
        "ordinary_offered_mbit": offered_mbit,
        "ambulance_latency_ms_mean": float(np.mean(values["ambulance_latency_ms"])),
        "ambulance_latency_ms_p95": float(
            np.percentile(values["ambulance_latency_ms"], 95)
        ),
        "ambulance_sla_violation_rate": sla_violations / steps,
        "prb_utilization_mean": float(np.mean(values["prb_utilization"])),
    }


def aggregate(episodes: pd.DataFrame) -> pd.DataFrame:
    excluded = {"lambda_O_packets_per_ue_s", "policy", "episode", "seed", "steps"}
    metrics = [column for column in episodes.columns if column not in excluded]
    rows = []
    for (lambda_o, policy), group in episodes.groupby(
        ["lambda_O_packets_per_ue_s", "policy"], sort=True
    ):
        row: dict[str, Any] = {
            "lambda_O_packets_per_ue_s": lambda_o,
            "policy": policy,
            "n_episodes": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["pooled_capacity_limited_rate"] = float(
            (group["ordinary_capacity_limited_rate"] * group["steps"]).sum()
            / group["steps"].sum()
        )
        row["pooled_queue_nonzero_rate"] = float(
            (group["ordinary_queue_nonzero_rate"] * group["steps"]).sum()
            / group["steps"].sum()
        )
        row["pooled_cap_hit_rate"] = float(
            group["ordinary_queue_cap_hits"].sum() / group["steps"].sum()
        )
        row["pooled_overflow_fraction"] = float(
            group["ordinary_overflow_mbit"].sum()
            / group["ordinary_offered_mbit"].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_lambda(summary: pd.DataFrame) -> dict[str, Any]:
    guaranteed = summary[summary["policy"] == "Guaranteed Load-based"].copy()
    static = summary[summary["policy"] == "Static"].set_index(
        "lambda_O_packets_per_ue_s"
    )
    guaranteed["static_cap_hit_rate"] = guaranteed[
        "lambda_O_packets_per_ue_s"
    ].map(static["pooled_cap_hit_rate"])
    guaranteed["static_overflow_fraction"] = guaranteed[
        "lambda_O_packets_per_ue_s"
    ].map(static["pooled_overflow_fraction"])
    guaranteed["passes_gate"] = (
        guaranteed["pooled_capacity_limited_rate"].between(0.08, 0.15)
        & guaranteed["pooled_queue_nonzero_rate"].between(0.05, 0.20)
        & (guaranteed["pooled_cap_hit_rate"] < 0.001)
        & (guaranteed["pooled_overflow_fraction"] < 0.0005)
        & (guaranteed["static_cap_hit_rate"] < 0.001)
        & (guaranteed["static_overflow_fraction"] < 0.0005)
    )
    eligible = guaranteed[guaranteed["passes_gate"]].copy()
    if eligible.empty:
        return {
            "selected_lambda_O": None,
            "gate_passed": False,
            "reason": "No tested lambda satisfies all load gates.",
            "candidates": guaranteed.to_dict(orient="records"),
        }
    eligible["distance_to_12pct_capacity_limited"] = (
        eligible["pooled_capacity_limited_rate"] - 0.12
    ).abs()
    selected = eligible.sort_values(
        ["distance_to_12pct_capacity_limited", "lambda_O_packets_per_ue_s"]
    ).iloc[0]
    return {
        "selected_lambda_O": float(selected["lambda_O_packets_per_ue_s"]),
        "gate_passed": True,
        "selection_policy": "Guaranteed Load-based",
        "selection_metrics": selected.to_dict(),
        "candidates": guaranteed.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    config_path = args.config_path if args.config_path.is_absolute() else PROJECT_ROOT / args.config_path
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir()
    base_config = load_config(config_path)
    rows = []
    for lambda_o in args.lambdas:
        config = deepcopy(base_config)
        config["traffic"]["lambda_vehicle"] = float(lambda_o)
        candidate_path = configs_dir / f"lambda_O_{lambda_o:g}.yaml"
        candidate_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        for policy_name, policy_class in POLICIES.items():
            env = RANSlicingEnv(candidate_path)
            policy = policy_class(candidate_path)
            for episode in range(args.episodes):
                rows.append(
                    run_episode(
                        env,
                        policy,
                        policy_name,
                        episode,
                        args.base_seed + episode,
                    )
                )
    episodes = pd.DataFrame(rows)
    summary = aggregate(episodes)
    selection = select_lambda(summary)
    episodes.to_csv(output_dir / "sweep_episodes.csv", index=False)
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

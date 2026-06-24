from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402
from utils.config import load_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_ran_slicing.zip"
TENSORBOARD_LOG_PATH = PROJECT_ROOT / "logs" / "dqn"
SUMMARY_PATH = PROJECT_ROOT / "results" / "dqn_training_summary.json"


class FiniteMetricsCallback(BaseCallback):
    """Fail fast if training emits non-finite rewards or core environment metrics."""

    INFO_KEYS = (
        "ambulance_latency_s",
        "ambulance_queue_mbit",
        "ordinary_queue_mbit",
        "ambulance_capacity_mbps",
        "ordinary_capacity_mbps",
        "ordinary_throughput_mbps",
        "prb_utilization",
        "reward_total",
    )

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.steps_checked = 0
        self.max_abs_reward = 0.0
        self.total_prb = 0
        self.q_ambulance_max = 0.0
        self.q_ordinary_max = 0.0

    def _on_training_start(self) -> None:
        self.total_prb = int(self.training_env.get_attr("total_prb")[0])
        self.q_ambulance_max = float(
            self.training_env.get_attr("q_ambulance_max")[0]
        )
        self.q_ordinary_max = float(
            self.training_env.get_attr("q_ordinary_max")[0]
        )

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        if rewards.size and not np.isfinite(rewards).all():
            raise FloatingPointError(f"Non-finite training reward at step {self.num_timesteps}.")
        if rewards.size:
            self.max_abs_reward = max(
                self.max_abs_reward,
                float(np.max(np.abs(rewards))),
            )

        new_obs = np.asarray(self.locals.get("new_obs", []), dtype=float)
        if new_obs.size and not np.isfinite(new_obs).all():
            raise FloatingPointError(
                f"Non-finite observation at step {self.num_timesteps}."
            )

        for info in self.locals.get("infos", []):
            for key in self.INFO_KEYS:
                if key in info and not np.isfinite(float(info[key])):
                    raise FloatingPointError(
                        f"Non-finite {key} at training step {self.num_timesteps}."
                    )
            if (
                not isinstance(info["prb_ambulance"], int)
                or not isinstance(info["prb_ordinary"], int)
                or info["prb_ambulance"] + info["prb_ordinary"] != self.total_prb
            ):
                raise AssertionError(
                    f"Invalid PRB allocation at step {self.num_timesteps}."
                )
            if not (
                0.0 <= float(info["ambulance_queue_after_mbit"]) <= self.q_ambulance_max
                and 0.0 <= float(info["ordinary_queue_after_mbit"]) <= self.q_ordinary_max
            ):
                raise AssertionError(
                    f"Queue bound violation at step {self.num_timesteps}."
                )
            component_total = sum(
                float(info[key]) for key in EpisodeMetricsCallback.COMPONENT_KEYS
            )
            if abs(component_total - float(info["reward_total"])) >= 1e-8:
                raise AssertionError(
                    f"Step reward component mismatch at {self.num_timesteps}."
                )
        self.steps_checked = self.num_timesteps
        return True


class EpisodeMetricsCallback(BaseCallback):
    """Persist demand-aware reward and QoS diagnostics for every episode."""

    COMPONENT_KEYS = (
        "reward_latency_excess",
        "reward_sla_violation",
        "reward_ordinary_throughput",
        "reward_resource_waste",
        "reward_action_change",
        "reward_ordinary_queue",
        "reward_ordinary_overflow",
    )
    FIELDNAMES = (
        "episode",
        "timesteps",
        "episode_reward",
        "emergency_steps",
        *COMPONENT_KEYS,
        "component_sum",
        "component_error",
        "sla_violations",
        "ordinary_demand_satisfaction_mean",
        "ordinary_target_eligible_steps",
        "ordinary_target_violations",
        "ordinary_conditional_target_violation_rate",
        "ambulance_queue_mbit_mean",
        "ambulance_queue_mbit_p95",
        "ambulance_queue_mbit_max",
        "ordinary_queue_mbit_mean",
        "ordinary_queue_mbit_p95",
        "ordinary_queue_mbit_max",
        "ambulance_queue_cap_hits",
        "ordinary_queue_cap_hits",
        "ambulance_overflow_mbit",
        "ordinary_overflow_mbit",
        "ordinary_queue_term_mean",
        "ordinary_overflow_term_mean",
    )

    def __init__(self, output_path: Path) -> None:
        super().__init__(verbose=0)
        self.output_path = output_path
        self._file: Any = None
        self._writer: csv.DictWriter | None = None
        self._episode_index = 0
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        self._reward = 0.0
        self._emergency_steps = 0
        self._components = {key: 0.0 for key in self.COMPONENT_KEYS}
        self._sla_violations = 0
        self._demand_satisfaction: list[float] = []
        self._target_eligible_steps = 0
        self._target_violations = 0
        self._ambulance_queue: list[float] = []
        self._ordinary_queue: list[float] = []
        self._ambulance_cap_hits = 0
        self._ordinary_cap_hits = 0
        self._ambulance_overflow = 0.0
        self._ordinary_overflow = 0.0
        self._ordinary_queue_terms: list[float] = []
        self._ordinary_overflow_terms: list[float] = []

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for info, _reward, done in zip(infos, rewards, dones):
            # Use the environment's float64 total so the persisted decomposition
            # is audited at the requested 1e-8 tolerance. SB3 vector rewards are
            # float32 and introduce avoidable episode-level rounding drift.
            self._reward += float(info["reward_total"])
            self._emergency_steps += int(bool(info["ambulance_emergency"]))
            for key in self.COMPONENT_KEYS:
                self._components[key] += float(info[key])
            self._sla_violations += int(info["ambulance_sla_violation"])
            self._demand_satisfaction.append(
                float(info["ordinary_demand_satisfaction"])
            )
            self._target_eligible_steps += int(info["ordinary_target_eligible"])
            self._target_violations += int(info["ordinary_target_violation"])
            self._ambulance_queue.append(float(info["ambulance_queue_after_mbit"]))
            self._ordinary_queue.append(float(info["ordinary_queue_after_mbit"]))
            self._ambulance_cap_hits += int(info["ambulance_queue_cap_hit"])
            self._ordinary_cap_hits += int(info["ordinary_queue_cap_hit"])
            self._ambulance_overflow += float(info["ambulance_overflow_mbit"])
            self._ordinary_overflow += float(info["ordinary_overflow_mbit"])
            self._ordinary_queue_terms.append(float(info["ordinary_queue_term"]))
            self._ordinary_overflow_terms.append(float(info["ordinary_overflow_term"]))
            if done:
                self._write_episode()
        return True

    def _write_episode(self) -> None:
        component_sum = float(sum(self._components.values()))
        component_error = component_sum - self._reward
        if abs(component_error) >= 1e-8:
            raise AssertionError(
                f"Episode reward component mismatch: {component_error}"
            )
        conditional_violation = (
            self._target_violations / self._target_eligible_steps
            if self._target_eligible_steps
            else 0.0
        )
        row = {
            "episode": self._episode_index,
            "timesteps": self.num_timesteps,
            "episode_reward": self._reward,
            "emergency_steps": self._emergency_steps,
            **self._components,
            "component_sum": component_sum,
            "component_error": component_error,
            "sla_violations": self._sla_violations,
            "ordinary_demand_satisfaction_mean": float(
                np.mean(self._demand_satisfaction)
            ),
            "ordinary_target_eligible_steps": self._target_eligible_steps,
            "ordinary_target_violations": self._target_violations,
            "ordinary_conditional_target_violation_rate": conditional_violation,
            "ambulance_queue_mbit_mean": float(np.mean(self._ambulance_queue)),
            "ambulance_queue_mbit_p95": float(
                np.percentile(self._ambulance_queue, 95)
            ),
            "ambulance_queue_mbit_max": float(np.max(self._ambulance_queue)),
            "ordinary_queue_mbit_mean": float(np.mean(self._ordinary_queue)),
            "ordinary_queue_mbit_p95": float(np.percentile(self._ordinary_queue, 95)),
            "ordinary_queue_mbit_max": float(np.max(self._ordinary_queue)),
            "ambulance_queue_cap_hits": self._ambulance_cap_hits,
            "ordinary_queue_cap_hits": self._ordinary_cap_hits,
            "ambulance_overflow_mbit": self._ambulance_overflow,
            "ordinary_overflow_mbit": self._ordinary_overflow,
            "ordinary_queue_term_mean": float(np.mean(self._ordinary_queue_terms)),
            "ordinary_overflow_term_mean": float(
                np.mean(self._ordinary_overflow_terms)
            ),
        }
        if self._writer is None or self._file is None:
            raise RuntimeError("Episode metrics writer was not initialized.")
        self._writer.writerow(row)
        self._file.flush()
        self._episode_index += 1
        self._reset_accumulator()

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


def parse_args() -> argparse.Namespace:
    """Parse optional training overrides while preserving no-argument defaults."""
    parser = argparse.ArgumentParser(description="Train DQN for RAN slicing.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
        help="Effective environment and DQN YAML configuration.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps_initial from configs/default_config.yaml.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional run name used for default output paths: "
            "models/<run_name>.zip, logs/<run_name>/, "
            "results/<run_name>_training_summary.json."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the trained model output path.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="Override the TensorBoard log directory.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Override the training summary JSON path.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed DQN and its environment.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Save a model and replay-buffer checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory; required when --checkpoint-freq is used.",
    )
    parser.add_argument(
        "--monitor-path",
        type=Path,
        default=None,
        help="Stable-Baselines3 Monitor CSV path for episode reward and length.",
    )
    parser.add_argument(
        "--config-snapshot-path",
        type=Path,
        default=None,
        help="Copy the effective YAML config to this path before training.",
    )
    parser.add_argument(
        "--git-diff-path",
        type=Path,
        default=None,
        help="Save the current Git diff for reproducibility when the worktree is dirty.",
    )
    parser.add_argument(
        "--episode-metrics-path",
        type=Path,
        default=None,
        help="Write per-episode reward components and demand-aware QoS metrics.",
    )
    parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Refuse to overwrite any requested model, log, summary, or checkpoint output.",
    )
    return parser.parse_args()


def build_hyperparameters(config: dict) -> dict:
    """Extract Stable-Baselines3 DQN hyperparameters from the YAML config."""
    dqn_config = config["dqn"]
    return {
        "learning_rate": float(dqn_config["learning_rate"]),
        "gamma": float(dqn_config["gamma"]),
        "batch_size": int(dqn_config["batch_size"]),
        "buffer_size": int(dqn_config["buffer_size"]),
        "learning_starts": int(dqn_config["learning_starts"]),
        "target_update_interval": int(dqn_config["target_update_interval"]),
        "exploration_initial_eps": float(dqn_config["exploration_initial_eps"]),
        "exploration_final_eps": float(dqn_config["exploration_final_eps"]),
        "exploration_fraction": float(dqn_config["exploration_fraction"]),
        "total_timesteps": int(dqn_config["total_timesteps_initial"]),
    }


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve model, TensorBoard, and summary paths from defaults and CLI args."""
    if args.run_name:
        model_path = PROJECT_ROOT / "models" / f"{args.run_name}.zip"
        tensorboard_log_path = PROJECT_ROOT / "logs" / args.run_name
        summary_path = PROJECT_ROOT / "results" / f"{args.run_name}_training_summary.json"
    else:
        model_path = MODEL_PATH
        tensorboard_log_path = TENSORBOARD_LOG_PATH
        summary_path = SUMMARY_PATH

    # Explicit path arguments take precedence over run-name-derived defaults.
    if args.model_path is not None:
        model_path = args.model_path
    if args.tensorboard_log is not None:
        tensorboard_log_path = args.tensorboard_log
    if args.summary_path is not None:
        summary_path = args.summary_path

    return (
        resolve_project_path(model_path),
        resolve_project_path(tensorboard_log_path),
        resolve_project_path(summary_path),
    )


def resolve_project_path(path: Path) -> Path:
    """Treat relative CLI paths as relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_dqn(args: argparse.Namespace | None = None) -> DQN:
    """Train DQN on RANSlicingEnv and save the model plus training metadata."""
    if args is None:
        args = parse_args()

    config_path = resolve_project_path(args.config_path)
    config = load_config(config_path)
    hyperparameters = build_hyperparameters(config)
    if args.total_timesteps is not None:
        hyperparameters["total_timesteps"] = int(args.total_timesteps)

    model_path, tensorboard_log_path, summary_path = resolve_output_paths(args)
    checkpoint_dir = (
        resolve_project_path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    monitor_path = resolve_project_path(args.monitor_path) if args.monitor_path else None
    config_snapshot_path = (
        resolve_project_path(args.config_snapshot_path)
        if args.config_snapshot_path
        else None
    )
    git_diff_path = (
        resolve_project_path(args.git_diff_path) if args.git_diff_path else None
    )
    episode_metrics_path = (
        resolve_project_path(args.episode_metrics_path)
        if args.episode_metrics_path
        else None
    )

    if args.checkpoint_freq is not None and args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be a positive integer.")
    if args.checkpoint_freq is not None and checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required with --checkpoint-freq.")
    if args.fail_if_exists:
        ensure_outputs_do_not_exist(
            model_path=model_path,
            tensorboard_log_path=tensorboard_log_path,
            summary_path=summary_path,
            checkpoint_dir=checkpoint_dir,
            monitor_path=monitor_path,
            episode_metrics_path=episode_metrics_path,
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_log_path.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
    if config_snapshot_path is not None:
        config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, config_snapshot_path)
    if git_diff_path is not None:
        git_diff_path.parent.mkdir(parents=True, exist_ok=True)
        save_git_diff(git_diff_path)

    # Use the project environment exactly as specified; evaluation is handled elsewhere.
    env = RANSlicingEnv(config_path)
    training_env = Monitor(env, filename=str(monitor_path)) if monitor_path else env

    # Match the simulation spec: MlpPolicy with two hidden layers [128, 128].
    policy_kwargs = {
        "net_arch": [128, 128],
        "activation_fn": th.nn.ReLU,
    }

    model = DQN(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=hyperparameters["learning_rate"],
        gamma=hyperparameters["gamma"],
        batch_size=hyperparameters["batch_size"],
        buffer_size=hyperparameters["buffer_size"],
        learning_starts=hyperparameters["learning_starts"],
        target_update_interval=hyperparameters["target_update_interval"],
        exploration_initial_eps=hyperparameters["exploration_initial_eps"],
        exploration_final_eps=hyperparameters["exploration_final_eps"],
        exploration_fraction=hyperparameters["exploration_fraction"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log_path),
        seed=args.seed,
        verbose=1,
    )

    finite_callback = FiniteMetricsCallback()
    callbacks: list[BaseCallback] = [finite_callback]
    if episode_metrics_path is not None:
        callbacks.append(EpisodeMetricsCallback(episode_metrics_path))
    if checkpoint_dir is not None and args.checkpoint_freq is not None:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{args.run_name or 'dqn'}_checkpoint",
                save_replay_buffer=True,
            )
        )

    # Run initial training only; no evaluation is performed in this script.
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"],
        tb_log_name=args.run_name or "dqn_ran_slicing",
        callback=CallbackList(callbacks),
    )
    elapsed_seconds = time.perf_counter() - start_time
    finished_at = datetime.now().astimezone()
    model.save(str(model_path))

    save_training_summary(
        hyperparameters=hyperparameters,
        run_name=args.run_name,
        model_path=model_path,
        tensorboard_log_path=tensorboard_log_path,
        summary_path=summary_path,
        seed=args.seed,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        checkpoint_dir=checkpoint_dir,
        monitor_path=monitor_path,
        config_snapshot_path=config_snapshot_path,
        git_diff_path=git_diff_path,
        episode_metrics_path=episode_metrics_path,
        finite_callback=finite_callback,
    )
    return model


def ensure_outputs_do_not_exist(
    model_path: Path,
    tensorboard_log_path: Path,
    summary_path: Path,
    checkpoint_dir: Path | None,
    monitor_path: Path | None,
    episode_metrics_path: Path | None,
) -> None:
    """Protect existing models and experiment artifacts from accidental overwrite."""
    model_zip_path = model_path if model_path.suffix == ".zip" else Path(f"{model_path}.zip")
    candidates = [model_zip_path, summary_path]
    if monitor_path is not None:
        candidates.append(monitor_path)
    if episode_metrics_path is not None:
        candidates.append(episode_metrics_path)
    existing = [path for path in candidates if path.exists()]
    if tensorboard_log_path.exists() and any(tensorboard_log_path.iterdir()):
        existing.append(tensorboard_log_path)
    if checkpoint_dir is not None and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        existing.append(checkpoint_dir)
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {formatted}")


def save_git_diff(path: Path) -> None:
    """Save the dirty-worktree patch that accompanies the recorded Git commit."""
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    path.write_bytes(result.stdout)


def git_metadata() -> dict[str, Any]:
    """Return the current commit and dirty status without mutating the repository."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def software_versions() -> dict[str, str]:
    packages = (
        "gymnasium",
        "stable-baselines3",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "PyYAML",
    )
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def save_training_summary(
    hyperparameters: dict,
    run_name: str | None,
    model_path: Path,
    tensorboard_log_path: Path,
    summary_path: Path,
    seed: int | None,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
    checkpoint_dir: Path | None,
    monitor_path: Path | None,
    config_snapshot_path: Path | None,
    git_diff_path: Path | None,
    episode_metrics_path: Path | None,
    finite_callback: FiniteMetricsCallback,
) -> None:
    """Persist the training run metadata for reproducibility."""
    summary = {
        "run_name": run_name,
        "from_scratch": True,
        "seed": seed,
        "total_timesteps": hyperparameters["total_timesteps"],
        "model_path": str(model_path),
        "tensorboard_log_path": str(tensorboard_log_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "monitor_path": str(monitor_path) if monitor_path else None,
        "config_snapshot_path": (
            str(config_snapshot_path) if config_snapshot_path else None
        ),
        "git_diff_path": str(git_diff_path) if git_diff_path else None,
        "episode_metrics_path": (
            str(episode_metrics_path) if episode_metrics_path else None
        ),
        "hyperparameters": hyperparameters,
        "policy_kwargs": {"net_arch": [128, 128], "activation_fn": "ReLU"},
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "training_monitor": {
            "steps_checked": finite_callback.steps_checked,
            "max_abs_reward": finite_callback.max_abs_reward,
            "non_finite_detected": False,
        },
        "software_versions": software_versions(),
        "git": git_metadata(),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    train_dqn(args)
    model_path, tensorboard_log_path, summary_path = resolve_output_paths(args)
    print(f"Saved trained DQN model to {model_path}")
    print(f"Saved TensorBoard logs to {tensorboard_log_path}")
    print(f"Saved training metadata to {summary_path}")


if __name__ == "__main__":
    main()

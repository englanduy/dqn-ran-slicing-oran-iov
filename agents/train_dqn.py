from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch as th
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv  # noqa: E402
from utils.config import load_config  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_ran_slicing.zip"
TENSORBOARD_LOG_PATH = PROJECT_ROOT / "logs" / "dqn"
SUMMARY_PATH = PROJECT_ROOT / "results" / "dqn_training_summary.json"


def parse_args() -> argparse.Namespace:
    """Parse optional training overrides while preserving no-argument defaults."""
    parser = argparse.ArgumentParser(description="Train DQN for RAN slicing.")
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

    config = load_config(CONFIG_PATH)
    hyperparameters = build_hyperparameters(config)
    if args.total_timesteps is not None:
        hyperparameters["total_timesteps"] = int(args.total_timesteps)

    model_path, tensorboard_log_path, summary_path = resolve_output_paths(args)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_log_path.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the project environment exactly as specified; evaluation is handled elsewhere.
    env = RANSlicingEnv(CONFIG_PATH)

    # Match the simulation spec: MlpPolicy with two hidden layers [128, 128].
    policy_kwargs = {
        "net_arch": [128, 128],
        "activation_fn": th.nn.ReLU,
    }

    model = DQN(
        policy="MlpPolicy",
        env=env,
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
        verbose=1,
    )

    # Run initial training only; no evaluation is performed in this script.
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"],
        tb_log_name="dqn_ran_slicing",
    )
    model.save(str(model_path))

    save_training_summary(
        hyperparameters=hyperparameters,
        run_name=args.run_name,
        model_path=model_path,
        tensorboard_log_path=tensorboard_log_path,
        summary_path=summary_path,
    )
    return model


def save_training_summary(
    hyperparameters: dict,
    run_name: str | None,
    model_path: Path,
    tensorboard_log_path: Path,
    summary_path: Path,
) -> None:
    """Persist the training run metadata for reproducibility."""
    summary = {
        "run_name": run_name,
        "total_timesteps": hyperparameters["total_timesteps"],
        "model_path": str(model_path),
        "tensorboard_log_path": str(tensorboard_log_path),
        "hyperparameters": hyperparameters,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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

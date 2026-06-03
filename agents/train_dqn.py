from __future__ import annotations

import json
import sys
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


def train_dqn() -> DQN:
    """Train DQN on RANSlicingEnv and save the model plus training metadata."""
    config = load_config(CONFIG_PATH)
    hyperparameters = build_hyperparameters(config)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_LOG_PATH.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        tensorboard_log=str(TENSORBOARD_LOG_PATH),
        verbose=1,
    )

    # Run initial training only; no evaluation is performed in this script.
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"],
        tb_log_name="dqn_ran_slicing",
    )
    model.save(str(MODEL_PATH))

    save_training_summary(hyperparameters)
    return model


def save_training_summary(hyperparameters: dict) -> None:
    """Persist the training run metadata for reproducibility."""
    summary = {
        "total_timesteps": hyperparameters["total_timesteps"],
        "hyperparameters": hyperparameters,
        "model_path": str(MODEL_PATH),
        "tensorboard_log_path": str(TENSORBOARD_LOG_PATH),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    train_dqn()
    print(f"Saved trained DQN model to {MODEL_PATH}")
    print(f"Saved TensorBoard logs to {TENSORBOARD_LOG_PATH}")
    print(f"Saved training metadata to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.config import load_config


def alpha_to_action(alpha_values: list[float] | np.ndarray, alpha: float) -> int:
    """Return the index of the valid action whose alpha_A is nearest to alpha."""
    values = np.asarray(alpha_values, dtype=np.float32)
    return int(np.argmin(np.abs(values - float(alpha))))


def action_to_alpha(alpha_values: list[float] | np.ndarray, action: int) -> float:
    """Return the ambulance PRB ratio alpha_A for a discrete action."""
    values = np.asarray(alpha_values, dtype=np.float32)
    if action < 0 or action >= len(values):
        raise ValueError(f"Invalid action {action}; expected [0, {len(values) - 1}].")
    return float(values[int(action)])


class BasePolicy:
    """Shared config loading and call interface for baseline policies."""

    def __init__(self, config_path: str | Path = "configs/default_config.yaml") -> None:
        self.config = load_config(config_path)
        self.alpha_values = np.asarray(
            self.config["action"]["alpha_values"],
            dtype=np.float32,
        )
        self.epsilon = float(self.config["qos"]["epsilon"])

    def __call__(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        return self.select_action(observation, info)

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        raise NotImplementedError


class StaticPolicy(BasePolicy):
    """Static slicing baseline: always choose alpha_A = 0.3."""

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        return alpha_to_action(self.alpha_values, 0.3)


class PriorityPolicy(BasePolicy):
    """Priority baseline: reserve more PRBs whenever any ambulance UE is present."""

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        # Prefer raw info when available; otherwise use normalized n_A from observation[0].
        n_ambulance = info.get("n_ambulance") if info is not None else None
        ambulance_present = bool(n_ambulance > 0) if n_ambulance is not None else observation[0] > 0
        alpha = 0.7 if ambulance_present else 0.1
        return alpha_to_action(self.alpha_values, alpha)


class LoadBasedPolicy(BasePolicy):
    """Load baseline using normalized rho_A and rho_O from the observation."""

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        # Observation indices follow the spec: rho_A at index 2 and rho_O at index 3.
        rho_ambulance = float(observation[2])
        rho_ordinary = float(observation[3])
        alpha = rho_ambulance / (rho_ambulance + rho_ordinary + self.epsilon)
        return alpha_to_action(self.alpha_values, alpha)


class GreedySLAPolicy(BasePolicy):
    """Greedy SLA baseline with one-step alpha_A increases/decreases."""

    def __init__(self, config_path: str | Path = "configs/default_config.yaml") -> None:
        super().__init__(config_path)
        self.latency_threshold = float(
            self.config["qos"]["ambulance_latency_threshold_s"]
        )
        self.ordinary_throughput_target = float(
            self.config["qos"]["ordinary_throughput_target_mbps"]
        )
        self.previous_action = alpha_to_action(self.alpha_values, 0.3)

    def reset(self) -> None:
        """Reset the internal previous action to the static baseline action."""
        self.previous_action = alpha_to_action(self.alpha_values, 0.3)

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        # Use raw info metrics consistently when available, matching the env's info dict.
        # Without info, fall back to normalized observation entries l_A and r_O.
        if (
            info is not None
            and "ambulance_latency_s" in info
            and "ordinary_throughput_mbps" in info
        ):
            latency_exceeds_sla = (
                float(info["ambulance_latency_s"]) > self.latency_threshold
            )
            latency_below_relief_point = (
                float(info["ambulance_latency_s"]) < 0.7 * self.latency_threshold
            )
            ordinary_below_target = (
                float(info["ordinary_throughput_mbps"])
                < self.ordinary_throughput_target
            )
        else:
            latency_norm = float(observation[6])
            ordinary_throughput_norm = float(observation[7])
            latency_exceeds_sla = latency_norm >= 1.0
            latency_below_relief_point = latency_norm < 0.7
            ordinary_below_target = ordinary_throughput_norm < 1.0

        if latency_exceeds_sla:
            self.previous_action = min(self.previous_action + 1, len(self.alpha_values) - 1)
        elif latency_below_relief_point and ordinary_below_target:
            self.previous_action = max(self.previous_action - 1, 0)

        return self.previous_action


class RandomPolicy(BasePolicy):
    """Random sanity-check baseline over the valid action set."""

    def __init__(
        self,
        config_path: str | Path = "configs/default_config.yaml",
        seed: int | None = None,
    ) -> None:
        super().__init__(config_path)
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> int:
        return int(self.rng.integers(0, len(self.alpha_values)))

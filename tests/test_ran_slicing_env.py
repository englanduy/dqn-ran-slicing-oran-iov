from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy

import numpy as np
import yaml
from gymnasium.utils.env_checker import check_env

from envs.ran_slicing_env import RANSlicingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"


class RecordingGenerator:
    """Delegate RNG operations while recording Poisson means and samples."""

    def __init__(self, generator: np.random.Generator) -> None:
        self.generator = generator
        self.poisson_calls: list[tuple[float, Any]] = []

    def poisson(self, lam: float, *args: Any, **kwargs: Any) -> Any:
        sample = self.generator.poisson(lam, *args, **kwargs)
        self.poisson_calls.append((float(lam), sample))
        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_nested_equal(left_item, right_item)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, (float, np.floating)):
        assert float(left) == float(right)
    else:
        assert left == right


def test_removed_embb_component_is_absent() -> None:
    env = RANSlicingEnv(CONFIG_PATH)
    forbidden_terms = ("embb", "surge", "high_rate")

    assert not any(
        term in attribute.lower()
        for attribute in vars(env)
        for term in forbidden_terms
    )
    assert not any(
        term in key.lower()
        for section in env.config.values()
        if isinstance(section, dict)
        for key in section
        for term in forbidden_terms
    )


def test_same_seed_and_actions_are_deterministic() -> None:
    env_a = RANSlicingEnv(CONFIG_PATH)
    env_b = RANSlicingEnv(CONFIG_PATH)
    obs_a, info_a = env_a.reset(seed=2026)
    obs_b, info_b = env_b.reset(seed=2026)
    assert_nested_equal(obs_a, obs_b)
    assert_nested_equal(info_a, info_b)

    for action in [0, 7, 3, 5, 1] * 40:
        transition_a = env_a.step(action)
        transition_b = env_b.step(action)
        assert_nested_equal(transition_a, transition_b)


def test_ordinary_arrival_uses_only_configured_vehicle_poisson_model() -> None:
    env = RANSlicingEnv(CONFIG_PATH)
    env.reset(seed=77)
    recorder = RecordingGenerator(env.rng_ordinary_traffic)
    env.rng_ordinary_traffic = recorder

    _, _, _, _, info = env.step(2)

    assert len(recorder.poisson_calls) == 1
    ordinary_mean, ordinary_packets = recorder.poisson_calls[0]
    traffic = env.config["traffic"]
    expected_mean = (
        float(traffic["lambda_vehicle"])
        * env.n_ordinary_vehicles
        * env.delta_t
    )
    expected_arrival_mbit = (
        int(ordinary_packets) * float(traffic["vehicle_packet_size_mbit"])
    )

    assert ordinary_mean == expected_mean
    assert info["ordinary_arrival_mbit"] == expected_arrival_mbit
    assert env.a_ordinary_max == (
        env.ordinary_vehicles_max
        * float(traffic["lambda_vehicle"])
        * env.delta_t
        * float(traffic["vehicle_packet_size_mbit"])
    )


def test_interface_and_full_episode_metrics_are_finite() -> None:
    env = RANSlicingEnv(CONFIG_PATH)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (13,)
    assert env.action_space.n == 8
    np.testing.assert_allclose(env.alpha_values, np.arange(0.1, 0.9, 0.1))
    assert env.episode_length == 200

    finite_info_keys = (
        "ambulance_arrival_mbit",
        "ordinary_arrival_mbit",
        "ambulance_capacity_mbps",
        "ordinary_capacity_mbps",
        "ambulance_queue_mbit",
        "ordinary_queue_mbit",
        "ambulance_latency_s",
        "ordinary_throughput_mbps",
        "ordinary_demand_mbps",
        "ordinary_demand_satisfaction",
        "prb_utilization",
        "reward_total",
    )

    for step in range(env.episode_length):
        observation, reward, terminated, truncated, info = env.step(step % 8)
        assert observation.shape == (13,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert all(np.isfinite(info[key]) for key in finite_info_keys)
        component_sum = sum(
            info[key]
            for key in (
                "reward_latency_excess",
                "reward_sla_violation",
                "reward_ordinary_throughput",
                "reward_resource_waste",
                "reward_action_change",
                "reward_ordinary_queue",
                "reward_ordinary_overflow",
            )
        )
        assert abs(component_sum - reward) < 1e-12
        assert 0.0 <= info["ambulance_queue_mbit"] <= env.q_ambulance_max
        assert 0.0 <= info["ordinary_queue_mbit"] <= env.q_ordinary_max
        assert info["ambulance_latency_s"] >= 0.0
        assert info["ordinary_throughput_mbps"] >= 0.0
        assert not terminated
        assert truncated is (step == env.episode_length - 1)


def test_gymnasium_environment_checker() -> None:
    check_env(RANSlicingEnv(CONFIG_PATH), skip_render_check=True)


def test_demand_aware_ordinary_reward_formula() -> None:
    env = RANSlicingEnv(CONFIG_PATH)
    env.reset(seed=123)
    _, _, _, _, info = env.step(2)
    demand = float(info["ordinary_demand_mbps"])
    served = float(info["ordinary_served_throughput_mbps"])
    expected = (
        1.0
        if demand == 0.0
        else min(
            1.0,
            served
            / (min(demand, env.ordinary_throughput_target) + env.epsilon),
        )
    )
    assert info["ordinary_demand_satisfaction"] == expected
    assert info["reward_ordinary_throughput"] == (
        env.config["reward"]["w_ordinary_throughput"] * expected
    )

    env.n_ordinary_vehicles = 0
    env.q_ordinary = 0.0
    _, _, _, _, zero_info = env.step(2)
    assert zero_info["ordinary_demand_mbps"] == 0.0
    assert zero_info["ordinary_demand_satisfaction"] == 1.0


def test_lambda_change_preserves_nonordinary_rng_trajectories(tmp_path: Path) -> None:
    base_config = RANSlicingEnv(CONFIG_PATH).config
    config_paths = []
    for lambda_o in (10.0, 11.5):
        config = deepcopy(base_config)
        config["traffic"]["lambda_vehicle"] = lambda_o
        path = tmp_path / f"lambda_{lambda_o}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config_paths.append(path)

    env_a = RANSlicingEnv(config_paths[0])
    env_b = RANSlicingEnv(config_paths[1])
    env_a.reset(seed=10_000)
    env_b.reset(seed=10_000)
    for action in [0, 7, 3, 5, 1] * 40:
        _, _, _, _, info_a = env_a.step(action)
        _, _, _, _, info_b = env_b.step(action)
        for key in (
            "ambulance_emergency",
            "n_ambulance",
            "n_ordinary_vehicles",
            "ambulance_arrival_mbit",
            "ambulance_capacity_mbps",
            "ordinary_capacity_mbps",
        ):
            assert info_a[key] == info_b[key]


def _penalty_config(tmp_path: Path, queue_weight: float, overflow_weight: float) -> Path:
    config = deepcopy(RANSlicingEnv(CONFIG_PATH).config)
    config["reward"]["w_ordinary_queue"] = queue_weight
    config["reward"]["w_ordinary_overflow"] = overflow_weight
    path = tmp_path / f"penalty_{queue_weight}_{overflow_weight}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_ordinary_queue_and_overflow_penalty_boundary_cases(tmp_path: Path) -> None:
    env = RANSlicingEnv(_penalty_config(tmp_path, 1.0, 10.0))

    env.reset(seed=111)
    env.q_ordinary = 0.0
    env.n_ordinary_vehicles = 0
    _, reward, _, _, empty_info = env.step(0)
    assert empty_info["ordinary_queue_after_mbit"] == 0.0
    assert empty_info["ordinary_queue_term"] == 0.0
    assert empty_info["reward_ordinary_queue"] == 0.0
    assert empty_info["ordinary_overflow_mbit"] == 0.0
    assert empty_info["ordinary_overflow_term"] == 0.0
    assert empty_info["reward_ordinary_overflow"] == 0.0
    assert np.isfinite(reward)

    env.reset(seed=222)
    env.q_ordinary = env.q_ordinary_max
    env.n_ordinary_vehicles = env.ordinary_vehicles_max
    _, reward, _, _, cap_info = env.step(7)
    assert cap_info["ordinary_queue_after_mbit"] == env.q_ordinary_max
    assert cap_info["ordinary_queue_term"] == 1.0
    assert cap_info["ordinary_overflow_mbit"] > 0.0
    assert 0.0 < cap_info["ordinary_overflow_term"] <= 1.0
    assert cap_info["reward_ordinary_queue"] == -1.0
    assert cap_info["reward_ordinary_overflow"] < 0.0
    assert np.isfinite(reward)


def test_queue_transition_is_committed_once_and_reward_decomposes(tmp_path: Path) -> None:
    env = RANSlicingEnv(_penalty_config(tmp_path, 2.0, 10.0))
    observation, _ = env.reset(seed=333)
    assert observation.shape == (13,)
    assert env.action_space.n == 8

    for action in range(8):
        observation, reward, _, _, info = env.step(action)
        expected_raw = max(
            0.0,
            info["ordinary_queue_before_mbit"]
            + info["ordinary_arrival_mbit"]
            - info["ordinary_capacity_mbps"] * env.delta_t,
        )
        expected_next = min(env.q_ordinary_max, expected_raw)
        assert info["ordinary_queue_after_mbit"] == expected_next
        assert env.q_ordinary == expected_next
        component_sum = sum(
            info[key]
            for key in (
                "reward_latency_excess",
                "reward_sla_violation",
                "reward_ordinary_throughput",
                "reward_resource_waste",
                "reward_action_change",
                "reward_ordinary_queue",
                "reward_ordinary_overflow",
            )
        )
        assert abs(component_sum - info["reward_total"]) < 1e-8
        assert abs(component_sum - reward) < 1e-8
        assert np.isfinite(observation).all()
        assert np.isfinite(list(info.values())).all()

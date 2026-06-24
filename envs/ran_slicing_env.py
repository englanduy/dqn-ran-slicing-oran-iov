from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.config import load_config


class RANSlicingEnv(gym.Env):
    """Gymnasium environment for dynamic two-slice RAN PRB allocation."""

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path = "configs/default_config.yaml") -> None:
        super().__init__()

        self.config = load_config(config_path)
        sim_cfg = self.config["simulation"]
        user_cfg = self.config["users"]
        traffic_cfg = self.config["traffic"]
        channel_cfg = self.config["channel"]
        queue_cfg = self.config["queue"]
        qos_cfg = self.config["qos"]
        action_cfg = self.config["action"]
        reward_cfg = self.config["reward"]

        self.default_seed = int(sim_cfg["seed"])
        self.total_prb = int(sim_cfg["total_prb"])
        self.delta_t = float(sim_cfg["delta_t"])
        self.episode_length = int(sim_cfg["episode_length"])

        self.max_ambulance = int(user_cfg["max_ambulance"])
        self.ordinary_vehicles_min = int(user_cfg["ordinary_vehicles_min"])
        self.ordinary_vehicles_max = int(user_cfg["ordinary_vehicles_max"])
        self.max_ordinary_users_norm = float(user_cfg["max_ordinary_users_norm"])

        self.r_prb = float(channel_cfg["r_prb_mbps"])
        self.eta_max = float(channel_cfg["eta_max"])

        self.q_ambulance_max = float(queue_cfg["ambulance_queue_max_mbit"])
        self.q_ordinary_max = float(queue_cfg["ordinary_queue_max_mbit"])

        self.latency_threshold = float(qos_cfg["ambulance_latency_threshold_s"])
        self.ordinary_throughput_target = float(qos_cfg["ordinary_throughput_target_mbps"])
        self.epsilon = float(qos_cfg["epsilon"])
        self.latency_excess_clip = float(reward_cfg.get("latency_excess_clip", 10.0))

        self.alpha_values = np.asarray(action_cfg["alpha_values"], dtype=np.float32)
        if self.alpha_values.shape != (8,):
            raise ValueError("RANSlicingEnv requires exactly 8 configured alpha values.")

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(13,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(8)

        # Maximum arrival rates are used only to normalize rho_A and rho_O.
        self.a_ambulance_max = (
            self.max_ambulance
            * float(traffic_cfg["lambda_ambulance_emergency"])
            * self.delta_t
            * float(traffic_cfg["ambulance_packet_size_mbit"])
        )
        self.a_ordinary_max = (
            self.ordinary_vehicles_max
            * float(traffic_cfg["lambda_vehicle"])
            * self.delta_t
            * float(traffic_cfg["vehicle_packet_size_mbit"])
        )

        self._has_reset = False
        self.current_step = 0
        self.n_ambulance = 0
        self.n_ordinary_vehicles = 0
        self.ambulance_emergency = False
        self.q_ambulance = 0.0
        self.q_ordinary = 0.0
        self.last_a_ambulance = 0.0
        self.last_a_ordinary = 0.0
        self.last_l_ambulance = 0.0
        self.last_r_ordinary = 0.0
        self.last_prb_utilization = 0.0
        self.last_eta_ambulance_avg = 0.0
        self.last_eta_ordinary_avg = 0.0
        self.alpha_ambulance_prev = float(self.alpha_values[2])

        # Initialized deterministically by reset(). Separate streams prevent a
        # change in one stochastic process from perturbing the others.
        self.rng_emergency = np.random.default_rng(self.default_seed)
        self.rng_ue = np.random.default_rng(self.default_seed)
        self.rng_ambulance_traffic = np.random.default_rng(self.default_seed)
        self.rng_ordinary_traffic = np.random.default_rng(self.default_seed)
        self.rng_channel = np.random.default_rng(self.default_seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and not self._has_reset:
            seed = self.default_seed
        super().reset(seed=seed)
        self._reset_rng_streams(seed)
        self._has_reset = True

        # Initialize the episode population within the configured simulation ranges.
        self.current_step = 0
        # Normal state starts with no more than one ambulance UE present.
        self.n_ambulance = self._sample_normal_ambulance_count()
        self.n_ordinary_vehicles = int(
            self.rng_ue.integers(
                self.ordinary_vehicles_min,
                self.ordinary_vehicles_max + 1,
            )
        )
        # Reset traffic processes, queues, and previous-step state features.
        self.ambulance_emergency = False
        self.q_ambulance = 0.0
        self.q_ordinary = 0.0
        self.last_a_ambulance = 0.0
        self.last_a_ordinary = 0.0
        self.last_l_ambulance = 0.0
        self.last_r_ordinary = 0.0
        self.last_prb_utilization = 0.0
        self.last_eta_ambulance_avg = 0.0
        self.last_eta_ordinary_avg = 0.0
        self.alpha_ambulance_prev = float(self.alpha_values[2])

        return self._get_observation(), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected Discrete(8).")

        traffic_cfg = self.config["traffic"]
        reward_cfg = self.config["reward"]

        # a. Map action to alpha_A.
        alpha_a = float(self.alpha_values[int(action)])

        # b. Allocate PRBs to Ambulance Slice and Ordinary Traffic Slice.
        prb_ambulance = int(round(alpha_a * self.total_prb))
        prb_ordinary = self.total_prb - prb_ambulance

        # c. Generate Ambulance arrivals using the Markov-modulated Poisson model.
        self._update_ambulance_state()
        lambda_ambulance = (
            float(traffic_cfg["lambda_ambulance_emergency"])
            if self.ambulance_emergency
            else float(traffic_cfg["lambda_ambulance_normal"])
        )
        ambulance_packets = self.rng_ambulance_traffic.poisson(
            lambda_ambulance * self.n_ambulance * self.delta_t
        )
        a_ambulance = ambulance_packets * float(traffic_cfg["ambulance_packet_size_mbit"])

        # c. Generate Ordinary Slice arrivals from ordinary vehicles only.
        ordinary_packets = self.rng_ordinary_traffic.poisson(
            float(traffic_cfg["lambda_vehicle"])
            * self.n_ordinary_vehicles
            * self.delta_t
        )
        a_ordinary = ordinary_packets * float(
            traffic_cfg["vehicle_packet_size_mbit"]
        )

        # d. Generate UE-level spectral efficiencies for each slice.
        eta_ambulance = self._sample_spectral_efficiency(self.n_ambulance)
        eta_ordinary = self._sample_spectral_efficiency(self.n_ordinary_vehicles)
        eta_ambulance_avg = float(np.mean(eta_ambulance)) if eta_ambulance.size else 0.0
        eta_ordinary_avg = float(np.mean(eta_ordinary)) if eta_ordinary.size else 0.0

        # e. Apply simple intra-slice PRB sharing.
        prb_per_ambulance_ue = self._intra_slice_prb_share(
            prb_ambulance,
            eta_ambulance.size,
        )
        prb_per_ordinary_ue = self._intra_slice_prb_share(
            prb_ordinary,
            eta_ordinary.size,
        )

        # f. Compute UE capacities and aggregate them into slice capacities.
        ambulance_ue_capacity = prb_per_ambulance_ue * self.r_prb * eta_ambulance
        ordinary_ue_capacity = prb_per_ordinary_ue * self.r_prb * eta_ordinary
        c_ambulance = float(np.sum(ambulance_ue_capacity))
        c_ordinary = float(np.sum(ordinary_ue_capacity))

        q_ambulance_current = self.q_ambulance
        q_ordinary_current = self.q_ordinary
        offered_ambulance = q_ambulance_current + a_ambulance
        offered_ordinary = q_ordinary_current + a_ordinary

        # g. Compute Ambulance latency using the specified RAN-side formula.
        latency_numerator_mbit = offered_ambulance
        latency_denominator_mbps = c_ambulance + self.epsilon
        l_ambulance = latency_numerator_mbit / latency_denominator_mbps
        reconstructed_latency_s = latency_numerator_mbit / latency_denominator_mbps

        # h. Compute SLA violation, Ordinary throughput, PRB utilization, and reward.
        ambulance_sla_violation = int(l_ambulance > self.latency_threshold)
        ambulance_offered_load_mbps = a_ambulance / self.delta_t
        ordinary_offered_load_mbps = a_ordinary / self.delta_t
        ambulance_demand_mbps = offered_ambulance / self.delta_t
        ordinary_demand_mbps = offered_ordinary / self.delta_t
        r_ambulance = min(ambulance_demand_mbps, c_ambulance)
        r_ordinary = min(ordinary_demand_mbps, c_ordinary)

        # Compute the Ordinary next queue before reward so queue and overflow
        # penalties use the exact same transition that is committed below.
        raw_q_ordinary = max(
            0.0,
            offered_ordinary - c_ordinary * self.delta_t,
        )
        ordinary_overflow_mbit = max(
            0.0,
            raw_q_ordinary - self.q_ordinary_max,
        )
        next_q_ordinary = min(self.q_ordinary_max, raw_q_ordinary)
        used_prb_ambulance = self._used_prb(
            offered_ambulance,
            prb_ambulance,
            eta_ambulance,
        )
        used_prb_ordinary = self._used_prb(
            offered_ordinary,
            prb_ordinary,
            eta_ordinary,
        )
        prb_utilization = (used_prb_ambulance + used_prb_ordinary) / self.total_prb

        rho_ambulance = a_ambulance / (self.a_ambulance_max + self.epsilon)
        latency_excess_raw = max(0.0, l_ambulance / self.latency_threshold - 1.0)
        latency_excess_clipped = min(latency_excess_raw, self.latency_excess_clip)
        sla_violation_indicator = ambulance_sla_violation
        if ordinary_demand_mbps == 0.0:
            ordinary_demand_satisfaction = 1.0
        else:
            ordinary_reward_denominator = (
                min(ordinary_demand_mbps, self.ordinary_throughput_target)
                + self.epsilon
            )
            ordinary_demand_satisfaction = min(
                1.0,
                r_ordinary / ordinary_reward_denominator,
            )
        ordinary_throughput_term = ordinary_demand_satisfaction
        ordinary_capacity_limited = int(
            ordinary_demand_mbps > c_ordinary + self.epsilon
        )
        ambulance_capacity_limited = int(
            ambulance_demand_mbps > c_ambulance + self.epsilon
        )
        ordinary_target_eligible = int(
            ordinary_demand_mbps >= self.ordinary_throughput_target
        )
        ordinary_target_violation = int(
            ordinary_target_eligible
            and r_ordinary + self.epsilon < self.ordinary_throughput_target
        )
        resource_waste_term = (
            alpha_a
            if rho_ambulance < float(reward_cfg["rho_ambulance_low"])
            and alpha_a > float(reward_cfg["alpha_ambulance_high"])
            else 0.0
        )
        action_change_term = abs(alpha_a - self.alpha_ambulance_prev)
        ordinary_queue_term = next_q_ordinary / self.q_ordinary_max
        if a_ordinary > 0.0:
            ordinary_overflow_term = min(
                1.0,
                ordinary_overflow_mbit / (a_ordinary + self.epsilon),
            )
        else:
            ordinary_overflow_term = 1.0 if ordinary_overflow_mbit > 0.0 else 0.0
        reward_latency_excess = (
            -float(reward_cfg["w_latency_excess"]) * latency_excess_clipped
        )
        reward_sla_violation = (
            -float(reward_cfg["w_sla_violation"]) * sla_violation_indicator
        )
        reward_ordinary_throughput = (
            float(reward_cfg["w_ordinary_throughput"]) * ordinary_throughput_term
        )
        reward_resource_waste = (
            -float(reward_cfg["w_resource_waste"]) * resource_waste_term
        )
        reward_action_change = (
            -float(reward_cfg["w_action_change"]) * action_change_term
        )
        reward_ordinary_queue = (
            -float(reward_cfg["w_ordinary_queue"]) * ordinary_queue_term
        )
        reward_ordinary_overflow = (
            -float(reward_cfg["w_ordinary_overflow"]) * ordinary_overflow_term
        )
        resource_waste_penalty = float(reward_cfg["w_resource_waste"]) * resource_waste_term
        action_change_penalty = float(reward_cfg["w_action_change"]) * action_change_term
        reward_total = (
            reward_latency_excess
            + reward_sla_violation
            + reward_ordinary_throughput
            + reward_resource_waste
            + reward_action_change
            + reward_ordinary_queue
            + reward_ordinary_overflow
        )

        # i. Update queues using Q_s(t+1)=max(0,Q_s(t)+A_s(t)-C_s(t)*delta_t).
        raw_q_ambulance = max(
            0.0,
            offered_ambulance - c_ambulance * self.delta_t,
        )
        ambulance_overflow_mbit = max(0.0, raw_q_ambulance - self.q_ambulance_max)
        ambulance_queue_cap_hit = int(ambulance_overflow_mbit > 0.0)
        ordinary_queue_cap_hit = int(ordinary_overflow_mbit > 0.0)
        self.q_ambulance = min(self.q_ambulance_max, raw_q_ambulance)
        self.q_ordinary = next_q_ordinary

        # Store current metrics for the next normalized observation.
        self.last_a_ambulance = a_ambulance
        self.last_a_ordinary = a_ordinary
        self.last_l_ambulance = l_ambulance
        self.last_r_ordinary = r_ordinary
        self.last_prb_utilization = prb_utilization
        self.last_eta_ambulance_avg = eta_ambulance_avg
        self.last_eta_ordinary_avg = eta_ordinary_avg
        self.alpha_ambulance_prev = alpha_a
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.episode_length

        info = {
            "alpha_A": alpha_a,
            "prb_ambulance": prb_ambulance,
            "prb_ordinary": prb_ordinary,
            "ambulance_arrival_mbit": a_ambulance,
            "ordinary_arrival_mbit": a_ordinary,
            "ambulance_offered_load_mbps": ambulance_offered_load_mbps,
            "ordinary_offered_load_mbps": ordinary_offered_load_mbps,
            "ambulance_demand_mbps": ambulance_demand_mbps,
            "ordinary_demand_mbps": ordinary_demand_mbps,
            "ambulance_capacity_mbps": c_ambulance,
            "ordinary_capacity_mbps": c_ordinary,
            "ambulance_served_throughput_mbps": r_ambulance,
            "ordinary_served_throughput_mbps": r_ordinary,
            "ambulance_queue_before_mbit": q_ambulance_current,
            "ordinary_queue_before_mbit": q_ordinary_current,
            "latency_numerator_mbit": latency_numerator_mbit,
            "latency_denominator_mbps": latency_denominator_mbps,
            "ambulance_queue_after_mbit": self.q_ambulance,
            "ordinary_queue_after_mbit": self.q_ordinary,
            "ambulance_queue_cap_hit": ambulance_queue_cap_hit,
            "ordinary_queue_cap_hit": ordinary_queue_cap_hit,
            "ambulance_overflow_mbit": ambulance_overflow_mbit,
            "ordinary_overflow_mbit": ordinary_overflow_mbit,
            # Existing queue fields are after-update queue values.
            "ambulance_queue_mbit": self.q_ambulance,
            "ordinary_queue_mbit": self.q_ordinary,
            "ambulance_latency_s": l_ambulance,
            "reconstructed_latency_s": reconstructed_latency_s,
            "ambulance_sla_violation": ambulance_sla_violation,
            "ordinary_throughput_mbps": r_ordinary,
            "ordinary_demand_satisfaction": ordinary_demand_satisfaction,
            "ambulance_capacity_limited": ambulance_capacity_limited,
            "ordinary_capacity_limited": ordinary_capacity_limited,
            "ordinary_target_eligible": ordinary_target_eligible,
            "ordinary_target_violation": ordinary_target_violation,
            "prb_utilization": prb_utilization,
            "latency_excess_raw": latency_excess_raw,
            "latency_excess_clipped": latency_excess_clipped,
            "sla_violation_indicator": sla_violation_indicator,
            "ordinary_throughput_reward": reward_ordinary_throughput,
            "resource_waste_penalty": resource_waste_penalty,
            "action_change_penalty": action_change_penalty,
            "ordinary_queue_term": ordinary_queue_term,
            "ordinary_overflow_term": ordinary_overflow_term,
            "reward_latency_excess": reward_latency_excess,
            "reward_sla_violation": reward_sla_violation,
            "reward_ordinary_throughput": reward_ordinary_throughput,
            "reward_resource_waste": reward_resource_waste,
            "reward_action_change": reward_action_change,
            "reward_ordinary_queue": reward_ordinary_queue,
            "reward_ordinary_overflow": reward_ordinary_overflow,
            "reward_total": float(reward_total),
            "n_ambulance": self.n_ambulance,
            "n_ordinary_vehicles": self.n_ordinary_vehicles,
            "ambulance_emergency": self.ambulance_emergency,
        }

        # j. Return the next observation and Gymnasium step tuple.
        return self._get_observation(), float(reward_total), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        observation = np.asarray(
            [
                self.n_ambulance / self.max_ambulance,
                self.n_ordinary_vehicles / self.max_ordinary_users_norm,
                self.last_a_ambulance / (self.a_ambulance_max + self.epsilon),
                self.last_a_ordinary / (self.a_ordinary_max + self.epsilon),
                self.q_ambulance / self.q_ambulance_max,
                self.q_ordinary / self.q_ordinary_max,
                self.last_l_ambulance / self.latency_threshold,
                self.last_r_ordinary / self.ordinary_throughput_target,
                self.last_prb_utilization,
                self.last_eta_ambulance_avg / self.eta_max,
                self.last_eta_ordinary_avg / self.eta_max,
                self.alpha_ambulance_prev,
                1.0 if self.ambulance_emergency else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, 0.0, 1.0).astype(np.float32)

    def _update_ambulance_state(self) -> None:
        traffic_cfg = self.config["traffic"]
        if self.ambulance_emergency:
            if self.rng_emergency.random() < float(traffic_cfg["p_off"]):
                self.ambulance_emergency = False
                self.n_ambulance = self._sample_normal_ambulance_count()
        elif self.rng_emergency.random() < float(traffic_cfg["p_on"]):
            self.ambulance_emergency = True
            self.n_ambulance = int(
                self.rng_ue.integers(1, self.max_ambulance + 1)
            )

    def _sample_normal_ambulance_count(self) -> int:
        return int(self.rng_ue.integers(0, min(1, self.max_ambulance) + 1))

    def _sample_spectral_efficiency(self, n_users: int) -> np.ndarray:
        if n_users <= 0:
            return np.asarray([], dtype=np.float32)

        channel_cfg = self.config["channel"]
        states = self.rng_channel.choice(
            3,
            size=n_users,
            p=[
                float(channel_cfg["poor_probability"]),
                float(channel_cfg["normal_probability"]),
                float(channel_cfg["good_probability"]),
            ],
        )
        eta = np.empty(n_users, dtype=np.float32)

        poor = states == 0
        normal = states == 1
        good = states == 2
        eta[poor] = self.rng_channel.uniform(
            float(channel_cfg["eta_poor_min"]),
            float(channel_cfg["eta_poor_max"]),
            size=int(np.sum(poor)),
        )
        eta[normal] = self.rng_channel.uniform(
            float(channel_cfg["eta_normal_min"]),
            float(channel_cfg["eta_normal_max"]),
            size=int(np.sum(normal)),
        )
        eta[good] = self.rng_channel.uniform(
            float(channel_cfg["eta_good_min"]),
            float(channel_cfg["eta_good_max"]),
            size=int(np.sum(good)),
        )

        return eta

    def _reset_rng_streams(self, seed: int | None) -> None:
        if seed is None:
            entropy: int | list[int] = self.np_random.integers(
                0,
                np.iinfo(np.uint32).max,
                size=4,
                dtype=np.uint32,
            ).tolist()
        else:
            entropy = int(seed)
        streams = np.random.SeedSequence(entropy).spawn(5)
        (
            self.rng_emergency,
            self.rng_ue,
            self.rng_ambulance_traffic,
            self.rng_ordinary_traffic,
            self.rng_channel,
        ) = tuple(np.random.default_rng(stream) for stream in streams)

    def _intra_slice_prb_share(self, prb: int, n_users: int) -> float:
        if prb <= 0 or n_users <= 0:
            return 0.0
        return prb / n_users

    def _used_prb(self, offered_load: float, allocated_prb: int, eta: np.ndarray) -> float:
        if offered_load <= 0.0 or allocated_prb <= 0 or eta.size == 0:
            return 0.0

        average_prb_rate = self.r_prb * float(np.mean(eta))
        required_prb = offered_load / (average_prb_rate + self.epsilon)
        return float(min(allocated_prb, required_prb))

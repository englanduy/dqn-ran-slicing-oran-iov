import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv


def main() -> None:
    env = RANSlicingEnv("configs/default_config.yaml")
    max_raw = 0.0
    max_clipped = 0.0
    min_reward = float("inf")

    for episode in range(5):
        observation, _ = env.reset(seed=1000 + episode)
        assert observation.shape == (13,)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == (13,)

            raw = float(info["latency_excess_raw"])
            clipped = float(info["latency_excess_clipped"])
            assert clipped <= env.latency_excess_clip + 1e-12
            assert clipped == min(raw, env.latency_excess_clip)

            max_raw = max(max_raw, raw)
            max_clipped = max(max_clipped, clipped)
            min_reward = min(min_reward, float(info["reward_total"]))

    print("Reward clipping validation passed.")
    print(f"latency_excess_clip: {env.latency_excess_clip}")
    print(f"max latency_excess_raw observed: {max_raw}")
    print(f"max latency_excess_clipped observed: {max_clipped}")
    print(f"minimum reward_total observed: {min_reward}")
    if max_raw <= env.latency_excess_clip:
        print("No raw latency excess above the clip was observed in this short random run.")


if __name__ == "__main__":
    main()

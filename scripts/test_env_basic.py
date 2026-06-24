import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv


def main() -> None:
    env = RANSlicingEnv("configs/default_config.yaml")

    observation, _ = env.reset(seed=42)
    assert observation.shape == (13,)
    print("Observation shape:", observation.shape)
    print("Observation min/max:", observation.min(), observation.max())
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    action = env.action_space.sample()
    _, reward, terminated, truncated, info = env.step(action)
    print("Random action:", action)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info keys:", sorted(info.keys()))


if __name__ == "__main__":
    main()

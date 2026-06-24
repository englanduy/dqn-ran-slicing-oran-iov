import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ran_slicing_env import RANSlicingEnv


def main() -> None:
    env = RANSlicingEnv()
    observation, _ = env.reset(seed=42)
    assert observation.shape == (13,)
    check_env(env, skip_render_check=True)
    print("RANSlicingEnv passed Gymnasium environment check.")


if __name__ == "__main__":
    main()

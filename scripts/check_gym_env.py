from gymnasium.utils.env_checker import check_env

from envs.ran_slicing_env import RANSlicingEnv


def main() -> None:
    env = RANSlicingEnv()
    observation, _ = env.reset(seed=42)
    assert observation.shape == (13,)
    check_env(env, skip_render_check=True)
    print("RANSlicingEnv passed Gymnasium environment check.")


if __name__ == "__main__":
    main()

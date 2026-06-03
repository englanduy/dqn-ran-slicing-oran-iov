from envs.ran_slicing_env import RANSlicingEnv


def main() -> None:
    env = RANSlicingEnv("configs/default_config.yaml")

    observation, _ = env.reset(seed=42)
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

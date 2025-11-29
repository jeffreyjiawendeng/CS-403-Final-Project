import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from miniarm_env import MiniArmPendulumEnv


def main():
    # Adjust this path to where your XML actually lives
    here = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(here, "Robot", "miniArm_with_pendulum.xml")

    env = MiniArmPendulumEnv(
        xml_path=xml_path,
        controlled_actuator_name="wrist_pitch",  # choose joint
        frame_skip=5,
        max_episode_steps=1000,
    )

    # Optional: check the environment for Gym compatibility
    check_env(env, warn=True)

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        tau=0.02,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
    )

    # Train
    model.learn(total_timesteps=200_000)

    # Save policy
    save_path = os.path.join(here, "sac_miniarm_pendulum")
    model.save(save_path)
    print(f"Saved SAC model to {save_path}")


if __name__ == "__main__":
    main()

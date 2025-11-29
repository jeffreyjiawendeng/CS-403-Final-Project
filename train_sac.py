# train_sac.py
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from miniarm_env import MiniArmPendulumEnv


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(here, "Robot", "miniArm_with_pendulum.xml")

    env = MiniArmPendulumEnv(
        xml_path=xml_path,
        controlled_actuator_name="wrist_pitch",  # change for a different joint
        frame_skip=5,
        max_episode_steps=2000,
    )

    check_env(env, warn=True)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=0.99,
        tau=0.02,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
    )

    model.learn(total_timesteps=500_000)

    save_path = os.path.join(here, "sac_miniarm_pendulum")
    model.save(save_path)
    print(f"Saved SAC model to {save_path}.zip")


if __name__ == "__main__":
    main()

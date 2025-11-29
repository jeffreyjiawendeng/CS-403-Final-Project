import os
import time

from stable_baselines3 import SAC

from miniarm_env import MiniArmPendulumEnv


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(here, "Robot", "miniArm_with_pendulum.xml")
    model_path = os.path.join(here, "sac_miniarm_pendulum.zip")

    env = MiniArmPendulumEnv(
        xml_path=xml_path,
        controlled_actuator_name="wrist_pitch",
        frame_skip=5,
        max_episode_steps=1000,
    )

    model = SAC.load(model_path, env=env)

    obs, _ = env.reset()
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step:4d}, reward={reward: .3f}, "
            f"upright={info['upright']: .3f}, torque={info['torque']: .3f}"
        )

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()

import os
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class MiniArmPendulumEnv(gym.Env):

    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str,
        controlled_actuator_name: str = "wrist_pitch",
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert os.path.exists(xml_path), f"XML not found: {xml_path}"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Use your keyframe "init_pose" as reset state, if present
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_pose")
        except mujoco.Error:
            key_id = -1
        self._init_key_id = key_id

        # IDs for things we care about
        self.pend_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )
        self.controlled_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, controlled_actuator_name
        )

        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        # Action space: 1D torque command for ONE actuator (normalized -1..1)
        # We'll scale this to the actuator ctrlrange
        ctrl_range = self.model.actuator_ctrlrange[self.controlled_actuator_id]
        self._ctrl_low = float(ctrl_range[0])
        self._ctrl_high = float(ctrl_range[1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: qpos + qvel for all DOFs
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

    # ---------- Utility ----------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _compute_uprightness(self) -> float:
        """
        Returns cos(angle between pendulum local z-axis and world +z).
        1.0  -> perfectly upright
        0.0  -> horizontal
        -1.0 -> upside down
        """
        quat = self.data.body(self.pend_body_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R_flat, quat)
        R = R_flat.reshape(3, 3)
        local_z = R[:, 2]  # third column is local z in world frame
        return float(local_z[2])

    # ---------- Gym API ----------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        if self._init_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._init_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Small random perturbation around initial pose
        noise_scale = 0.01
        self.data.qpos[:] += noise_scale * self.np_random.standard_normal(self.model.nq)
        self.data.qvel[:] += noise_scale * self.np_random.standard_normal(self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Clip normalized action and scale to actuator ctrlrange
        a = np.clip(action, -1.0, 1.0)[0]
        torque = self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

        # Zero all controls and apply torque only to the controlled actuator
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.controlled_actuator_id] = torque

        # Step the simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        upright = self._compute_uprightness()

        # Reward: keep upright, penalize torque and velocities a bit
        vel_cost = 1e-3 * np.sum(self.data.qvel**2)
        torque_cost = 1e-4 * (torque**2)
        reward = upright - vel_cost - torque_cost

        # Terminate if pendulum falls below horizontal
        terminated = upright < 0.0
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "upright": upright,
            "torque": torque,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass

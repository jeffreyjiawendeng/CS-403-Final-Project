# miniarm_env.py
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
        max_episode_steps: int = 2000,  # long enough to see push + recovery
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert os.path.exists(xml_path), f"XML not found: {xml_path}"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Use keyframe "init_pose" if present
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_pose")
        except mujoco.Error:
            key_id = -1
        self._init_key_id = key_id

        # Body & actuator IDs
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

        # ---- Action space: control ONLY one actuator ----
        ctrl_range = self.model.actuator_ctrlrange[self.controlled_actuator_id]
        self._ctrl_low = float(ctrl_range[0])
        self._ctrl_high = float(ctrl_range[1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ---- Observation: qpos + qvel ----
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # ---- Push disturbance parameters (similar to your viewer) ----
        self.push_duration = 1          # total push time (s)
        self.push_force_min = 0.0005      # start around your first checkpoint
        self.push_force_max = 0.0065      # beyond 0.0025 second checkpoint
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)
        self.point = np.zeros(3, dtype=np.float64)

        # episode-specific push schedule
        self.push_start_time = None
        self.push_mid_time = None
        self.push_end_time = None
        self.push_force = None

    # ---------- helpers ----------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _compute_uprightness(self) -> float:
        """
        1.0  -> upright (local z aligned with world +z)
        0.0  -> horizontal
        -1.0 -> upside down
        """
        quat = self.data.body(self.pend_body_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R_flat, quat)
        R = R_flat.reshape(3, 3)
        local_z = R[:, 2]
        return float(local_z[2])

    def _sample_push(self):
        """
        Sample one push event for this episode:
        - Start time ∈ [0.5, 3.5] seconds
        - Force magnitude ∈ [push_force_min, push_force_max]
        """
        self.push_start_time = float(self.np_random.uniform(0.5, 3.5))
        self.push_mid_time = self.push_start_time + 0.5 * self.push_duration
        self.push_end_time = self.push_start_time + self.push_duration
        self.push_force = float(
            self.np_random.uniform(self.push_force_min, self.push_force_max)
        )

    def _apply_external_push(self):
        """Apply +F/-F on the pendulum body, like your viewer code."""
        if self.push_start_time is None:
            return

        t = self.data.time
        if self.push_start_time <= t < self.push_mid_time:
            # +F in y-direction
            self.force[:] = 0.0
            self.torque[:] = 0.0
            self.point[:] = 0.0
            self.force[1] = self.push_force
            mujoco.mj_applyFT(
                self.model,
                self.data,
                self.force,
                self.torque,
                self.point,
                self.pend_body_id,
                self.data.qfrc_applied,
            )
        elif self.push_mid_time <= t < self.push_end_time:
            # -F in y-direction
            self.force[:] = 0.0
            self.torque[:] = 0.0
            self.point[:] = 0.0
            self.force[1] = -self.push_force
            mujoco.mj_applyFT(
                self.model,
                self.data,
                self.force,
                self.torque,
                self.point,
                self.pend_body_id,
                self.data.qfrc_applied,
            )

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

        # small random perturbation
        noise_scale = 0.01
        self.data.qpos[:] += noise_scale * self.np_random.standard_normal(self.model.nq)
        self.data.qvel[:] += noise_scale * self.np_random.standard_normal(self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        # sample a new push for this episode
        self._sample_push()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # normalized action -> torque
        a = float(np.clip(action, -1.0, 1.0)[0])
        torque = self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

        # only the chosen actuator is controlled
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.controlled_actuator_id] = torque

        for _ in range(self.frame_skip):
            self._apply_external_push()
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        upright = self._compute_uprightness()

        # reward: stay upright, penalize motion/torque
        vel_cost = 1e-3 * np.sum(self.data.qvel**2)
        torque_cost = 1e-4 * (torque**2)
        reward = upright - vel_cost - torque_cost

        terminated = upright < 0.0
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "upright": upright,
            "torque": torque,
            "push_force": self.push_force,
            "push_start_time": self.push_start_time,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

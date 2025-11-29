# YourControlCode.py
import os
import mujoco
import numpy as np
from stable_baselines3 import SAC


class YourCtrl:
    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        model_name: str = "sac_miniarm_pendulum.zip",
        actuator_name: str = "wrist_pitch",
    ):
        """
        Controller that uses a trained SAC policy for a single joint.
        Falls back to PD control if the SAC model can't be loaded.
        """
        self.m = m
        self.d = d
        self.actuator_name = actuator_name

        # Find actuated joint
        self.actuator_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_name
        )

        # Torque limits for this actuator
        ctrl_range = self.m.actuator_ctrlrange[self.actuator_id]
        self.ctrl_low = float(ctrl_range[0])
        self.ctrl_high = float(ctrl_range[1])

        # Where the SAC model is stored (same folder as this file)
        base_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(base_dir, model_name)

        if not os.path.exists(model_path):
            print(
                f"[YourCtrl] WARNING: SAC model not found at {model_path}. "
                f"Falling back to PD controller."
            )
            self.model = None

            # PD fallback: original behavior
            self.init_qpos = d.qpos.copy()
            self.kp = 150.0
            self.kd = 5.2
        else:
            print(f"[YourCtrl] Loading SAC model from {model_path}")
            self.model = SAC.load(model_path)
            self.model.policy.eval()

            # Observation size for SAC (qpos + qvel)
            self.obs_dim = self.m.nq + self.m.nv

    # ---------- helpers ----------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.d.qpos, self.d.qvel])

    def _sac_step(self):
        """Use SAC to compute torque for the chosen actuator."""
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)

        a = float(np.clip(action, -1.0, 1.0)[0])

        torque = self.ctrl_low + (a + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        # Zero all controls, only set one actuator
        self.d.ctrl[:] = 0.0
        self.d.ctrl[self.actuator_id] = torque

    def _pd_step(self):
        """Fallback PD on the first 6 joints (your original CtrlUpdate)."""
        for i in range(6):
            self.d.ctrl[i] = (
                150.0 * (self.init_qpos[i] - self.d.qpos[i])
                - 5.2 * self.d.qvel[i]
            )

    # ---------- called from Run_PendulumEnv each step ----------

    def CtrlUpdate(self):
        if self.model is not None:
            self._sac_step()
        else:
            self._pd_step()
        return True

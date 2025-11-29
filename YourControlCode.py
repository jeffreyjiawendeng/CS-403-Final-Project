import os
import mujoco
import numpy as np
from stable_baselines3 import SAC


class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData,
                 model_name: str = "sac_miniarm_pendulum.zip",
                 actuator_name: str = "wrist_pitch"):
        """
        Controller that uses a trained SAC policy.

        If the SAC model file is not found, falls back to the old PD controller.
        """
        self.m = m
        self.d = d
        self.actuator_name = actuator_name

        # Find the actuator we want to control (only one joint)
        self.actuator_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_name
        )

        # Torque range for that actuator
        ctrl_range = self.m.actuator_ctrlrange[self.actuator_id]
        self.ctrl_low = float(ctrl_range[0])
        self.ctrl_high = float(ctrl_range[1])

        # Where to load the SAC model from
        base_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(base_dir, model_name)

        if not os.path.exists(model_path):
            # Fallback: PD control on all 6 joints (your original behavior)
            print(f"[YourCtrl] WARNING: SAC model not found at {model_path}. "
                  f"Falling back to PD controller.")
            self.model = None

            self.init_qpos = d.qpos.copy()
            self.kp = 150.0
            self.kd = 5.2
        else:
            print(f"[YourCtrl] Loading SAC model from {model_path}")
            # Load SAC without attaching a Gym env (we just use predict())
            self.model = SAC.load(model_path)
            # Not strictly necessary, but ensures policy is in eval mode
            self.model.policy.eval()

            # Observation dimension: qpos + qvel (must match training)
            self.obs_dim = self.m.nq + self.m.nv

    # --------- helpers ---------

    def _get_obs(self) -> np.ndarray:
        """Build observation: concatenated qpos and qvel (same as training env)."""
        return np.concatenate([self.d.qpos, self.d.qvel])

    def _sac_step(self):
        """Compute torque using SAC policy and apply only to one actuator."""
        obs = self._get_obs()
        # SAC expects obs shape (obs_dim,) or (1, obs_dim); SB3 handles both
        action, _ = self.model.predict(obs, deterministic=True)

        # Action was trained with Box(low=-1, high=1, shape=(1,))
        a = float(np.clip(action, -1.0, 1.0)[0])

        # Map normalized action [-1, 1] to actual torque range
        torque = self.ctrl_low + (a + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        # Zero all controls, only set the chosen actuator
        self.d.ctrl[:] = 0.0
        self.d.ctrl[self.actuator_id] = torque

    def _pd_step(self):
        """Your original PD controller on the first 6 joints."""
        for i in range(6):
            self.d.ctrl[i] = 150.0 * (self.init_qpos[i] - self.d.qpos[i]) \
                             - 5.2 * self.d.qvel[i]

    # --------- called each simulation step from Run_PendulumEnv ---------

    def CtrlUpdate(self):
        if self.model is not None:
            self._sac_step()
        else:
            self._pd_step()
        return True

import mujoco
import numpy as np
from scipy.linalg import inv, eig
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from mujoco import viewer
from stable_baselines3.common.callbacks import BaseCallback
class RenderCallback(BaseCallback):
    def __init__(self, freq=1):
        super().__init__()
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            try:
                self.training_env.envs[0].render()
            except:
                pass
        return True
    
class myenv(gym.Env):
  metadata = {"render_modes": ["human"]}

  def __init__(self, model, data, render_mode=None):
    super().__init__()

    # Load MuJoCo model + data
    self.model = model
    self.data = data
    self.push_force= 0.0005
    self.render_mode = render_mode
    self.balance_count = 0
    self.next_pushing_time = 0.5
    
    if render_mode == "human":
        self.renderer = mujoco.viewer.launch_passive(self.model, self.data)
    else:
        self.renderer = mujoco.Renderer(self.model, width=700, height=400)
    
    actuators = self.model.nu  # number of actuators
    # can be scaled later
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(actuators,), dtype=np.float32
    )
    
    # can experiment to include other inputs
    obs_dim = self.model.nq + self.model.nv
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    
  def _get_obs(self):
      # can change to include other inputs
      # qpos and qvel are the typical MuJoCo observations
      return np.concatenate([self.data.qpos, self.data.qvel], dtype=np.float32)
  
  def step(self, action):
      # Scale action to MuJoCo actuator range if needed
      
      # Here we assume action is directly torque
      min_ctrl = self.model.actuator_ctrlrange[:, 0]
      max_ctrl = self.model.actuator_ctrlrange[:, 1]

      # may have to adjust for scaling
      scaled = min_ctrl + (action + 1) * 0.5 * (max_ctrl - min_ctrl)
      self.data.ctrl[:] = scaled
      # self.data.ctrl[:] = action
      
      force = np.zeros((3,))
      torque = np.zeros((3,))
      point = np.zeros((3,))

      pushing_trial_gap = 4.0
      pushing_duration = 0.1
      pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")  # Ensure correct object type
                    
      # push pendulum positive direction first
      if(self.next_pushing_time < self.data.time and self.data.time < self.next_pushing_time + pushing_duration/2):
            
        force[1] = self.push_force
        mujoco.mj_applyFT(self.model, self.data, force, torque, point, pend_id, self.data.qfrc_applied)

        # push pendulum negative direction second
      if(self.next_pushing_time + pushing_duration/2 < self.data.time and self.data.time < self.next_pushing_time + pushing_duration):
        force[1] = -self.push_force
        mujoco.mj_applyFT(self.model, self.data, force, torque, point, pend_id, self.data.qfrc_applied)
        
        # increment balance count on success and increase pushing force and repeat above
    #   if(self.next_pushing_time + pushing_duration + 0.5 < self.data.time):
    #     self.balance_count += 1
    #     self.next_pushing_time += pushing_trial_gap
    #     self.push_force += 0.001
    #     print("Balance Count: ", self.balance_count)
    #     print("Next Pushing Force: ", self.push_force)

      # Step physics
      mujoco.mj_step(self.model, self.data)
      if self.render_mode == "human":
        self.renderer.sync()
      # Compute reward
      reward = self.compute_reward()
      
      # print(reward)
      # Check termination
      terminated = self.is_terminated()
      truncated  = False  # unless you're using time limits

      return self._get_obs(), reward, terminated, truncated, {}
  def get_local_z(self):
      pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")  # Ensure correct object type
      quat = self.data.body(pend_id).xquat
      R_flat = np.empty(9, dtype=np.float64)
      mujoco._functions.mju_quat2Mat(R_flat, quat)
      R = R_flat.reshape(3, 3)
      # The local z-axis in world coordinates is the third column of the rotation matrix.
      local_z = R[:, 2]
      return local_z[2]

  # need to fix this
  def compute_reward(self):
      # Example: reward for keeping an inverted pendulum upright
      # angle = self.data.qpos[0]
      z = self.get_local_z()
      # reward = 1.0 - 0.1 * angle**2
      reward = z
      
      return reward

  def is_terminated(self):
      # Example termination rule
      # checks if robot fails, ie pendulum ends up on the ground

      return np.any(np.abs(self.data.qpos[6]) > np.pi/1.2) or self.get_local_z() < 0 or self.data.time > 1.1
      # return abs(angle) > np.pi/2  # pendulum falls

  def render(self):
      if self.render_mode == "human":
          
          return
      self.renderer.render()
      return self.renderer.read_pixels()

  def close(self):
      pass
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = 0.01 * self.np_random.standard_normal(self.model.nq)
    self.data.qvel[:] = 0.01 * self.np_random.standard_normal(self.model.nv)
    return self._get_obs(), {}

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Original and modified model paths
    robot_model = os.path.join(dir_path, "./Robot/miniArm_with_pendulum.xml")
    
    mj_model = mujoco.MjModel.from_xml_path(robot_model)
    mj_data = mujoco.MjData(mj_model)

    e = myenv(mj_model, mj_data, render_mode="human")

    model = SAC(policy="MlpPolicy",env= e, verbose=1)
    model.learn(total_timesteps=10000, callback=RenderCallback(freq=1))
    model.save("SAC_test")
    # save the model in cache so it can be imported later
if __name__ == '__main__':
    main()
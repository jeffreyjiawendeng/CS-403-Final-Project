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
from stable_baselines3.common.vec_env import SubprocVecEnv

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

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SACMetricsCallback(BaseCallback):
    """
    Logs:
      - reward history
      - reward range (min/max)
      - episode length history
      - actor/critic losses
      - additional SAC metrics
    """

    def __init__(self, verbose=1, print_interval=1000):
        super().__init__(verbose)
        self.print_interval = print_interval

        # --- histories ---
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        # self.actor_losses = []
        # self.critic_losses = []
        # self.entropies = []
        # self.q_values = []

        # internal temp storage
        self._current_ep_reward = 0.0
        self._current_ep_len = 0


    def _on_step(self) -> bool:
        """Runs every environment step."""

        reward = self.locals["rewards"][0]  # support VecEnv
        done = self.locals["dones"][0]

        # Track per-step reward
        self.rewards.append(reward)
        self._current_ep_reward += reward
        self._current_ep_len += 1

        # Episode ended
        if done:
            self.episode_rewards.append(self._current_ep_reward)
            self.episode_lengths.append(self._current_ep_len)

            if self.verbose > 0:
                print(f"[Episode Done] reward={self._current_ep_reward:.2f}, length={self._current_ep_len}")

            self._current_ep_reward = 0.0
            self._current_ep_len = 0

        # Print running statistics
        if self.n_calls % self.print_interval == 0 and len(self.rewards) > 0:
            print("\n================ TRAINING METRICS ================")
            print(f"Steps:              {self.n_calls}")
            print(f"Reward min/max:     {np.min(self.rewards):.2f} / {np.max(self.rewards):.2f}")
            if len(self.episode_rewards) > 0:
                print(f"Mean ep reward:     {np.mean(self.episode_rewards[-50:]):.2f} (last 50)")
                print(f"Mean ep length:     {np.mean(self.episode_lengths[-50:]):.1f}")
            # if len(self.actor_losses) > 0:
            #     print(f"Actor loss (last):  {self.actor_losses[-1]:.4f}")
            # if len(self.critic_losses) > 0:
            #     print(f"Critic loss (last): {self.critic_losses[-1]:.4f}")
            # print(f"Entropy (last):     {self.entropies[-1] if len(self.entropies)>0 else 'N/A'}")
            print("=================================================\n")

        # Return True to continue training
        return True


    # def _on_training_step(self) -> None:
    #     """Logs SAC losses and other metrics."""
    #     # SAC has: actor_loss, critic_loss, ent_coef_loss, ent_coef
    #     if "actor_loss" in self.locals:
    #         self.actor_losses.append(float(self.locals["actor_loss"]))

    #     if "critic_loss" in self.locals:
    #         self.critic_losses.append(float(self.locals["critic_loss"]))

    #     # Temperature (entropy) information
    #     if "log_ent_coef" in self.locals:
    #         ent_coef = np.exp(self.locals["log_ent_coef"])
    #         self.entropies.append(float(ent_coef))

    #     # Q-values (optional, if available)
    #     if "q_values" in self.locals:
    #         self.q_values.append(self.locals["q_values"].mean())


    def get_stats(self):
        """Return all collected metrics."""
        return {
            "rewards": self.rewards,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            # "actor_losses": self.actor_losses,
            # "critic_losses": self.critic_losses,
            # "entropies": self.entropies,
            # "q_values": self.q_values,
        }

class simenv(gym.Env):
  # metadata = {"render_modes": ["human"]}

  def __init__(self, model_path, render_mode=None):
    super().__init__()

    # Load MuJoCo model + data
    self.model = mujoco.MjModel.from_xml_path(model_path)
    self.data = mujoco.MjData(self.model)
    self.push_force= 0.0005
    self.render_mode = render_mode
    self.balance_count = 0
    self.next_pushing_time = 0.5
    # self.pushing_duration = 0.1
    if render_mode == "human":
        self.renderer = mujoco.viewer.launch_passive(self.model, self.data)
    else:
        self.renderer = mujoco.Renderer(self.model)
    
    actuators = self.model.nu  # number of actuators
    # can be scaled later
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(actuators,), dtype=np.float32
    )
    
    # can experiment to include other inputs
    obs_dim = self.model.nq + self.model.nv
    # obs_dim = self.model.nq + self.model.nv + self.model.nv
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    
  def _get_obs(self):
      # can change to include other inputs
      # qpos and qvel are the typical MuJoCo observations
      # self.data.qfrc_inverse
      # return np.concatenate([self.data.qpos, self.data.qvel, self.data.qfrc_inverse], dtype=np.float32)
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
      
      mujoco.mj_step(self.model, self.data)

      # Step physics
    #   if self.render_mode == "human":
    #     self.renderer.sync()
      # Compute reward
      reward = self.compute_reward()
      
      # print(reward)
      # Check termination
      terminated = self.is_terminated()
      
      
      truncated  = self.data.time > 1.1  # unless you're using time limits

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
      
      reward = -1*(self.data.qpos[6] ** 2 + 0.01 * np.linalg.norm(self.data.ctrl))
      
      # reward should be based on how long 
      return reward

  def is_terminated(self):
      # Example termination rule
      # checks if robot fails, ie pendulum ends up on the ground

      return self.get_local_z() < 0
      # return abs(angle) > np.pi/2  # pendulum falls

  def render(self):
      if self.render_mode == "human":
          
          return
    #   self.renderer.render()
      return self.renderer.read_pixels()

  def close(self):
      pass
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)
    # self.data.qpos[:] = 0.01 * self.np_random.standard_normal(self.model.nq)
    # self.data.qvel[:] = 0.01 * self.np_random.standard_normal(self.model.nv)
    # self.next_pushing_time = 0.5 + 0.01 * self.np_random.standard_normal()
    # self.push_force = 0.0005 +  0.01 * self.np_random.standard_normal()
    return self._get_obs(), {}

class randomenv(gym.Env):
  # metadata = {"render_modes": ["human"]}

  def __init__(self, model_path, render_mode=None):
    super().__init__()

    # Load MuJoCo model + data
    self.model = mujoco.MjModel.from_xml_path(model_path)
    self.data = mujoco.MjData(self.model)
    # self.push_force= 0.0005
    self.render_mode = render_mode
    self.np_random = np.random.default_rng()
    # self.balance_count = 0
    
    # randomize position
    self.data.qpos[:] = 0.01 * self.np_random.standard_normal(self.model.nq)
    # randomize velocity
    self.data.qvel[:] = 0.01 * self.np_random.standard_normal(self.model.nv)
    # randomize pushing time
    self.next_pushing_time = 0.5 + self.np_random.uniform(-0.05, 0.05)
    # randomize push force magnitude
    self.push_force = self.np_random.uniform(0.005, 0.02)
    self.pushing_duration = 0.05 + self.np_random.uniform(-0.005,0.005)
    self.direction = np.random.choice([-1, 1])
    if render_mode == "human":
        self.renderer = mujoco.viewer.launch_passive(self.model, self.data)
    else:
        self.renderer = mujoco.Renderer(self.model)
    
    actuators = self.model.nu  # number of actuators
    # can be scaled later
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(actuators,), dtype=np.float32
    )
    
    # can experiment to include other inputs
    obs_dim = self.model.nq + self.model.nv
    # obs_dim = self.model.nq + self.model.nv + self.model.nv
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    
  def _get_obs(self):
      # can change to include other inputs
      # qpos and qvel are the typical MuJoCo observations
      # self.data.qfrc_inverse
      # return np.concatenate([self.data.qpos, self.data.qvel, self.data.qfrc_inverse], dtype=np.float32)
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

      # pushing_trial_gap = 4.0
      pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")  # Ensure correct object type
                    
      # push pendulum positive direction first
      if(self.next_pushing_time < self.data.time and self.data.time < self.next_pushing_time + self.pushing_duration):
            
        force[1] = self.push_force * self.direction
        mujoco.mj_applyFT(self.model, self.data, force, torque, point, pend_id, self.data.qfrc_applied)

      
      mujoco.mj_step(self.model, self.data)

      # Step physics
      if self.render_mode == "human":
        self.renderer.sync()
      # Compute reward
      reward = self.compute_reward()
      
      # print(reward)
      # Check termination
      terminated = self.is_terminated()
      
      truncated = False
      # truncated  = self.data.time > 1.1  # unless you're using time limits

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
      # z = self.get_local_z()
      
      # implement colision else where
      # reward should be the pendulum status, height of the EE frame, 
      # reward = 1.0 - 0.1 * angle**2
      
      reward = -1*(self.data.qpos[6] ** 2 + 0.01 * np.linalg.norm(self.data.ctrl))
      # reward = -1*(self.data.qpos[6]**2)
      # reward should be based on how long 
      return reward

  def is_terminated(self):
      # Example termination rule
      # checks if robot fails, ie pendulum ends up on the ground

      return self.get_local_z() < 0
      # return abs(angle) > np.pi/2  # pendulum falls

  def render(self):
      if self.render_mode == "human":
          
          return
    #   self.renderer.render()
      return self.renderer.read_pixels()

  def close(self):
      pass
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)
    # randomize position
    self.data.qpos[:] = 0.01 * self.np_random.standard_normal(self.model.nq)
    # randomize velocity
    self.data.qvel[:] = 0.01 * self.np_random.standard_normal(self.model.nv)
    # randomize pushing time
    self.next_pushing_time = 0.5 + self.np_random.uniform(-0.05, 0.05)
    # randomize push force magnitude
    self.push_force = self.np_random.uniform(0.005, 0.02)
    
    self.pushing_duration = 0.05 + self.np_random.uniform(-0.005,0.005)
    self.direction = np.random.choice([-1, 1])
   #  mujoco.mj_forward(self.model, self.data)

    return self._get_obs(), {}

def make_simenv(xml_path):
    def _init():
        env = simenv(xml_path)
        return env
    return _init

def make_randenv(xml_path):
    def _init():
        env = randomenv(xml_path)
        return env
    return _init

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Original and modified model paths
    robot_model = os.path.join(dir_path, "./Robot/miniArm_with_pendulum.xml")
    
    # mj_model = mujoco.MjModel.from_xml_path(robot_model)
    # mj_data = mujoco.MjData(mj_model)
    num_envs = 8
    env = SubprocVecEnv([make_randenv(robot_model) for _ in range(num_envs)])
    # env = randomenv(robot_model, "human")

    model = SAC(policy="MlpPolicy",env= env, verbose=0)
    # model = SAC.load("SAC_random", env)
    # model.load_replay_buffer("SAC_random_buffer")
    
    # model.learn(total_timesteps=10000, callback=SACMetricsCallback())
    time_steps = 500_000
    chunks = 100_000
    for i in range(int(time_steps/chunks)):
        print(f"check point {i}")
        model.learn(total_timesteps=chunks, callback=SACMetricsCallback())
        model.save(f"SAC_random_anglereward")
        model.save_replay_buffer( "SAC_random_anglereward_buffer")
    # save the model in cache so it can be imported later
if __name__ == '__main__':
    main()
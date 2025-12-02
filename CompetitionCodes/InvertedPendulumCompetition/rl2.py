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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import matplotlib.pyplot as plt
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


class randomenv(gym.Env):
  metadata = {"render_modes": ["human"]}

  def __init__(self, model_path, render_mode=None):
    super().__init__()

    # Load MuJoCo model + data
    self.model = mujoco.MjModel.from_xml_path(model_path)
    self.data = mujoco.MjData(self.model)
    # self.balance_count = 0
    self.next_pushing_time = 0.5
    # randomize push force magnitude
    self.push_force = 0.0005
    self.pushing_duration = 0.05
    self.render_mode = render_mode

    
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
    #obs_dim = self.model.nq + self.model.nv
    obs_dim = 34
    # obs_dim = self.model.nq + self.model.nv + self.model.nv
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    
  def _get_obs(self):
      # can change to include other inputs
      q = self.data.qpos[:6]
      dq = self.data.qvel[:6]
      ee_bid = self.model.body('EE_Frame').id
      ee_pos = self.data.body(ee_bid).xpos.copy()      # (3,)
      ee_quat = self.data.body(ee_bid).xquat.copy()     # (4,)
      vel = np.zeros(6)
      mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, ee_bid, vel, flg_local=False)
      ee_lin_vel = vel[:3]
      ee_ang_vel = vel[3:]
      
      pend_jid = self.model.joint('pend_roll').id
      theta     = self.data.joint(pend_jid).qpos
      sin_theta =np.sin(theta)
      cos_theta = np.cos(theta)

      theta_dot = self.data.joint(pend_jid).qvel
      pend_bid = self.model.body('pendulum').id
      pend_pos = self.data.body(pend_bid).xpos.copy()

      ee_to_pendulum = pend_pos - ee_pos

      return np.concatenate(
          [q, 
           dq,
           ee_pos,
           ee_quat,
           ee_lin_vel,
           ee_ang_vel,
           sin_theta,
           cos_theta,
           theta_dot,
           pend_pos,
           ee_to_pendulum
           ], 
          dtype=np.float32)
  
  def step(self, action):
      # Scale action to MuJoCo actuator range if needed
      
      # Here we assume action is directly torque
      # min_ctrl = self.model.actuator_ctrlrange[:, 0]
      max_ctrl = self.model.actuator_ctrlrange[:, 1]

      # may have to adjust for scaling
      scaled = action * max_ctrl
      self.data.ctrl[:] = scaled
      # self.data.ctrl[:] = action
      
      force = np.zeros((3,))
      torque = np.zeros((3,))
      point = np.zeros((3,))

      # pushing_trial_gap = 4.0
      pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")  # Ensure correct object type
                    
      # push pendulum positive direction first
      if(self.next_pushing_time < self.data.time and self.data.time < self.next_pushing_time + self.pushing_duration):
            
        force[1] = self.push_force
        mujoco.mj_applyFT(self.model, self.data, force, torque, point, pend_id, self.data.qfrc_applied)

      if(self.next_pushing_time + self.pushing_duration/2 < self.data.time and self.data.time < self.next_pushing_time + self.pushing_duration):
        force[1] = -1*self.push_force
        mujoco.mj_applyFT(self.model, self.data, force, torque, point, pend_id, self.data.qfrc_applied)
      mujoco.mj_step(self.model, self.data)

      # Step physics
    #   if self.render_mode == "human":
    #     self.renderer.sync()
      # Compute reward
      reward = self.compute_reward()
      
      # Check termination
      terminated = self.is_terminated()
      if terminated:
          reward = -50
      truncated = False

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
    # min_ctrl = self.model.actuator_ctrlrange[:, 0]
    max_ctrl = self.model.actuator_ctrlrange[:, 1]

    ee_bid = self.model.body('EE_Frame').id
    vel = np.zeros(6)

    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, ee_bid, vel, flg_local=False)

    angle = self.data.qpos[6]
    angle_vel = self.data.qvel[6]
    action = self.data.ctrl
    
    # reward 6
    # reward = np.cos(angle) ** 2 - 0.01*angle_vel**2 - 0.1 * np.linalg.norm(action) -0.01 * np.linalg.norm(vel)
    
    # reward 7
    reward = -1*(angle**2 + 0.1 * angle_vel**2) -0.1 * np.linalg.norm(vel) - 0.1 * np.linalg.norm(action)
    # reward = np.cos(angle) ** 2
    # reward = -1*(self.data.qpos[6] ** 2 + 0.01 * np.linalg.norm(self.data.ctrl))
    # hyper parameters weight for each: 0.1, 0.01, 0.001 --> 9 combinations
    
    return reward

  def is_terminated(self):
      return self.get_local_z() < 0

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
    
    return self._get_obs(), {}

def make_randenv(xml_path):
    def _init():
        env = randomenv(xml_path)
        return env
    return _init

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Original and modified model paths
    robot_model = os.path.join(dir_path, "./Robot/miniArm_with_pendulum.xml")
    num_envs = 8
    env = SubprocVecEnv([make_randenv(robot_model) for _ in range(num_envs)])
    # env =randomenv(robot_model)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = SAC(policy="MlpPolicy",env= env, 
        learning_rate = 0.0001,
        batch_size=512,
        gamma=0.99,
        learning_starts=12000,
        policy_kwargs=dict(
            net_arch=dict(
                pi = [256, 256, 256],
                qf = [256, 256, 256],
            )
        ),
        target_entropy = -6,
        tensorboard_log="./logs_sac2/",
        verbose=0)
    # model = SAC(policy="MlpPolicy",env= env)
    # model = SAC.load("SAC_random", env)
    # # # model.load_replay_buffer("SAC_random_buffer")
    
    callback = SACMetricsCallback()

    # model.learn(total_timesteps=10000, callback = callback)
    time_steps = 5_000_000
    chunks = 100_000
    model.learn(total_timesteps=time_steps, callback = callback)
    # for i in range(int(time_steps/chunks)):
    #     print(f"check point {i}")
    #     model.learn(total_timesteps=chunks, callback = callback)
    model.save(f"SAC_hp_obs_reward7")
    model.save_replay_buffer( "SAC_hp_obs_reward7_buffer")
    
    # rewards = callback.episode_rewards
    # print(rewards)
    # print(callback.episode_lengths)
    # return rewards
if __name__ == '__main__':
    main()
    # plt.plot(range(len(rewards)), rewards)
    # plt.xlabel("Timesteps")
    # plt.ylabel("Episode Reward")
    # plt.title("SB3 Reward vs Timesteps")
    # plt.grid(True)
    # plt.savefig("test.png")
    # plt.show()

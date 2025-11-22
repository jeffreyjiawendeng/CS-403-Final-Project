import mujoco
from stable_baselines3 import SAC
import os
import numpy as np
class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    # model
    self.m = m
    # data
    self.d = d
    

    # e = myenv(self.m, self.d, render_mode=None)
    # initial positions
    self.init_qpos = d.qpos.copy()
    self.model = SAC.load("SAC_random_anglereward")
    # Control gains (using similar values to CircularMotion)
    # ??????
    self.kp = 50.0
    self.kd = 3.0
  
    

  def CtrlUpdate(self):
    # unit of control is based on gear, but it looks like torque
    # if RL/ML, we can think of action space be be 6 numbers in R. so output would be R^6
    # input, can be orientation (and maybe velocity) of each part + what torque each joint feels (this would require sensors, so not likely)
    # what do we include in state space?
    # print(self.d.qpos)
    # for i in range(6):
    obs = np.concatenate([self.d.qpos, self.d.qvel], dtype=np.float32)
      # self.d.ctrl[i] = self.kp*(self.init_qpos[i] - self.d.qpos[i])  + self.kd*(-1*self.d.qvel[i])
      
      
    action, _ = self.model.predict(obs)
    # print(action)
    for i in range(6):
      self.d.ctrl[i] = action[i]
    # self.d.ctrl[0] = -1* self.d.qfrc_applied[0]
    # self.d.ctrl[1] = -1*self.d.qfrc_applied[1]
    # self.d.ctrl[4] = 10
    # print(self.d.qfrc_applied)
    return True 


# some questions before answering:
# what should be the action space of a policy? - R^6, real number value representing torque squashed in appropriate range
# what should be a state space of a policy? --> can be a lot of things, will have to prob experiment with them
  # makes most sense to have orientation (in quaternion) and position () and qfrc_inverse (or actuator_force and qfrc_applied)
  # in total this would be 
# is my environment broken? --> no, you may have to adjust it for reset
# we can do either RL or some complex dynamics solution
# for dynamics solution, theoretically you can just move to counteract the applied force/torque, but the issue is reaction time, machine has to be one step ahead


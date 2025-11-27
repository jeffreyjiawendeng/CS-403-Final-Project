import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
# from stable_baselines3 import SAC

import numpy as np
from scipy.linalg import solve_discrete_are
import mujoco

  
class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    # model
    self.model = m
    # data
    self.data = d
    
    
    # actuator_moment = np.zeros((self.m.nu, self.m.nv))
    # mujoco.mju_sparse2dense(
    # actuator_moment,
    # self.d.actuator_moment.reshape(-1),
    # self.d.moment_rownnz,
    # self.d.moment_rowadr,
    # self.d.moment_colind.reshape(-1),
    # )
    # mujoco.mj_forward(self.model, self.data)
    self.data.qacc = 0  # Assert that there is no the acceleration.
    mujoco.mj_inverse(self.model, self.data)
    # print(self.data.qfrc_inverse)
    
    # may have to save a specific value
    self.initial_qpos = self.data.qpos.copy()
    
    # the force we want to stabilize I think
    
    # may have to save a specific value
    self.force = self.data.qfrc_inverse.copy()
    
    
    self.data.ctrl = self.force[:6]
    self.data.qpos = self.initial_qpos
    
    # gets A and B values
    A = np.zeros((2*self.model.nv, 2*self.model.nv))
    B = np.zeros((2*self.model.nv, self.model.nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B, None, None)
    
    nu =self.model.nu  # Alias for the number of actuators.
    R = np.eye(nu)
    
    # Q = np.eye(2*self.model.nv)
    # 1, 3, 4
    Q = np.diag([10,100,10,100,100,10,100, 1, 1, 1, 1, 1, 1, 1])
    # print(self.model.nv)
    
    # different Q values
    # cost should be 
    
    
    # Solve discrete Riccati equation.
    P = solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    self.K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # print(self.initial_qpos)
    
    # may have to reset with initialization
    mujoco.mj_resetData(self.model, self.data)

   
  def CtrlUpdate(self):
    # self.data.qacc = 0  # Assert that there is no the acceleration.

    dq = np.zeros(self.model.nv)
    mujoco.mj_differentiatePos(self.model, dq, 1, self.initial_qpos, self.data.qpos)
    dx = np.hstack((dq, self.data.qvel)).T
    self.data.ctrl = self.force[:6] - self.K @ dx
    # self.data.ctrl = np.clip(self.force[:6] - self.K @ dx,
    #                      self.model.actuator_ctrlrange[:,0],
    #                      self.model.actuator_ctrlrange[:,1])
    # self.data.ctrl[:] = self.force[:6]
    return True 



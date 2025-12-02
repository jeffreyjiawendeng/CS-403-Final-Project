import numpy as np
from scipy.linalg import solve_discrete_are
import mujoco



  
class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    # model
    self.model = m
    # data
    self.data = d
    
    
    self.data.qacc = 0  
    mujoco.mj_inverse(self.model, self.data)
    
    self.initial_qpos = self.data.qpos.copy()
   
    self.force = self.data.qfrc_inverse.copy()
    
    self.data.ctrl = self.force[:6]
   
    A = np.zeros((2*self.model.nv, 2*self.model.nv))
    B = np.zeros((2*self.model.nv, self.model.nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B, None, None)
    
       
    nu =self.model.nu  
    R = np.diag([1,1,1,1,1,5])
    
    Q = np.diag([100,500,100,500,500,100,500, 1, 1, 1, 1, 1, 1, 1])
    
    
    P = solve_discrete_are(A, B, Q, R)

    self.K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    
    mujoco.mj_resetData(self.model, self.data)

   
  def CtrlUpdate(self):
   

    

    dq = np.zeros(self.model.nv)
    mujoco.mj_differentiatePos(self.model, dq, 1, self.initial_qpos, self.data.qpos)
    dx = np.hstack((dq, self.data.qvel)).T
    cmd = (self.force[:6] - self.K @ dx)
    self.data.ctrl[:] = cmd
    # self.data.ctrl = np.clip(cmd,
    #                      self.model.actuator_ctrlrange[:,0],
    #                      self.model.actuator_ctrlrange[:,1])
    # self.data.ctrl[:] = self.force[:6]
    # print(self.force) 
    return True 



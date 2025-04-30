#!/usr/bin/python3
#-*-coding: UTF-8 -*-
from msg_py import joystick_pb2,robot_pb2,chassis_pb2,pose_pb2,imu_pb2
import crpilot
import pycrmw
import time
import sys,os
import signal
import logging
from collections import deque
import numpy as np
import json


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/var/log/humanoid/observation.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s: %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

pycrmw.Init(sys.argv)

class ObservationNode:
  def __init__(self):
    self.count = 0
    self.motor_pos_list = []
    self.motor_vel_list = []
    self.motor_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.double)  # 单位: rad/s
    self.motor_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.double)  # 单位: rad/s

    self.imu_reader = self.node.CreateReader("/imu",self.OnImu)
    self.humanoid_info_reader_hip = self.node.CreateReader("/humanoid_info/hip",self.OnHumanoidHipInfo)
    self.humanoid_info_reader_leg = self.node.CreateReader("/humanoid_info/leg",self.OnHumanoidLegInfo)
    self.humanoid_info_reader_ankle = self.node.CreateReader("/humanoid_info/ankle",self.OnHumanoidAnkleInfo)

    self.joystick_reader = self.node.CreateReader("/ctrl_cmd/joystick",self.OnJoycontrol)


  def OnImu(self,t:imu_pb2.Imu):
    if not self.CheckForImuData(t):
        self.imu = t
        print('self.imu.timestamp',t.timestamp)
        self.imu_recv = True
        #print('IMU data got............')
        self.TryToQueueObservation()

  def OnHumanoidHipInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_hip_info = t     
      self.humanoid_hip_info_recv = True
      print('self.humanoid_hip_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      self.TryToQueueObservation()

  def OnHumanoidLegInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_leg_info = t
      self.humanoid_leg_info_recv = True
      print('self.humanoid_leg_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      self.TryToQueueObservation()

  def OnHumanoidAnkleInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_ankle_info = t
      self.humanoid_ankle_info_recv = True
      print('self.humanoid_ankle_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      self.TryToQueueObservation()

  def CatHumanoidInfo(self):
    self.count += 1
    # 硬赋值
    # ------------------ pos -----------------
    # hip
    self.motor_pos[0] = self.humanoid_hip_info.joint_state.pos[0]
    self.motor_pos[1] = self.humanoid_hip_info.joint_state.pos[1]
    self.motor_pos[6] = self.humanoid_hip_info.joint_state.pos[2]
    self.motor_pos[7] = self.humanoid_hip_info.joint_state.pos[3]  
    # leg pitch
    self.motor_pos[2] = self.humanoid_leg_info.joint_state.pos[0]
    self.motor_pos[3] = self.humanoid_leg_info.joint_state.pos[1]
    self.motor_pos[8] = self.humanoid_leg_info.joint_state.pos[2]
    self.motor_pos[9] = self.humanoid_leg_info.joint_state.pos[3]      
    # ankle
    self.motor_pos[4] = self.humanoid_ankle_info.joint_state.pos[0]
    self.motor_pos[5] = self.humanoid_ankle_info.joint_state.pos[1]
    self.motor_pos[10] = self.humanoid_ankle_info.joint_state.pos[2]
    self.motor_pos[11] = self.humanoid_ankle_info.joint_state.pos[3]  

    # ------------------ vel -----------------
    # hip
    self.motor_vel[0] = self.humanoid_hip_info.joint_state.speed[0]
    self.motor_vel[1] = self.humanoid_hip_info.joint_state.speed[1]
    self.motor_vel[6] = self.humanoid_hip_info.joint_state.speed[2]
    self.motor_vel[7] = self.humanoid_hip_info.joint_state.speed[3]  
    # leg pitch
    self.motor_vel[2] = self.humanoid_leg_info.joint_state.speed[0]
    self.motor_vel[3] = self.humanoid_leg_info.joint_state.speed[1]
    self.motor_vel[8] = self.humanoid_leg_info.joint_state.speed[2]
    self.motor_vel[9] = self.humanoid_leg_info.joint_state.speed[3]      
    # ankle
    self.motor_vel[4] = self.humanoid_ankle_info.joint_state.speed[0]
    self.motor_vel[5] = self.humanoid_ankle_info.joint_state.speed[1]
    self.motor_vel[10] = self.humanoid_ankle_info.joint_state.speed[2]
    self.motor_vel[11] = self.humanoid_ankle_info.joint_state.speed[3]  
 
                          

  def TryToQueueObservation(self):
    "整理组合observation信息"
    
    if self.humanoid_hip_info_recv or self.humanoid_leg_info_recv or self.humanoid_ankle_info_recv:
      self.CatHumanoidInfo()
      self.motor_pos_list.append(self.motor_pos.tolist())
      self.motor_vel_list.append(self.motor_vel.tolist())
      # Reset flags
      self.imu_recv = False
      self.humanoid_info_recv = False
      self.joycontrol_recv = False
            
if __name__ == "__main__":
  obs = ObservationNode()
  stop_num = 500
  if obs.count > stop_num:
    with open('./data/dof_pos_list_real.json','w') as file:
        json.dump(obs.motor_pos_list, file)
    with open('./data/dof_vel_list_mujoco.json','w') as file:
        json.dump(obs.motor_vel_list, file)

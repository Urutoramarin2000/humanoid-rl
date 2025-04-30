#!/usr/bin/python3
#-*-coding: UTF-8 -*-
import imu_pb2,joystick_pb2,robot_pb2,chassis_pb2
import crpilot
import pycrmw
import time
import sys,os
import signal
import logging
'''
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/var/log/observation.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s: %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
'''
pycrmw.Init(sys.argv)

class ObservationNode:
  def __init__(self):
    self.node = pycrmw.Node("RL_Observation_node")
    # topic writer
    self.obs_writer = self.node.CreateWriter("/RL/observation",robot_pb2.RLObservation)
    self.obs = robot_pb2.RLObservation()
    self.sequence = 0
    self.timestamp = int(time.time()*1e9)

    # topic reader
    self.imu_reader = self.node.CreateReader("/imu",self.OnImu)
    self.humanoid_info_reader = self.node.CreateReader("/debug/humanoid_info/leg",self.OnHumanoidInfo)
    self.joystick_reader = self.node.CreateReader("/ctrl_cmd/joystick",self.OnJoycontrol)

    self.running_flag = True
    # get topic msg flag
    self.imu_recv = False
    self.humanoid_info_recv = False
    self.joycontrol_recv = False

    self.imu = 0
    self.joycontrol = 0
    self.humanoid_info = 0

    # input cmd
    self.speed = 0
    self.theta = 0

  def OnImu(self,t:imu_pb2.Imu):
    self.imu = t
    self.imu_recv = True
    
  def OnHumanoidInfo(self,t:robot_pb2.HumanoidInfo):
    self.humanoid_info = t
    self.left_leg = t.left_leg
    self.right_leg = t.right_leg
    self.humanoid_info_recv = True
  
  def OnJoycontrol(self,t:chassis_pb2.VehicleCommond):
    self.joycontrol = t
    self.speed = t.drive.speed
    self.theta = t.drive.steer 
    self.joycontrol_recv = True

  def signal_handler(self, signal):
    print('signal_number:', signal)
    self.running_flag = False

  def PublishObs(self):
    if not self.humanoid_info_recv or not self.joycontrol_recv or not self.imu_recv:
      logger.error("No main Message")
      return
    

  def Update(self):
    self.joycontrol_recv = False
    self.imu_recv = False
    self.humanoid_info_recv = False
    signal.signal(signal.SIGINT, self.signal_handler)
    print(self.imu)
    print(self.joycontrol)
    print(self.humanoid_info)
    time.sleep(3)
    #TODO:组帧及发布话题
    # self.obs.timestamp = int(time.time()*1e9)
    # self.obs.sequence += 1
    # # input cmd init
    # self.obs.cmd_state.vx = self.speed
    # self.obs.cmd_state.angle_yaw = self.theta
    # joint state init

    # action state init

    # imu state init

if __name__ == "__main__":
  obs = ObservationNode()
  if (not pycrmw.IsOK()):
    os._exit(0)
  while obs.running_flag:
    obs.Update()
    

#!/usr/bin/python3
#-*-coding: UTF-8 -*-
import joystick_pb2,robot_pb2,chassis_pb2,robot_pb2,pose_pb2,imu_pb2
import crpilot
import pycrmw
import time
import sys,os
import signal
import logging
from collections import deque

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
    self.node = pycrmw.Node("RL_Observation_node")
    # topic writer
    self.obs_writer = self.node.CreateWriter("/RL/observation",robot_pb2.RLObservation)
    self.obs = robot_pb2.RLObservation()
    self.sequence = 0
    self.timestamp = int(time.time()*1e9)

    # topic reader
    self.imu_reader = self.node.CreateReader("/imu",self.OnImu)
    self.humanoid_info_reader = self.node.CreateReader("/humanoid_info/leg",self.OnHumanoidInfo)
    self.joystick_reader = self.node.CreateReader("/ctrl_cmd/joystick",self.OnJoycontrol)

    self.running_flag = True
    # get topic msg flag
    self.imu_recv = False
    self.humanoid_info_recv = False
    self.joycontrol_recv = False

    # Initialize an empty deque with a maximum length of 3
    self.observation_queue = deque(maxlen=3)
    #new adds
    self.last_complete_observation = None
    self.imu = None
    self.humanoid_info = None


    # input cmd
    self.speed = 0
    self.theta = 0

  def OnImu(self,t:imu_pb2.Imu):
    if not self.CheckForImuData(t):
        self.imu = t
        self.imu_recv = True
        #self.TryToQueueObservation()


    
  def OnHumanoidInfo(self,t:robot_pb2.HumanoidInfo):
    if not self.CheckForHumanoidInfoData(t):
        self.humanoid_info = t
        self.humanoid_info_recv = True
        #self.TryToQueueObservation()

  def CheckForImuData(self, t: imu_pb2.Imu):
    if t is None:
      return False
    for i in range(3):
      if abs(t.acceleration.x) > 0 or not isinstance(t.acceleration.x):
      #if abs(t.acceleration[i]) > 1 or not isinstance(t.acceleration[i], (float, int)) :
        return False
      #if abs(t.angular_velocity[i]) > 1 or not isinstance(t.angular_velocity[i], (float, int) ):
      #  return False
    return True
  
  def CheckForHumanoidInfoData(self, t: robot_pb2.HumanoidInfo):
    if t is None:
      return False
    for i in range(12):
      if abs(t.joint_state.pos[i]) > 1 or isinstance(t.joint_state.pos[i], (float, int)):
        return False
    return True

  def CheckForJoycontrolData(self, t: chassis_pb2.VehicleCommond):
    if t is None:
      return False
    if abs(t.drive.speed) > 1:
      return False
    if abs(t.drive.steer) > 1:
      return False
    return True

  def OnJoycontrol(self,t:chassis_pb2.VehicleCommond):
    if self.CheckForJoycontrolData(t):
      self.joycontrol = t
      self.joycontrol_recv = True
      self.TryToQueueObservation()

  def TryToQueueObservation(self):
      # Use the last complete observation if current data is incomplete
    imu_data = self.imu if self.imu is not None else (self.last_complete_observation['imu'] if self.last_complete_observation else None)
    humanoid_info_data = self.humanoid_info if self.humanoid_info is not None else (self.last_complete_observation['humanoid_info'] if self.last_complete_observation else None)
    joycontrol_data = self.joycontrol if self.joycontrol is not None else (self.last_complete_observation['joycontrol'] if self.last_complete_observation else None)
    if imu_data and humanoid_info_data and joycontrol_data:

          # Combine imu and humanoid_info into a single observation
      observation = {
        'imu': imu_data,
        'humanoid_info': humanoid_info_data,
        'joycontrol': joycontrol_data,
        'timestamp': int(time.time() * 1e9)
      }

        # Append the observation to the queue
      self.observation_queue.append(observation)

        # Store the current complete observation as the last complete one
      self.last_complete_observation = observation

        # Reset flags
      self.imu_recv = False
      self.humanoid_info_recv = False

  def GetObservation(self):   
        # Retrieve and remove the oldest observation from the queue, if available
      if self.observation_queue:
        return self.observation_queue.popleft()
      else:
        print("Senors Data Queue is empty.")
        return None

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
    #TODO:组帧及发布话题
    print(self.imu)
    self.obs.timestamp = int(time.time()*1e9)
    self.obs.sequence += 1
    # input cmd init
    self.obs.cmd_state.vx = self.speed
    self.obs.cmd_state.angle_yaw = self.theta
    # joint state init

    # action state init

    # imu state init

class RLPublish:
  def __init__(self) -> None:
    self.node = pycrmw.Node("RL_Action_node")
    #topic writer
    self.action_writer = self.node.CreateWriter("/RL/action",robot_pb2.RLAction)
    self.rlaction = robot_pb2.RLAction()
    """
    action => joint_num, control_mode, cmd([0,0,0..,0]), sequence, timestamp, time_step, running_flag
    """
    self.sequence = 0
    #for i in range(12):
      #这个是个list？
      #self.action.cmd.append(0)
    self.time_step = 0.01 # 100HZ
    self.running_flag = True # 程序退出flag
  
  def signal_handler(self, signal):
    print('signal_number:', signal)
    self.running_flag = False

  def PublishAction(self, data):
    #可改发布话题的条件
    signal.signal(signal.SIGINT, self.signal_handler)
    #while send_flag:
    rlaction = robot_pb2.RLAction()
    rlaction.timestamp = int(time.time()*1e9)
    rlaction.joint_num = len(data) # 12个关节电机
    rlaction.control_mode = robot_pb2.INTERPOLATE_MODE
    self.sequence += 1
    rlaction.sequence = self.sequence
    for i in range(len(data)):
        rlaction.action.cmd.append(data[i]) # 12维数据 赋值，此处省略
    self.action_writer.Write(rlaction)
    #last_timestamp = self.action.timestamp
    
    time.sleep(self.time_step)

if __name__ == "__main__":
  obs = ObservationNode()
  act = RLPublish()
  if (not pycrmw.IsOK()):
    os._exit(0)
  while act.running_flag:
    #obs.Update()
    act.PublishAction([1,2,3,4,5,6])
    


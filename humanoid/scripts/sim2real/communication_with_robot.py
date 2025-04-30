#!/usr/bin/python3
#-*-coding: UTF-8 -*-
import robot_pb2,chassis_pb2,pose_pb2,imu_pb2
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
    self.joycontrol = None
    # topic reader
#    self.imu_reader = self.node.CreateReader("/imu",self.OnImu)
#    self.humanoid_info_reader = self.node.CreateReader("/humanoid_info/leg",self.OnHumanoidInfo)
#    self.joystick_reader = self.node.CreateReader("/ctrl_cmd/joystick",self.OnJoycontrol)
    #print('1')
    self.running_flag = True
    # get topic msg flag
    self.imu_recv = False
    self.humanoid_info_recv = False
    self.joycontrol_recv = False
    self.humanoid_leg_info_recv = False
    self.humanoid_ankle_info_recv = False
    #print('2')
    # Initialize an empty deque with a maximum length of 3
    self.observation_queue = deque(maxlen=3)
    #new adds
    self.last_complete_observation = None

    # input cmd
    self.speed = 0
    self.theta = 0

    self.imu = None
    self.joycontrol = None
    self.humanoid_info = None
    self.humanoid_leg_info = None
    self.humanoid_ankle_info = None
    #print('3')
    self.imu_last = None
    self.joycontrol_last = None
    self.humanoid_info_last = None
    self.observation = {
        'imu': self.imu,
        'humanoid_info': self.humanoid_info,
        'joycontrol': self.joycontrol,
        'timestamp': int(time.time() * 1e9)
    }
    #print('obsevation init', self.observation)
    self.imu_reader = self.node.CreateReader("/imu/ahrs",self.OnImu)
    # self.humanoid_info_reader_hip = self.node.CreateReader("/humanoid_info/hip",self.OnHumanoidHipInfo)
    self.humanoid_info_reader_leg = self.node.CreateReader("/RL/base_info/pitch",self.OnHumanoidLegInfo)
    self.humanoid_info_reader_ankle = self.node.CreateReader("/RL/base_info/ankle",self.OnHumanoidAnkleInfo)
    
    self.joystick_reader = self.node.CreateReader("/ctrl_cmd/joystick",self.OnJoycontrol)
    



  def OnImu(self,t:imu_pb2.Imu):
    self.imu = t
    self.imu_recv = True
    if self.imu != None and self.humanoid_leg_info != None and self.humanoid_ankle_info != None:
      self.TryToQueueObservation()
    #if not self.CheckForImuData(t):
    #    self.imu = t
    #    #print('self.imu.timestamp',t.timestamp)
    #    self.imu_recv = True
    #    #print('IMU data got............')
    #    self.TryToQueueObservation()

  def OnHumanoidHipInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_hip_info = t
      self.humanoid_hip_info_recv = True
      #print('self.humanoid_hip_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      if self.imu != None and self.humanoid_leg_info != None and self.humanoid_ankle_info != None:
        self.TryToQueueObservation()

  def OnHumanoidLegInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_leg_info = t
      self.humanoid_leg_info_recv = True
      #print('self.humanoid_leg_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      if self.imu != None and self.humanoid_leg_info != None and self.humanoid_ankle_info != None:
        self.TryToQueueObservation()

  def OnHumanoidAnkleInfo(self,t:robot_pb2.HumanoidRLInfo):
    #if not self.CheckForHumanoidInfoData(t):
      self.humanoid_ankle_info = t
      self.humanoid_ankle_info_recv = True
      #print('self.humanoid_ankle_info.timestamp',t.timestamp)
      #print("-------------------------------")
      #print("humanoid_info got .........")
      self.TryToQueueObservation()

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
    #if self.CheckForJoycontrolData(t):
    self.joycontrol = t
    self.joycontrol_recv = True
    #print("joycontrol: ", self.joycontrol)
    #print('joycontrol info got.....................')
    self.TryToQueueObservation()
  
  def MergeHumanoidInfo(self):
    obs_base_info = robot_pb2.HumanoidRLInfo()
    obs_base_info.CopyFrom(self.humanoid_leg_info)
    obs_base_info.joint_state.pos.insert(1,self.humanoid_ankle_info.joint_state.pos[0]) # 14 13 18 17 16 22 21
    obs_base_info.joint_state.pos.insert(4,self.humanoid_ankle_info.joint_state.pos[2]) # 14 13 18 17 19 16 22 21
    obs_base_info.joint_state.pos.insert(5,self.humanoid_ankle_info.joint_state.pos[3]) # 14 13 18 17 19 20 16 22 21
    obs_base_info.joint_state.pos.insert(7,self.humanoid_ankle_info.joint_state.pos[1]) # 14 13 18 17 19 20 16 15 22 21
    obs_base_info.joint_state.pos.append(self.humanoid_ankle_info.joint_state.pos[4]) # 14 13 18 17 19 20 16 15 22 21 23
    obs_base_info.joint_state.pos.append(self.humanoid_ankle_info.joint_state.pos[5]) # 14 13 18 17 19 20 16 15 22 21 23 24
    
    obs_base_info.joint_state.speed.insert(1,self.humanoid_ankle_info.joint_state.speed[0]) # 14 13 18 17 16 22 21
    obs_base_info.joint_state.speed.insert(4,self.humanoid_ankle_info.joint_state.speed[2]) # 14 13 18 17 19 16 22 21
    obs_base_info.joint_state.speed.insert(5,self.humanoid_ankle_info.joint_state.speed[3]) # 14 13 18 17 19 20 16 22 21
    obs_base_info.joint_state.speed.insert(7,self.humanoid_ankle_info.joint_state.speed[1]) # 14 13 18 17 19 20 16 15 22 21
    obs_base_info.joint_state.speed.append(self.humanoid_ankle_info.joint_state.speed[4]) # 14 13 18 17 19 20 16 15 22 21 23
    obs_base_info.joint_state.speed.append(self.humanoid_ankle_info.joint_state.speed[5]) # 14 13 18 17 19 20 16 15 22 21 23 24
    return obs_base_info

  def CatHumanoidInfo(self):
    # 定义腿部关节名称
    hip_joints = ["hip_roll", "hip_yaw"]
    hip_indices = {
        "left_leg": [0, 1],
        "right_leg": [2, 3]
    }
    leg_joints = ["thigh_back", "thigh_front"]
    leg_indices = {
        "left_leg": [0, 1],
        "right_leg": [2, 3]
    }
    ankle_joints = ["calve_left", "calve_right"]
    ankle_indices = {
        "left_leg": [0, 1],
        "right_leg": [2, 3]
    }
      # 更新关节位置和速度
    def update_joint_info(joint_info, humanoid_info, joint_names, indices):
        for idx, joint_name in enumerate(joint_names):
            joint = getattr(humanoid_info, joint_name)
            joint.pos = joint_info.joint_state.pos[indices[idx]]
            joint.vel = joint_info.joint_state.vel[indices[idx]]

    # 更新左腿和右腿的髋关节、腿部和踝关节信息
    update_joint_info(self.humanoid_hip_info, self.humanoid_info.left_leg, hip_joints, hip_indices["left_leg"])
    update_joint_info(self.humanoid_hip_info, self.humanoid_info.right_leg, hip_joints, hip_indices["right_leg"])
    update_joint_info(self.humanoid_leg_info, self.humanoid_info.left_leg, leg_joints, leg_indices["left_leg"])
    update_joint_info(self.humanoid_leg_info, self.humanoid_info.right_leg, leg_joints, leg_indices["right_leg"])
    update_joint_info(self.humanoid_ankle_info, self.humanoid_info.left_leg, ankle_joints, ankle_indices["left_leg"])
    update_joint_info(self.humanoid_ankle_info, self.humanoid_info.right_leg, ankle_joints, ankle_indices["right_leg"])

    # 硬赋值
    # # ------------------ pos -----------------
    # # hip
    # self.humanoid_info.left_leg.hip_roll.pos = self.humanoid_hip_info.joint_state.pos[0]
    # self.humanoid_info.left_leg.hip_yaw.pos = self.humanoid_hip_info.joint_state.pos[1]
    # self.humanoid_info.right_leg.hip_roll.pos = self.humanoid_hip_info.joint_state.pos[2]
    # self.humanoid_info.right_leg.hip_yaw.pos = self.humanoid_hip_info.joint_state.pos[3]  
    # # leg pitch
    # self.humanoid_info.left_leg.thigh_back.pos = self.humanoid_leg_info.joint_state.pos[0]
    # self.humanoid_info.left_leg.thigh_front.pos = self.humanoid_leg_info.joint_state.pos[1]
    # self.humanoid_info.right_leg.thigh_back.pos = self.humanoid_leg_info.joint_state.pos[2]
    # self.humanoid_info.right_leg.thigh_front.pos = self.humanoid_leg_info.joint_state.pos[3]      
    # # ankle
    # self.humanoid_info.left_leg.calve_left.pos = self.humanoid_ankle_info.joint_state.pos[0]
    # self.humanoid_info.left_leg.calve_right.pos = self.humanoid_ankle_info.joint_state.pos[1]
    # self.humanoid_info.right_leg.calve_left.pos = self.humanoid_ankle_info.joint_state.pos[2]
    # self.humanoid_info.right_leg.calve_right.pos = self.humanoid_ankle_info.joint_state.pos[3]  

    # # ------------------ vel -----------------
    # # hip
    # self.humanoid_info.left_leg.hip_roll.vel = self.humanoid_hip_info.joint_state.vel[0]
    # self.humanoid_info.left_leg.hip_yaw.vel = self.humanoid_hip_info.joint_state.vel[1]
    # self.humanoid_info.right_leg.hip_roll.vel = self.humanoid_hip_info.joint_state.vel[2]
    # self.humanoid_info.right_leg.hip_yaw.vel = self.humanoid_hip_info.joint_state.vel[3]  
    # # leg pitch
    # self.humanoid_info.left_leg.thigh_back.vel = self.humanoid_leg_info.joint_state.vel[0]
    # self.humanoid_info.left_leg.thigh_front.vel = self.humanoid_leg_info.joint_state.vel[1]
    # self.humanoid_info.right_leg.thigh_back.vel = self.humanoid_leg_info.joint_state.vel[2]
    # self.humanoid_info.right_leg.thigh_front.vel = self.humanoid_leg_info.joint_state.vel[3]      
    # # ankle
    # self.humanoid_info.left_leg.calve_left.vel = self.humanoid_ankle_info.joint_state.vel[0]
    # self.humanoid_info.left_leg.calve_right.vel = self.humanoid_ankle_info.joint_state.vel[1]
    # self.humanoid_info.right_leg.calve_left.vel = self.humanoid_ankle_info.joint_state.vel[2]
    # self.humanoid_info.right_leg.calve_right.vel = self.humanoid_ankle_info.joint_state.vel[3]       

  def TryToQueueObservation(self):
    " 作为最后一个topic,整理组合observation信息,但并没有整合成RLObservation"
      # Use the last complete observation if current data is incomplete
    # imu_data = self.imu if self.imu is not None else (self.last_complete_observation['imu'] if self.last_complete_observation else None)
    # humanoid_info_data = self.humanoid_info if self.humanoid_info is not None else (self.last_complete_observation['humanoid_info'] if self.last_complete_observation else None)
    # joycontrol_data = self.joycontrol if self.joycontrol is not None else (self.last_complete_observation['joycontrol'] if self.last_complete_observation else None)
  
        # q.append(humanoid_data.left_leg.hip_roll.pos)
        # q.append(humanoid_data.left_leg.hip_yaw.pos)
        # q.append(humanoid_data.left_leg.thigh_back.pos)
        # q.append(humanoid_data.left_leg.thigh_front.pos)
        # q.append(humanoid_data.left_leg.calve_left.pos)
        # q.append(humanoid_data.left_leg.calve_right.pos)
    #   实际如果丢包，也能一直拿到最后一帧的信息
    #imu_data = self.imu
    #humanoid_info_data = self.humanoid_info
    #joycontrol_data = self.joycontrol
    #print('----------------------------')
    #print('imu_data  got 2 .............')
    #print("imu is true: ", self.imu)
    #print("humanoid_info  is true: ", self.humanoid_info)
    #print("joycontrol  is true: ", self.joycontrol)
    # self.CatHumanoidInfo()
    # if self.imu_recv or self.humanoid_info_recv or self.joycontrol_recv:
    if self.imu == None or self.humanoid_leg_info == None or self.humanoid_ankle_info == None: return
    if self.imu_recv or self.humanoid_leg_info_recv or self.humanoid_ankle_info_recv:
    
      # Combine imu and humanoid_info into a single observation
      self.observation = {
        'imu': self.imu,
        'humanoid_info': self.MergeHumanoidInfo(),
        'joycontrol': self.joycontrol,
        'timestamp': int(time.time() * 1e9)
      }
      #print('observation data2 :', self.observation)

        # Append the observation to the queue
#      self.observation_queue.append(self.observation)

        # Store the current complete observation as the last complete one
      self.last_complete_observation = self.observation

        # Reset flags
      self.imu_recv = False
      # self.humanoid_info_recv = False
      self.humanoid_leg_info_recv = False
      self.humanoid_ankle_info_recv = False
      self.joycontrol_recv = False
      
  def GetObservation(self):   
        # Retrieve and remove the oldest observation from the queue, if available
      #if self.observation_queue:
      #  return self.observation_queue.popleft()
      #else:
      #  print("Senors Data Queue is empty.")
      #  return None
      if self.imu_recv and self.humanoid_info_recv and self.joycontrol_recv:
        return self.observation
      else:
        return self.last_complete_observation

  def signal_handler(self, signal):
    print('signal_number:', signal)
    self.running_flag = False
    self.init_flag = False

  def PublishObs(self):
    " 发布RLObservation消息,目前没啥用,信息直接整合传到sim2real的run_policy里了"
    if not self.humanoid_info_recv or not self.joycontrol_recv or not self.imu_recv:
      logger.error("No main Message")
      return
    # 发布观察数据
    self.obs_writer.publish(self.obs)
    self.get_logger().info("Observation published.")
    self.sequence += 1  # 增加序列号
    
  def Update(self):
    "更新RLObservation消息,调用PublishObs循环发布RLObservation消息,目前没啥用"
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
    for i in range(12):
      self.rlaction.action.cmd.append(0)
      self.rlaction.last_action.cmd.append(0)
    self.time_step = 0.005 # 200HZ
    self.running_flag = True # 程序退出flag
    self.finish_init = False
    self.last_publish_time = 0
    self.max_dt = 0
    signal.signal(signal.SIGINT, self.signal_handler)
  
  def signal_handler(self, signal):
    print('signal_number:', signal)
    self.running_flag = False
  
  def PublishAction(self, data):
    if not self.finish_init: 
        time.sleep(3)
        self.finish_init = True
    # now = int(time.time()*1e6)
    # if self.last_publish_time == 0: 
    #   self.last_publish_time = now
    # dt = now-self.last_publish_time
    # if (dt > self.max_dt):
    #   self.max_dt = dt
      # print("sequence: {} dt: {}".format(self.sequence,self.max_dt))
    # print("sequence: {} dt: {}".format(self.sequence,dt))
    #可改发布话题的条件
    
    #while send_flag:
    #start = int(time.time()*1e6)
    # rlaction = robot_pb2.RLAction()
    
    self.rlaction.timestamp = int(time.time()*1e9)
    self.rlaction.joint_num = len(data) # 12个关节电机
    self.rlaction.control_mode = robot_pb2.TORQUE_MODE
    self.sequence += 1
    self.rlaction.sequence = self.sequence
    
    for i in range(len(data)):
        self.rlaction.action.cmd[i] = data[i]
        # self.rlaction.action.cmd.append(data[i]) # 12维数据 赋值，此处省略
    
    self.action_writer.Write(self.rlaction)
    # if pycrmw.IsOK():
    #     #if dt > 4000: print("sequence: {} dt: {}".format(self.sequence,dt))
    #     #print("sequence: {} dt: {}".format(self.sequence,dt))
    #     start = int(time.time()*1e6)
    #     #print("communication: {}".format(time.time()))
    #     self.action_writer.Write(self.rlaction)
        
    
    # self.last_publish_time = now
    # if (end - start > self.max_dt):
    #   self.max_dt = end - start
      #print("sequence: {} max_dt: {}".format(self.sequence,self.max_dt))
    # print("sequence: {} end - start: {}".format(self.sequence,end - start))
    # if (self.sequence == 41 or self.sequence == 35): print(self.rlaction)
    #last_timestamp = self.action.timestamp
    
    #time.sleep(self.time_step)

if __name__ == "__main__":
  # obs = ObservationNode()
  act = RLPublish()
  if (not pycrmw.IsOK()):
    os._exit(0)
  while act.running_flag:
    #obs.Update()
    act.PublishAction([0,0,0,0,0,0,0,0,0,0,0,0.01])

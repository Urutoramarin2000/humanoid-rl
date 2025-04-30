"该版本中observation信息是从整合好的RLObservation消息信息中拿到的"
import math
import numpy as np
# import mujoco, mujoco_viewer
# from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
# from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import CowaCfg
import torch
import signal
from communication_with_robot import ObservationNode, RLPublish
import time
import threading
import os
import sys

lock = threading.Lock()
last_action_data = [0] * 12
time_step = 0.01

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def run_policy(policy, cfg, observation_data):
    #仿真步长是1000Hz，但是上游数据是100Hz
    #sim_dt = 0.001 
    start_time = time.time()
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    hist_obs = deque()
    for _ in range (cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    #获取消息，todo：get_msg
    #data = get_msg()
    count_lowlevel = 0
    # observation_data = ObservationNode()
    #publish_data = RLPublish()
    last_timestamp = observation_data.obs.timestamp
    # last_timestamp = observation_data.GetObservation()['timestamp']   #   修改前

    last_quaternion = R.from_quat([0.0, 0.0, 0.0, 1.0]) #初始化为0，但实际是否有初始的角度
    last_velocity = np.array([0.0, 0.0, 0.0])

    #这里仿真是1000Hz的，但是上游数据是100Hz的，
    while(1):   
        #这里是100Hz的callback
        imu_data = observation_data.obs.imu_state

        #humanoid_data.joint_state.pos[12]
        #humanoid_data.joint_state.speed[12]
        humanoid_data = observation_data.obs.joint_state

        #joycontrol_data.drive.speed, joycontrol_data.drive.steer (相当于theta，没有vy)
        joycontrol_data = observation_data.obs.cmd_state
        timestamp = observation_data.obs.timestamp
        #需要获取q, dq, quat, v, omega, gvec 

        #计算q和dq  
        #有问题,想要读取的是关节pos和speed,pos应该读取humanoid_data.left_leg[0:6].pos以及humanoid_data.right_leg[0:6].pos
        #speed应该读取humanoid_data.left_leg[0:6].speed以及humanoid_data.right_leg[0:6].speed
        q, dq= [], []
        for i in range(12):
            q.append(observation_data.obs.joint_state.pos[i])
            dq.append(observation_data.obs.joint_state.speed[i])
        #   问题：   q在这里指的不只是关节pos
        #           q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        #           dq = self.dof_vel * self.obs_scales.dof_vel
        q.append(humanoid_data.left_leg.hip_roll.pos)
        q.append(humanoid_data.left_leg.hip_yaw.pos)
        q.append(humanoid_data.left_leg.thigh_back.pos)
        q.append(humanoid_data.left_leg.thigh_front.pos)
        q.append(humanoid_data.left_leg.calve_left.pos)
        q.append(humanoid_data.left_leg.calve_right.pos)
        q.append(humanoid_data.right_leg.hip_roll.pos)
        q.append(humanoid_data.right_leg.hip_yaw.pos)
        q.append(humanoid_data.right_leg.thigh_back.pos)
        q.append(humanoid_data.right_leg.thigh_front.pos)
        q.append(humanoid_data.right_leg.calve_left.pos)
        q.append(humanoid_data.right_leg.calve_right.pos)
        q = np.array(q).astype(np.double)
        dq.append(humanoid_data.left_leg.hip_roll.speed)
        dq.append(humanoid_data.left_leg.hip_yaw.speed)
        dq.append(humanoid_data.left_leg.thigh_back.speed)
        dq.append(humanoid_data.left_leg.thigh_front.speed)
        dq.append(humanoid_data.left_leg.calve_left.speed)
        dq.append(humanoid_data.left_leg.calve_right.speed)
        dq.append(humanoid_data.right_leg.hip_roll.speed)
        dq.append(humanoid_data.right_leg.hip_yaw.speed)
        dq.append(humanoid_data.right_leg.thigh_back.speed)
        dq.append(humanoid_data.right_leg.thigh_front.speed)
        dq.append(humanoid_data.right_leg.calve_left.speed)
        dq.append(humanoid_data.right_leg.calve_right.speed)
        dq = np.array(dq).astype(np.double)
        dt = (timestamp - last_timestamp) / 1e9
        last_timestamp = timestamp
        
        #计算四元素
        angular_velocity = np.array([imu_data.angular.x, imu_data.angular.y, imu_data.angular.z])  # 单位: rad/s
        rotation_vector = angular_velocity * dt
        rotation = R.from_rotvec(rotation_vector)
        current_quaternion = rotation * last_quaternion
        last_quaternion = current_quaternion
        current_quaternion = current_quaternion.as_quat()
        current_quaternion = current_quaternion.astype(np.double)
        #拿rotation
        r = R.from_quat(current_quaternion)
        #计算重力
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

        #计算速度v，局部坐标系,imu的速度误差大的    #   暂时不需要求速度v，之后如果加入了对速度v的估计网络，可以再在actor的obs加入速度v
        acceration = np.array([imu_data.acceleration.x, imu_data.acceleration.y, imu_data.acceleration.z])
        current_velocity = last_velocity + acceration * dt
        current_velocity = current_velocity.astype(np.double)
        last_velocity = current_velocity    

        #计算角速度 omega 
        omega = np.array([imu_data.angular.x, imu_data.angular.y, imu_data.angular.z]).astype(np.double)
        
        #需要根据输入配置更改每帧的观察值的维度
        obs =  np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
        eu_ang = quaternion_to_euler_array(current_quaternion)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        #将数据放入obs
        obs[0, 0] = math.sin(2 * math.pi * count_lowlevel *  dt  / 0.5)  # 问题：0.64是cycle time，网路训练时，参数为0.5
        obs[0, 1] = math.cos(2 * math.pi * count_lowlevel *  dt  / 0.5)
        obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
        obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 29:41] = action
        obs[0, 41:44] = omega
        obs[0, 44:47] = eu_ang
        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
        for i in range(cfg.env.frame_stack):
            policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
        #inference delay
        try:
            start_inference_time = time.time()
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            end_inference_time = time.time()
            print("Inference delay: ", end_inference_time - start_inference_time)
            #实机返回的是action，不是target
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            count_lowlevel += 1

            #这里差发送的代码，发送action   (发送action采用另一个线程,send_action里实现)
            global action_cmd
            action_cmd = action.tolist()
            with lock:
                last_action_data = action_cmd
        except Exception as e:
            print("Error in computing policy:", e)
        elapsed_time = time.time() - start_time
        time.sleep(max(0, time_step - elapsed_time))    # 控制每次循环为time_step = 0.01s

        #publish_data.PublishAction(action_cmd)

def signal_handler(signum, frame):
    print("Interrupt received, shutting down...")
    # 执行清理操作
    publish_data.running_flag = False
    

def send_action(publish_data):    
    while True:
        with lock:
            current_action_data = action_cmd
        publish_data.PublishAction(current_action_data)
        #time.sleep(time_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    args = parser.parse_args()

    if not os.path.isfile(args.load_model):
        print(f"Error: Model file {args.load_model} does not exist.")
        sys.exit(1)

    class Sim2RealCfg(CowaCfg):

        class sim_config:
            #sim_duration = 60.0
            #这里改成100Hz
            dt = 0.01
            #decimation = 10
        """
        class robot_config:
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
        """
    observation_data = ObservationNode()
    publish_data = RLPublish()

    try:
        #python scripts/sim2sim.py --load_model /path/to/export/model.pt
        policy = torch.jit.load(args.load_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    run_policy_thread = threading.Thread(target=run_policy, args=(policy, Sim2RealCfg(), observation_data))
    publish_thread = threading.Thread(target=send_action, args=(publish_data,))

    run_policy_thread.start()
    publish_thread.start()
    
    try:
        run_policy_thread.join()
        publish_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
        publish_data.running_flag = False
        run_policy_thread.join()
        publish_thread.join()

import math
import numpy as np
# import mujoco, mujoco_viewer
# from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
# from humanoid import LEGGED_GYM_ROOT_DIR
from cowa_config import CowaCfg
import torch
import signal
from communication_with_robot import ObservationNode, RLPublish
import time
import threading
import os
import sys
from collections import deque

lock = threading.Lock()
last_action_data = [0] * 12
time_step = 0.01

action_cmd_queue = deque(maxlen=15)
action_cmd_queue.append([0] * 12)

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def QuaternionMultiply(q1, q2):
    """
    Multiply two quaternions.

    :param q1: First quaternion [w, x, y, z].
    :param q2: Second quaternion [w, x, y, z].
    :return: Resulting quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z
  
def Quaternion2Euler(w,x,y,z):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    :param w: Real part of the quaternion.
    :param x: i-component of the quaternion.
    :param y: j-component of the quaternion.
    :param z: k-component of the quaternion.
    :return: Tuple (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll_x = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch_y = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch_y = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw_z = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll_x,pitch_y,yaw_z])
    #return np.array([0,0,0])

def write_imu2file(timestamp,r,p,y,gap):
    with open('imu_data.txt','a') as f:
        f.write(str(timestamp) + ' ' + str(r) + ' ' + str(p) + ' ' + str(y) + ' ' + str(gap) + '\n')

def run_policy(policy, cfg, observation_data):
    #仿真步长是1000Hz，但是上游数据是100Hz
    #sim_dt = 0.001 
    ratio = 0.9
    publish_data.PublishAction([0]*12)
    start_time = time.time()
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    # last_actions = np.zeros(cfg.env.num_actions, dtype=np.double)
    hist_obs = deque()
    for _ in range (cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    global action_cmd
    action_cmd = [0] * 12
    action_scale = [0] * 12
    #获取消息，todo：get_msg
    #data = get_msg()
    # observation_data = ObservationNode()
    #publish_data = RLPublish()
    #last_timestamp = observation_data.GetObservation()['timestamp']
    
    last_quaternion = R.from_quat([0.0, 0.0, 0.0, 1.0]) #初始化为0，但实际是否有初始的角度
    last_velocity = np.array([0.0, 0.0, 0.0])

    #这里仿真是1000Hz的，但是上游数据是100Hz的
    print_count = 0
    while(1):   
        proc_start_time = time.time()
        #imu_data, humanoid_data, joycontorl_data = observation_data.GetObservation()
        #imu_data.angular.x, imu_data.angular.y, imu_data.angular.z
        #imu_data.acceleration.x, imu_data.acceleration.y, imu_data.acceleration.z
        #这里是100Hz的callback
        imu_data = observation_data.GetObservation()['imu']

        #humanoid_data.joint_state.pos[12]
        #humanoid_data.joint_state.speed[12]
        humanoid_data = observation_data.GetObservation()['humanoid_info']

        #joycontrol_data.drive.speed, joycontrol_data.drive.steer (相当于theta，没有vy)
        joycontrol_data = observation_data.GetObservation()['joycontrol']
        #timestamp = observation_data.GetObservation()['timestamp']
        #需要获取q, dq, quat, v, omega, gvec 

        #计算q和dq  
        #有问题,想要读取的是关节pos和speed,pos应该读取humanoid_data.left_leg[0:6].pos以及humanoid_data.right_leg[0:6].pos
        #speed应该读取humanoid_data.left_leg[0:6].speed以及humanoid_data.right_leg[0:6].speed
        q = np.array(humanoid_data.joint_state.pos).astype(np.double)
        dq = np.array(humanoid_data.joint_state.speed).astype(np.double)
        # q, dq= [], []
        # for i in range(12):
        #     q.append(humanoid_data.joint_state.pos[i])
        #     dq.append(humanoid_data.joint_state.speed[i])
        #   问题：   q在这里指的不只是关节pos
        #           q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        #           dq = self.dof_vel * self.obs_scales.dof_vel
        # q.append(humanoid_data.left_leg.hip_roll.pos)
        # q.append(humanoid_data.left_leg.hip_yaw.pos)
        # q.append(humanoid_data.left_leg.thigh_back.pos)
        # q.append(humanoid_data.left_leg.thigh_front.pos)
        # q.append(humanoid_data.left_leg.calve_left.pos)
        # q.append(humanoid_data.left_leg.calve_right.pos)
        # q.append(humanoid_data.right_leg.hip_roll.pos)
        # q.append(humanoid_data.right_leg.hip_yaw.pos)
        # q.append(humanoid_data.right_leg.thigh_back.pos)
        # q.append(humanoid_data.right_leg.thigh_front.pos)
        # q.append(humanoid_data.right_leg.calve_left.pos)
        # q.append(humanoid_data.right_leg.calve_right.pos)
        # q = np.array(q).astype(np.double)
        # dq.append(humanoid_data.left_leg.hip_roll.speed)
        # dq.append(humanoid_data.left_leg.hip_yaw.speed)
        # dq.append(humanoid_data.left_leg.thigh_back.speed)
        # dq.append(humanoid_data.left_leg.thigh_front.speed)
        # dq.append(humanoid_data.left_leg.calve_left.speed)
        # dq.append(humanoid_data.left_leg.calve_right.speed)
        # dq.append(humanoid_data.right_leg.hip_roll.speed)
        # dq.append(humanoid_data.right_leg.hip_yaw.speed)
        # dq.append(humanoid_data.right_leg.thigh_back.speed)
        # dq.append(humanoid_data.right_leg.thigh_front.speed)
        # dq.append(humanoid_data.right_leg.calve_left.speed)
        # dq.append(humanoid_data.right_leg.calve_right.speed)
        # dq = np.array(dq).astype(np.double)
        #dt = (timestamp - last_timestamp) / 1e9
        #last_timestamp = timestamp
        
        # q_offset = [math.sqrt(2)/2, 0, 0, -math.sqrt(2)/2]
        # w_new,x_new,y_new,z_new = QuaternionMultiply(q_offset,[imu_data.transform[3],imu_data.transform[4],imu_data.transform[5],imu_data.transform[6]])
        eu_ang = Quaternion2Euler(imu_data.transform[3],imu_data.transform[4],imu_data.transform[5],imu_data.transform[6])
        eu_ang[2] = eu_ang[2] - 1.57
        # print("r:{} p:{} y:{}".format(eu_ang[0],eu_ang[1],eu_ang[2]))
        # #计算四元素
        #angular_velocity = np.array([0.0,0.0,0.0])
        angular_velocity = np.array([imu_data.angular.x, imu_data.angular.y, imu_data.angular.z]).astype(np.double)  # 单位: rad/s

        # # --------- 去除angular_velocity的静态偏差 ---------------
        # angular_velocity_bias = {-0.0001821213132515822, -3.8718704392068665e-05, -2.9278032642150688e-05}
        # angular_velocity_var = {3.583585088085489e-08, 2.951815440870468e-08, 1.4126412613859379e-08}
        # for axis in range(3):
        #     angular_velocity[axis] -= angular_velocity_bias[axis]
        #     angular_velocity[axis] -= np.random.normal(0, np.sqrt(angular_velocity_var[axis]), angular_velocity[axis].shape)
        
        # rotation_vector = angular_velocity * dt
        # rotation = R.from_rotvec(rotation_vector)
        # current_quaternion = rotation * last_quaternion
        # last_quaternion = current_quaternion
        # current_quaternion = current_quaternion.as_quat()
        # current_quaternion = current_quaternion.astype(np.double)
        # #拿rotation
        # r = R.from_quat(current_quaternion)
        # #计算重力
        # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        # # 计算得到rotation angular
        # eu_ang = quaternion_to_euler_array(current_quaternion)
        # eu_ang[eu_ang > math.pi] -= 2 * math.pi

        #计算速度v，局部坐标系,imu的速度误差大的    #   暂时不需要求速度v，之后如果加入了对速度v的估计网络，可以再在actor的obs加入速度v
        # acceration = np.array([imu_data.acceleration.x, imu_data.acceleration.y, imu_data.acceleration.z])
        # current_velocity = last_velocity + acceration * dt
        # current_velocity = current_velocity.astype(np.double)
        # last_velocity = current_velocity    
        
        #需要根据输入配置更改每帧的观察值的维度
        obs =  np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
        action[4] = np.clip(action[4],-1.0,0.05)
        action[10] = np.clip(action[10],-1.0,0.05)
        #将数据放入obs
        # obs[0, 0] = math.sin(2 * math.pi * count_lowlevel / 100.0  / 1.2)  # 问题：0.64是cycle time，网路训练时，参数为0.5
        # obs[0, 1] = math.cos(2 * math.pi * count_lowlevel / 100.0  / 1.2)
        obs[0, 0] = math.sin(2 * math.pi * (time.time()-start_time) / 1.2)  # 问题：0.64是cycle time，网路训练时，参数为0.5
        obs[0, 1] = math.cos(2 * math.pi * (time.time()-start_time) / 1.2)
        obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
        obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 29:41] = action
        obs[0, 41:44] = angular_velocity
        obs[0, 44:47] = eu_ang
        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
        for i in range(cfg.env.frame_stack):
            policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
        now = int(time.time()*1e9)
        write_imu2file(now,eu_ang[0],eu_ang[1],eu_ang[2],(now-imu_data.timestamp))
        #inference delay
        try:
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action[5] = 0
            action[11] = 0
            action_scale = action[:] * 0.25
            #action[8] = action[6] * 1.25
            #action[9] = action[10] * 1.25
            action_scale[0] = action_scale[0] / 1.5
            action_scale[1] = action_scale[1] / 1.5
            action_scale[7] = action_scale[7] / 1.5
            action_scale[6] = action_scale[6] / 1.5

            action_scale[0] = np.clip(action_scale[0],-1.0,0.05)
            action_scale[6] = np.clip(action_scale[6],-1.0,0.05)

            action_scale[1] = np.clip(action_scale[1],-1.0,0.05)
            action_scale[7] = np.clip(action_scale[7],-1.0,0.05)

            #action_scale[4] = action_scale[4] * 1.5
            #action_scale[10] = action_scale[10] * 1.5
            action_scale[4] = np.clip(action_scale[4],-1.0,0.05)
            action_scale[10] = np.clip(action_scale[10],-1.0,0.05)
            #action[0] = np.clip(action[0],-1.0,0.05)
            #action[6] = np.clip(action[6],-1.0,0.05)
            #action = torch.clamp(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # action_ratio = 0.9 * action + (1 - ratio) * last_actions    # 滤波
            # last_actions = action
            # hist_action.append(action_ratio)

            #这里差发送的代码，发送action   
            action_cmd = action_scale.tolist()
            #action_cmd[4] = 0
            #action_cmd[10] = 0
            #if (len(action_cmd_queue) < 15):
            #    action_cmd_queue.append(action_cmd)
            #    action_cmd_send = [0] * 12
            #else:
            action_cmd_send = action_cmd
            # if action_cmd_queue[0] is None:
            #     action_cmd_queue[0] = action_cmd
            #把action_cmd_queue pop出来
            # action_cmd_send = action_cmd_queue.popleft()
            #转成action相同的格式
            action_send = np.array(action_cmd_send)
            with lock:
                last_action_data = action_cmd
        except Exception as e:
            print("Error in computing policy:", e)
        publish_data.PublishAction(action_send)
        if (print_count < 6):
            #print("obs:",obs,print_count)
            print("input",policy_input)
            print('action_cmd:',action_send,print_count)
            print_count += 1
        elapsed_time = time.time() - proc_start_time
        time.sleep(max(0, time_step - elapsed_time))    # 控制每次循环为time_step = 0.01s

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
        class robot_config:
            default_dof_pos = np.array([0., 0., -0.3, 0.6, -0.4, 0.,    #   left leg joint position
                                        0., 0., -0.3, 0.6, -0.4, 0. ], dtype=np.double)
            dof_vel_limits = np.array([3.14, 3.14, 2., 1.8000, 1.57, 1.2,
                                       3.14, 3.14, 2., 1.8000, 1.57, 1.2], dtype=np.double)
            dof_pos_min = np.array([-2.5000, -0.50000, -2.0000, -0.1000, -0.600, -0.080, 
                                    -2.5000, -0.5000, -2.0000, -0.1000, -0.600, -0.080], dtype=np.double)
            dof_pos_max = np.array([0.1200, 0.5000, 0.2300, 1.8000, 0.60000, 0.080, 
                                    0.1200, 0.5000, 0.2300, 1.8000, 0.60000, 0.080], dtype=np.double)

    observation_data = ObservationNode()
    time.sleep(5)
    #print('---------------')
    #print('obs : ', observation_data)
    publish_data = RLPublish()
    

    try:
        #python scripts/sim2sim.py --load_model /path/to/export/model.pt
        policy = torch.jit.load(args.load_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
#    run_policy_thread = threading.Thread(target=run_policy, args=(policy, Sim2RealCfg(), observation_data))
#    publish_thread = threading.Thread(target=send_action, args=(publish_data,))
    run_policy(policy, Sim2RealCfg(), observation_data)
#    run_policy_thread.start()
#    publish_thread.start()
    
#    try:
#        run_policy_thread.join()
#        publish_thread.join()
#    except KeyboardInterrupt:
#        print("Interrupted by user, shutting down...")
#        publish_data.running_flag = False
#        run_policy_thread.join()
#        publish_thread.join()

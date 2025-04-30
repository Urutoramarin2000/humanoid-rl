
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
import json

lock = threading.Lock()
last_action_data = [0] * 12
time_step = 0.01

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

# 四元数转欧拉角
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])



#计算torque的，现在没有用
def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_policy(policy, cfg, publish_data, calibration_number):
    #仿真步长是1000Hz，但是上游数据是100Hz
    #sim_dt = 0.001 
    with open('/home/cowa/下载/actions_data.json', 'r') as file:
        action_data = json.load(file)
    for action in action_data: 
        start_time = time.time()
        if len(action) == 12:
            action_cmd = np.array(action)
            #min_clip_actions = np.array([-0.1, -0.1, -0.1, ...])
            min_clip_actions = -0.1
            max_clip_actions = 0.1
            action_cmd = np.clip(action_cmd, min_clip_actions, max_clip_actions)
            mask = torch.zeros_like(action_cmd, dtype=torch.bool)
            mask[calibration_number] = 1
            action_cmd *= mask
            action_cmd = action_cmd.tolist()
            publish_data.PublishAction(action_cmd)
        elapsed_time = time.time() - start_time
        time.sleep(max(0, time_step - elapsed_time)) 
 

def signal_handler(signum, frame):
    print("Interrupt received, shutting down...")
    # 执行清理操作
    publish_data.running_flag = False
    

def send_action(publish_data):    
    while True:
        with lock:
            current_action_data = action_cmd
            current_action_data = [0,0,0,0,0,0,0,0,0,0,0,0.01]
        print('current_action_data = ', current_action_data)
        publish_data.PublishAction(last_action_data)
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
    #observation_data = ObservationNode()
    #time.sleep(5)
    #print('---------------')
    #print('obs : ', observation_data)
    publish_data = RLPublish()
    

    try:
        #python scripts/sim2sim.py --load_model /path/to/export/model.pt
        policy = torch.jit.load(args.load_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    calibration_number = 1
    run_policy(policy, Sim2RealCfg(), publish_data, calibration_number)


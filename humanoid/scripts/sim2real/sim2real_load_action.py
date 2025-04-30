
import math
import numpy as np
# import mujoco, mujoco_viewer
# from tqdm import tqdm
from collections import deque
#from scipy.spatial.transform import Rotation as R
# from humanoid import LEGGED_GYM_ROOT_DIR
from cowa_config import CowaCfg
#import torch
import signal
from communication_with_robot import ObservationNode, RLPublish
import time
import threading
import os
import sys
import json

lock = threading.Lock()
last_action_data = [0] * 12
time_step = 0.01   # 100Hz
action_cmd_queue = deque(maxlen=2)
action_cmd_queue.append([0] * 12)

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def run_policy(cfg, publish_data, calibration_number):
    #仿真步长是1000Hz，但是上游数据是200Hz
    #sim_dt = 0.001 
    with open('/home/root/msg_single/actions_data.json', 'r') as file:
        action_data = json.load(file)
    for action in action_data: 
        start_time = time.time()
        if len(action) == 12:
            action_cmd = np.array(action)
            
            #min_clip_actions = np.array([-0.1, -0.1, -0.1, ...])
            # min_clip_actions = -0.9
            # max_clip_actions = 0.9
            # action_cmd = np.clip(action_cmd, min_clip_actions, max_clip_actions)
            action_cmd_queue.append(action_cmd)
            if len(action_cmd_queue) > 0:
                last_action_cmd = action_cmd_queue.popleft()
            else:
                last_action_cmd = (action_cmd + last_action_cmd) / 2
                action_cmd_queue.popleft()

            action_cmd = action_cmd.tolist()
            publish_data.PublishAction(action_cmd)
        elapsed_time = time.time() - start_time
        time.sleep(max(0, time_step - elapsed_time)) 
 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    #parser.add_argument('--load_model', type=str, required=True,
    #                    help='Run to load from.')
    args = parser.parse_args()

    #if not os.path.isfile(args.load_model):
    #    print(f"Error: Model file {args.load_model} does not exist.")
    #    sys.exit(1)

    class Sim2RealCfg(CowaCfg):

        class sim_config:
            #sim_duration = 60.0
            
            #这里改成100Hz
            dt = 0.005
            #decimation = 10

    publish_data = RLPublish()
    

    run_policy(Sim2RealCfg(), publish_data)


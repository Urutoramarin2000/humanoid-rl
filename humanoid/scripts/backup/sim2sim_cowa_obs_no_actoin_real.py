# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import json
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import CowaCfg
import torch
import time

class cmd:
    vx = 0.3
    vy = 0.00
    dyaw = 0.00


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

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''    
    # -0.65, 1.3, -0.65
    # init_data = [ 0., 0., -0,        #   root的x, y, z
    #               1., 0., 0., 0.,       #   root的四元数
    #               0., 0., -0.65, 1.3, -0.65, 0.,    #   left leg joint position
    #               0., 0., -0.65, 1.3, -0.65, 0. ]   #   right leg joint postion    
    init_data = [ 0., 0., 0.1,        #   root的x, y, z
                  1., 0., 0., 0.,       #   root的四元数
                  0., 0., -0.4, 0.8, -0.4, 0.,    #   left leg joint position
                  0., 0., -0.4, 0.8, -0.4, 0. ]   #   right leg joint postion
    q = data.qpos.astype(np.double) - init_data 
    # print('q', q)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def soft_clip(x, neg_thresholds, pos_thresholds, temperature=1):
    """
    对每个维度应用不同的非对称软裁剪。

    参数:
    x -- 要裁剪的张量。
    pos_thresholds -- 每个维度的正向裁剪阈值。
    neg_thresholds -- 每个维度的负向裁剪阈值。

    返回:
    软裁剪后的张量。
    """
    # 确保阈值张量与输入张量维度相同
    #pos_thresholds = pos_thresholds.unsqueeze(0)  # 增加批次维度
    #neg_thresholds = neg_thresholds.unsqueeze(0)

    # 处理每个维度的正数部分
    #print("soft_clip中tensor维度是一致的:", torch.equal(pos_thresholds.shape, x.shape))

    x_pos_clipped = np.tanh((x - pos_thresholds) * (1 / temperature)/ pos_thresholds) * pos_thresholds + pos_thresholds
    x_neg_clipped = np.tanh((x - neg_thresholds) * (1 / temperature) / neg_thresholds) * neg_thresholds + neg_thresholds

    # 合并正负部分
    x_clipped = np.where(x > 0, x_pos_clipped, x_neg_clipped)
    return x_clipped

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    start_time = time.time()
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # -0.65, 1.3, -0.65
    # init_data = [ 0., 0., -0.03,        #   root的x, y, z
    #               1., 0., 0., 0.,       #   root的四元数
    #               0., 0., -0.65, 1.3, -0.65, 0.,    #   left leg joint position
    #               0., 0., -0.65, 1.3, -0.65, -0. ]   #   right leg joint postion
    # -0.4, 0.8, -0.4
    init_data = [ 0., 0., 0.1,        #   root的x, y, z
                  1., 0., 0., 0.,       #   root的四元数
                  0., 0., -0.4, 0.8, -0.4, 0.,    #   left leg joint position
                  0., 0., -0.4, 0.8, -0.4, 0. ]   #   right leg joint postion
    data.qpos = init_data
    dt = cfg.sim_config.decimation * cfg.sim_config.dt
    kps_full = cfg.robot_config.tau_limit * cfg.safety.torque_limit / ( cfg.robot_config.dof_vel_limits * cfg.safety.vel_limit * dt)
    kps = kps_full / cfg.control.Kp_scale
    # kps = kps_full / 50
    print(data.qpos)   
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)

    action_list = []

    hist_obs = deque()

    cycle_time = 1
    hist_action = deque(maxlen=2)
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    hist_action.append(last_last_actions)
    hist_action.append(last_actions)
    count_lowlevel = 0
    


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        # print('pos',q)
        dq = dq[-cfg.env.num_actions:]
        # print('q',q)
        # 1000hz -> 200hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            # obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cycle_time)
            # obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cycle_time)
            if count_lowlevel < 500:
                obs[0, 0] = math.sin(2 * math.pi * (count_lowlevel * cfg.sim_config.dt)  / cycle_time)    # time.time和用仿真计算时间是一样的
                obs[0, 1] = math.cos(2 * math.pi * (count_lowlevel * cfg.sim_config.dt)  / cycle_time)
            else:
                obs[0, 0] = math.sin(2 * math.pi * (count_lowlevel * cfg.sim_config.dt)  / cycle_time)    # time.time和用仿真计算时间是一样的
                obs[0, 1] = math.cos(2 * math.pi * (count_lowlevel * cfg.sim_config.dt)  / cycle_time)
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

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # action = np.clip(action, cfg.robot_config.dof_pos_min - cfg.robot_config.default_dof_pos, cfg.robot_config.dof_pos_max - cfg.robot_config.default_dof_pos)
            action = np.clip(action, cfg.robot_config.dof_pos_min, cfg.robot_config.dof_pos_max)
            last_actions = hist_action[-1]
            last_last_actions = hist_action.popleft()

            # -------------- soft clip --------------
            last_actions_delta = last_actions - last_last_actions
            actions_delta = action - last_actions
            actions_delta_delta = actions_delta - last_actions_delta
            dof_acc_limits = cfg.safety.dof_acc_limits_ratio * cfg.robot_config.dof_vel_limits
            actions_delta_delta = soft_clip(actions_delta_delta, neg_thresholds = -cfg.safety.acc_limit * dof_acc_limits * dt, pos_thresholds = cfg.safety.acc_limit * dof_acc_limits*dt, temperature=2)        
            # action的变化率在dof的vel可执行范围
            actions_delta = last_actions_delta + actions_delta_delta
            actions_delta = soft_clip(actions_delta, neg_thresholds = -cfg.safety.vel_limit * cfg.robot_config.dof_vel_limits * dt, pos_thresholds = cfg.safety.vel_limit * cfg.robot_config.dof_vel_limits * dt,  temperature=2)             
            clamp_actions = last_actions + actions_delta
            # action限制在dof的pos可执行范围  
            clamp_actions_array = np.array(clamp_actions, dtype=np.double)
            action = np.clip(clamp_actions_array, cfg.robot_config.dof_pos_min - cfg.robot_config.default_dof_pos, cfg.robot_config.dof_pos_max - cfg.robot_config.default_dof_pos)
            ratio = 1
            action_ratio = ratio * action + (1 - ratio) * last_actions    # 滤波
            hist_action.append(action_ratio)
            action_list.append(action_ratio.tolist())
            # # -------------- 直出 --------------
            # ratio = 0.9
            # action_ratio = 0.9 * action + (1 - ratio) * last_actions    # 直出
            # hist_action.append(action_ratio)
            
            target_q = action_ratio * cfg.control.action_scale


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit * cfg.safety.torque_limit, cfg.robot_config.tau_limit * cfg.safety.torque_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

        # viewer.render()
        # paused = True
        # if not paused:
        #     mujoco.mj_step(model, data)
            


    with open('action_data_mujoco.json','w') as file:
        json.dump(action_list, file)
    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(CowaCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/assembly_cowa_robot_new_foot4/mjcf/cowa_robot.xml'
            sim_duration = 30.0
            dt = 0.001
            decimation = 4

        class robot_config:
            # default_dof_pos = np.array([0., 0., -0.65, 1.3, -0.65, 0.,    #   left leg joint position
            #                             0., 0., -0.65, 1.3, -0.65, 0. ], dtype=np.double)
            default_dof_pos = np.array([0., 0., -0.4, 0.8, -0.4, 0.,    #   left leg joint position
                                        0., 0., -0.4, 0.8, -0.4, 0. ], dtype=np.double)
            dof_vel_limits = np.array([0.6000, 0.6000, 2.5000, 1.5000, 2.2000, 1.2000, 0.6000, 0.6000, 2.5000,
                                       1.5000, 2.2000, 1.2000], dtype=np.double)
            dof_pos_min = np.array([-1.2000, -0.50000, -2.0000, -0.1000, -1.000, -0.300, 
                                    -1.2000, -0.5000, -2.0000, -0.1000, -1.000, -0.300], dtype=np.double)
            dof_pos_max = np.array([0.1200, 0.5000, 0.2300, 1.8000, 0.50000, 0.300, 
                                    0.1200, 0.5000, 0.2300, 1.8000, 0.50000, 0.300], dtype=np.double)
            # dof_pos_min = np.array([-1.2000, -1.0000, -2.0000, -0.1000, -1.0000, -0.5500, 
            #                         -1.2000, -1.0000, -2.0000, -0.1000, -1.0000, -0.5500], dtype=np.double)
            # dof_pos_max = np.array([0.1200, 1.0000, 0.2300, 1.8000, 1.0000, 0.5500, 
            #                         0.1200, 1.0000, 0.2300, 1.8000, 1.0000, 0.5500], dtype=np.double)
                                     
        #     kps_full = np.array([37777.7773, 37777.7773, 22666.6680, 37777.7773,  7727.2720, 14166.6650,
        # 37777.7773, 37777.7773, 22666.6680, 37777.7773,  7727.2720, 47222.2188], dtype=np.double)
        #     kps = kps_full / 15
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = np.array([100., 100., 250., 250.,  75.,  75., 100., 100., 250., 250.,  75., 75.], dtype=np.double)
            soft_clip = True

    policy = torch.jit.load(args.load_model)
    print('policy-----------------',policy)
    run_mujoco(policy, Sim2simCfg())

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR
import time
# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
# import matplotlib
# matplotlib.use('TkAgg') 
# import matplotlib.pyplot as plt
import json

def play(args):
    args.task = "cowa_ppo"
    args.run_name = 'v4'
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1024)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    #env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.domain_rand.randomize_rigids_after_start = False
    env_cfg.domain_rand.randomize_obs_update_delay = False
    env_cfg.domain_rand.randomize_actions_update_delay = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.control.action_smoothness = True
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.terrain.static_friction = 1
    env_cfg.terrain.dynamic_friction = 1


    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 120 # number of steps before plotting states
    actions_list = []
    actions_delat_list = []
    actions_delat_delat_list = []
    torque_list =  []
    dof_pos_list = []
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        # camera 足部特写
        # camera_offset = gymapi.Vec3(0.5, -2, -0.3)
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
        #                                             np.deg2rad(90))
        # camera 正常全身
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    last_pose = 0
    motor_vel_list = [[] for i in range(12)]
    ang_vel_list = []
    feet_height_list = []
    air_time_list = []
    feet_height_smooth_list = []
    vel_list = [[] for i in range(3)]
    action_output_list = [[] for i in range(12)]
    count = 0
    # for i in tqdm(range(stop_state_log)):
    for i in range(stop_state_log):
        # start_time = time.time()
        actions = policy(obs.detach()) 

        # actions_delat_list.append(actions_delta.cpu().detach().numpy().tolist()[0])
        # actions_delat_delat_list.append(actions_delta_delta.cpu().detach().numpy().tolist()[0])
        # end_time = time.time()
        # print('inference time: ', end_time - start_time)        p
        if FIX_COMMAND:
            env.commands[:, 0] = 0.0 # 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0
            env.commands[:, 3] = 0.0

        actions_plot = env.actions
        print(actions_plot.shape)


        # actions_list.append(actions_plot.cpu().detach().numpy().tolist()[0])
        # torque_list.append(env.torques.cpu().detach().numpy().tolist()[0])
        # dof_pos_list.append((env.dof_pos - env.default_dof_pos).cpu().detach().numpy().tolist()[0])
        actions_list.append(actions_plot.cpu().detach().numpy().tolist())
        torque_list.append(env.torques.cpu().detach().numpy().tolist()[0])
        dof_pos_list.append((env.dof_pos - env.default_dof_pos).cpu().detach().numpy().tolist())       
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())


        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
        #current_pose = env.dof_pos[robot_index, joint_index].item() * env_cfg.control.action_scale
        #current_pose_vel = (current_pose - last_pose) / 0.64
        #last_pose = current_pose
        #print('Current pose velocity: ', current_pose_vel)
        for i in range(12):
            motor_vel_list[i].append(env.dof_vel[robot_index, i].item())
        for i in range(12):
            action_output_list[i].append(actions[robot_index,i].item())
        ang_vel_list.append(env.base_ang_vel[robot_index, 2].item())
        feet_height_list.append((env.rigid_state[robot_index, 5, 2] - 0.14).item())
        air_time_list.append(env.feet_air_time[robot_index, 0].item())
        # feet_height_smooth_list.append(env.feet_height_smooth_sum[robot_index].item())
        for i in range(3):
            vel_list[i].append(env.base_lin_vel[robot_index, i].item())

        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env_cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
    if RENDER:
        video.release()
    print('size',len(actions_list))
    with open('actions_database.json','w') as file:
        json.dump(actions_list, file)
        # print(actions_list.shape)
    # with open('torque.json','w') as file:
    #     json.dump(torque_list, file)
    with open('dof_pos_listbase.json','w') as file:
        json.dump(dof_pos_list, file) 
    print("saved")
    # with open('actions_delat_data.json','a') as file:
    #     json.dump(actions_delat_list,file)
    # with open('actions_delat_delat_list.json','a') as file:
    #     json.dump(actions_delat_delat_list,file)
    print("--------------------------------")
    print("--------------------------------")
    # print("被clip的次数: ", count)
    # with open('motor_velocity_data.json', 'w') as file:
    #     json.dump(motor_vel_list, file)
    # with open('angle_velocity_data.json', 'w') as file:
    #     json.dump(ang_vel_list, file)  feet_height_smooth_list
    # with open('feet_height_smooth_list.json', 'w') as file:
    #     json.dump(feet_height_smooth_list, file)    
    # with open('feet_height_data.json', 'w') as file:
    #     json.dump(feet_height_list, file)  
    # with open('air_time_data.json', 'w') as file:
    #     json.dump(air_time_list, file) 
    # with open('velocity_data.json', 'w') as file:
    #     json.dump(vel_list, file)
    # with open('action_output_list.json', 'w') as file:
    #     json.dump(action_output_list , file)          
    logger.print_rewards()
    # logger.plot_states()
    
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)

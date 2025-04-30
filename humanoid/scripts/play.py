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
import json
import copy

class Integrated_RMA_policy(torch.nn.Module):
    def __init__(self, actor, adaptation_encoder, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.adaptation_encoder = copy.deepcopy(adaptation_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        latent = self.adaptation_encoder.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        return action_mean

    def export(self, path):
        path = os.path.join(path, "policy_rma.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class Integrated_EST_policy(torch.nn.Module):
    def __init__(self, actor, estimator, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        latent = self.estimator.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        return action_mean

    def export(self, path):
        path = os.path.join(path, "policy_est.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 


class Integrated_VAE_policy(torch.nn.Module):
    def __init__(self, actor, encoder, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.encoder = copy.deepcopy(encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        [z, vel], [latent_mu, latent_var, vel_mu, vel_var] = self.encoder.inference(obs[:,:(self.frame_stack - self.actor_input_stack) * self.num_single_obs])
        latent = torch.cat([latent_mu, vel_mu], dim=1)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        return action_mean

    def export(self, path):
        path = os.path.join(path, "policy_est.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 

def play(args):
    # args.task = "cowa_fix"
    # args.task = "cowa_est"
    # args.task = "cowa_vae"
    args.run_name = 'v4'
    print('args.task-------------------------',args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.asset.fix_base_link = False
    env_cfg.init_state.rand_init_dof = False
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3)
    env_cfg.env.lying_reset = False
    env_cfg.sim.max_gpu_contact_pairs = 2**10


    env_cfg.viewer.draw_base_com = True
    env_cfg.domain_rand.push_robots = False
    # added xxx
    env_cfg.domain_rand.randomize_rigids_after_start = False     #用于控制加到priv里面的东西，目前用不到
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.1, 0.2]  

    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [4.999, 5]    
    env_cfg.domain_rand.randomize_restitution = False           #加到priv里的东西   # 建议不加
    env_cfg.domain_rand.randomize_com_displacement = False      #加到priv里的东西
    env_cfg.domain_rand.com_displacement_range = [-0.000001, 0.00000001]
    env_cfg.domain_rand.randomize_inertia = True  
    env_cfg.domain_rand.randomize_inertia_range = [1.0000, 1.000001]  

    env_cfg.domain_rand.randomize_motor_strength = True        #加到priv里的东西
    env_cfg.domain_rand.motor_strength_range = [0.9, 0.900001] 
    env_cfg.domain_rand.randomize_PD_factor = True #             
    env_cfg.domain_rand.Kp_factor_range = [1.099, 1.1]            
    env_cfg.domain_rand.Kd_factor_range = [1.0999, 1.1]
    env_cfg.domain_rand.randomize_motor_offset = True #
    env_cfg.domain_rand.motor_offset_range = [-0.02, 0.02]

    env_cfg.domain_rand.randomize_lag_timesteps = True      # 模拟delay，对于lag用于给历史的action
    env_cfg.domain_rand.lag_timesteps = 2
    env_cfg.domain_rand.randomize_torque_delay = True
    env_cfg.domain_rand.torque_delay_steps = 4
    # env_cfg.domain_rand.randomize_obs_delay = False
    # env_cfg.domain_rand.obs_delay_steps = 1

    # agibot 添加了, vel,imu延迟    '
    env_cfg.domain_rand.add_dof_lag = True
    env_cfg.domain_rand.dof_lag_timesteps_range = [0, 4]
    env_cfg.domain_rand.add_imu_lag = True                    # 这个是 imu 的延迟
    env_cfg.domain_rand.imu_lag_timesteps_range = [5, 12] # 10~22ms

    # 关节 阻尼&摩擦
    env_cfg.domain_rand.randomize_joint_friction = True  
    env_cfg.domain_rand.joint_friction_range = [0.99, 1.00]      
    env_cfg.domain_rand.randomize_joint_damping = True 
    env_cfg.domain_rand.joint_damping_range = [0.99, 1.00]
    env_cfg.domain_rand.randomize_joint_armature = True

    env_cfg.domain_rand.randomize_coulomb_friction = True
    env_cfg.domain_rand.joint_coulomb_range = [0.9, 0.9]
    env_cfg.domain_rand.joint_viscous_range = [0.7, 0.70]
    env_cfg.domain_rand.action_noise = 0.0
    env_cfg.domain_rand.action_delay = 0.0    
    env_cfg.control.action_smoothness = False
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 1
    

    train_cfg.seed = 1000
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
        if not os.path.exists(path):
            os.makedirs(path)
        if args.task == "cowa_rma":
            print("==-=-=-=-=-=",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_RMA_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.adaptation_encoder,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path)
        elif args.task == "cowa_est":
            print("==-=-=-=-=-=",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path)
        elif args.task == "cowa_vae":
            print("==-=-=-=-=-=",env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.vae.encoder,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack - 1, 1)
        else:
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    actions_list = []
    actions_delat_list = []
    actions_delat_delat_list = []
    torque_list =  []
    dof_pos_list = []
    ref_pos_list = []
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

    motor_vel_list = []
    ang_vel_list = []
    base_euler_xyz_list = []
    feet_dis_list = []
    feet_height_list = []
    air_time_list = []
    feet_height_smooth_list = []
    vel_list = [[] for i in range(3)]
    count = 0

    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    # for i in tqdm(range(stop_state_log)):
    for i in range(stop_state_log):
        if args.task == "cowa_rma":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        elif args.task == "cowa_est":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        elif args.task == "cowa_dual":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        elif args.task == "cowa_vae":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - 1) * env_cfg.env.num_single_obs:], obs.detach()[:,:(env_cfg.env.frame_stack - 1) * env_cfg.env.num_single_obs])

        else :
            actions = policy(obs.detach()) 

        # actions_delat_list.append(actions_delta.cpu().detach().numpy().tolist()[robot_index])
        # actions_delat_delat_list.append(actions_delta_delta.cpu().detach().numpy().tolist()[robot_index])
        # end_time = time.time()
        # print('inference time: ', end_time - start_time)        p
        if FIX_COMMAND:
            env.commands[:, 0] = 0. # 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.0
        else:
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

            if CoM_offset_compensate:
                if i > 200 and i < 600:
                    vel_cmd[:] = env.commands[:, 0] #* np.clip((i - 200) / 400.0, 0, 1)
                # elif i>= 800 and i < 1400:
                #     vel_cmd[:] = env.commands[:, 0] * np.clip((1400 - i) / 600.0, 0, 1)
                else:
                    vel_cmd[:] = 0
                vel_err_intergral += (
                    (vel_cmd - env.base_lin_vel[:, 0])
                    * env.dt
                    * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
                )
                vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
                env.commands[:, 0] = vel_cmd + vel_err_intergral
        # print("vel_x:", env.base_lin_vel[:,0])

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        # print(env.lag_buffer)
        actions_scaled = 0.25 * actions
        hip_roll_yaw_indices = [0,1,6,7]
        # ankle_pitch_indices = [4,10]
        actions_scaled[:,hip_roll_yaw_indices] *= env_cfg.control.extra_hip_roll_yaw_scale
        # actions_scaled[:,ankle_pitch_indices] *= env_cfg.control.extra_ankle_pitch_scale
        actions_list.append(actions_scaled.cpu().detach().numpy().tolist()[robot_index])
        ref_pos_list.append((env.ref_dof_pos - env.default_dof_pos).cpu().detach().numpy().tolist()[robot_index])
        torque_list.append(env.torques.cpu().detach().numpy().tolist()[robot_index])
        dof_pos_list.append((env.dof_pos - env.default_dof_pos).cpu().detach().numpy().tolist()[robot_index])
        # feet_dis_list.append(env.foot_dist.cpu().detach().numpy().tolist()[robot_index])

        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_pos[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
        
        motor_vel_list.append(env.dof_vel.cpu().detach().numpy().tolist()[robot_index])
        ang_vel_list.append(env.base_ang_vel[robot_index, 2].item())
        base_euler_xyz_list.append(env.base_euler_xyz.cpu().detach().numpy().tolist()[robot_index])
        
        
        feet_height_list.append((env.rigid_state[robot_index, 5, 2] - 0.05).item())
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
    # print('size',len(actions_list))
    with open('./data/actions_data.json','w') as file:
        json.dump(actions_list, file)
    with open('./data/feet_dis_list.json','w') as file:
        json.dump(feet_dis_list, file)        
    with open('./data/ref_pos_list.json','w') as file:
        json.dump(ref_pos_list, file)
    with open('./data/torque.json','w') as file:
        json.dump(torque_list, file)
    with open('./data/dof_pos_list.json','w') as file:
        json.dump(dof_pos_list, file) 
    print("saved")
    with open('./data/actions_delat_data.json','a') as file:
        json.dump(actions_delat_list,file)
    with open('./data/actions_delat_delat_list.json','a') as file:
        json.dump(actions_delat_delat_list,file)
    print("--------------------------------")
    print("--------------------------------")
    print("被clip的次数: ", count)
    with open('./data/motor_velocity_data.json', 'w') as file:
        json.dump(motor_vel_list, file)
    # with open('angle_velocity_data.json', 'w') as file:
    #     json.dump(ang_vel_list, file)  feet_height_smooth_list
    with open('./data/feet_height_smooth_list.json', 'w') as file:
        json.dump(feet_height_smooth_list, file)    
    with open('./data/feet_height_data.json', 'w') as file:
        json.dump(feet_height_list, file)  
    with open('./data/air_time_data.json', 'w') as file:
        json.dump(air_time_list, file) 
    with open('./data/velocity_data.json', 'w') as file:
        json.dump(vel_list, file)     
    with open('./data/base_euler_xyz_data.json', 'w') as file:
        json.dump(base_euler_xyz_list, file)  
       
    logger.print_rewards()
    # logger.plot_states()
    
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)

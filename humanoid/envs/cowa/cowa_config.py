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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CowaCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 3
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 71 + 12 + 1 + 1 + 1 + 3 + 12 + 12 + 1 + 1 + 1 + 1 + 2 + 3
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 2048
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

        # privileged obs
        priv_observe_friction = True  # 1
        priv_observe_base_mass = True # 1
        priv_observe_restitution = True #1
        priv_observe_com_displacement = True    # 3
        
        priv_observe_motor_strength = True  # 12
        priv_observe_motor_offset = True    # 12 感觉还是不要了比较好
        priv_observe_body_height = True # 1
        priv_observe_gravity = False    # 3
        priv_observe_measure_heights = False # 187
    class safety:
        # safety factors
        pos_limit = 0.9
        # vel_limit = 0.8
        # acc_limit = 0.8
        vel_limit = 1
        acc_limit = 0.6
        dof_acc_limits_ratio = 6    # acc_limit = dof_acc_limits_ratio * dof_vel_limits
        torque_limit = 1    #0.85 xxx 1

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_robot/urdf/cowa_robot copy 2.urdf' # wh
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_robot/urdf/cowa_robot.urdf' # wh
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowanoid_description/urdf/cowanoid_simplied_0319.urdf' # xxw
        # /home/zhengde.ma/humanoid-gym-main/resources/robots/assembly_cowa_robot_new_foot3/urdf
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/urdf/XBot-L.urdf' # wh
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/assembly_cowa_robot_new_foot4/urdf/assembly_cowa_robot_new_foot4.urdf' # wh
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_robot_whole_body/urdf/cowa_robot_whole_body.urdf' # wh
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_humanoid/urdf/cowa_humanoid.urdf' # wh
        #cowa_humanoid_debug_com        # cowa_humanoid
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_humanoid_big_foot/urdf/cowa_humanoid_big_foot.urdf' # wh
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_humanoid_new_foot/urdf/cowa_humanoid_new_foot.urdf'

        name = "cowa_robot"
        foot_name = "ankle_roll"
        knee_name = "knee"
        disable_gravity = False
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False #
        replace_cylinder_with_capsule = False
        fix_base_link = False
        fix_base_link_height = 1.8  # fix the base of the robot at the height
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 1
        dynamic_friction = 1
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 5  # number of terrain cols (types)
        max_init_terrain_level = 0  # starting curriculum state  wende 改的，本来是10
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        # terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        terrain_proportions = [0, 0, 0, 1., 0., 0, 0]   #wende 改的，本来是[0, 0, 1, 0., 0., 0, 0] 
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.1
            dof_vel = 0.1 #0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.05 # 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        # 大腿 0.404    小腿 0.388      脚踝到脚底 0.05    root到大腿0.442
        # pos = [0.0, 0.0, 1.22]  # 14 dof版本, 
        # pos = [0.0, 0.0, 1.31]
        pos = [0.0, 0.0, 1.06]  # 12dof版本,半蹲高度1.06, 地形最高高度为0.1
        rot = [0.0,  0.0, 0.0, 1] # x,y,z,w [quat]
        rand_init_dof = True
        default_joint_angles = {  
            'left_leg_roll_joint': 0.0,
            'left_leg_yaw_joint': 0.0,
            'left_ankle_roll_joint': 0.0,
            'right_leg_roll_joint': 0.0,
            'right_leg_yaw_joint': 0.0,

            'right_ankle_roll_joint': 0.0,

            # 'left_leg_pitch_joint': -0.4,      # Unitree pos
            # 'left_knee_joint': 0.8,     
            # 'left_ankle_pitch_joint': -0.4, 
            # 'right_leg_pitch_joint': -0.4,   
            # 'right_knee_joint': 0.8,
            # 'right_ankle_pitch_joint': -0.4,

            'left_leg_pitch_joint': 0.0758,      # 原地可站立
            'left_knee_joint': 0.2638,     
            'left_ankle_pitch_joint': 0.188,  #0.188, 
            'right_leg_pitch_joint': 0.0758,   
            'right_knee_joint': 0.2638,
            'right_ankle_pitch_joint': 0.188,    #0.188,

            # 'left_upper_arm_pitch_joint': 0.0,
            # 'right_upper_arm_pitch_joint': 0.0
        }


    class control(LeggedRobotCfg.control):
        # PD Drive parameters:   xxw ，leg_pitch 和 knee 原本是500，wende改成800
        stiffness = {'leg_roll': 600.0, 'leg_yaw': 800.0, 'leg_pitch': 200, 
                     'knee': 290.0, 'ankle_pitch': 100, 'ankle_roll': 150}
        damping = {'leg_roll': 3, 'leg_yaw': 3, 'leg_pitch': 15, 
                    'knee': 15, 'ankle_pitch': 3, 'ankle_roll': 2}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1/3.0  # 0.25
        
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 50Hz=200/4 # 100Hz=500/5

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        action_clip_mode = 0 # see GymDofDriveModeFlags (0 is just clip actoin, 1 is clip action/vel/acc, 2 is soft_clip action/vel/acc, 3 is soft_clip + hanning window)

        passive_ankle_roll_joint = True
        # extra_hip_roll_yaw_scale = 1 # 2/3.
        clip_ankle_pitch = False
        clip_hip_roll = False

    class sim(LeggedRobotCfg.sim):
        # web_vis = True
        # port = 6001      # zmp的端口，如果训练启动卡住，则这个端口被占用，更改以下就行
        # web_vis_envs = 1 #用于控制web可视化的机器人个数，建议不要超过5个，性能影响很大
        # keep_default_viewer = True

        dt = 0.002  # 500 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  # xxw
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):


        push_robots = False 
        push_interval_s = 4
        max_push_vel_xy = 0.2   # 0.2
        max_push_ang_vel = 0.4  # 0.4

        action_noise = 0.02
        action_delay = 0.1

        rand_interval_s = 10    ## TODO
        randomize_rigids_after_start = True     #用于控制加到priv里面的东西，目前用不到
        randomize_friction = True             # xxw True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-2.5, 2.5]              # 加负载
        randomize_restitution = True           #加到priv里的东西
        restitution_range = [0, 1.0]            #加到priv里的东西
        randomize_com_displacement = True      #加到priv里的东西
        com_displacement_range = [-0.03, 0.03]  #加到priv里的东西
        randomize_inertia = True    
        randomize_inertia_range = [0.9, 1.1]

        randomize_motor_strength = True        #加到priv里的东西
        motor_strength_range = [0.9, 1.1]       #加到priv里的东西

        randomize_PD_factor = True #             
        Kp_factor_range = [0.9, 1.1]            
        Kd_factor_range = [0.9, 1.1]

        randomize_motor_offset = True #
        # default_motor_offset = [0.00, -0.00, 0.007, 0.02, 0.0, 0.0,\
        #                         0.0, -0.00, 0.007, 0.02, 0.0, 0.0,]
        motor_offset_range = [-0.03, 0.03]

        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0

        randomize_gravity = False # 建议不加
        gravity_range = [-1.0, 1.0]         #

        randomize_lag_timesteps = True      # 模拟delay，对于lag用于给历史的action
        lag_timesteps = 2       #2~4ms walk these ways 加固定action延迟

        randomize_torque_delay = True
        torque_delay_steps = 4

        # randomize_obs_delay = False #用队列加固定obs延迟
        # obs_delay_steps = 1

        # agibot
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = True
        dof_lag_timesteps_range = [0, 4] # 1~4ms

        # add_dof_pos_vel_lag = False        # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟同
        # randomize_dof_vel_lag_timesteps = True
        # randomize_dof_vel_lag_timesteps_perstep = True          # 不常用always False
        # dof_vel_lag_timesteps_range = [7, 25]
        
        add_imu_lag = True                    # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = True         # 不常用always False
        imu_lag_timesteps_range = [5, 12] # 实际10~22ms

        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.10, 0.70]

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = True
        # default_joint_friction = [0.02, 0.02, 0.0002, 0.0002, 0.0002, 0.1,\
        #                           0.02, 0.02, 0.0002, 0.0002, 0.0002, 0.1,] #new
        default_joint_friction = [0.02, 0.02, 0.005, 0.02, 0.1, 0.1,\
                                 0.02, 0.02, 0.003, 0.02, 0.1, 0.1]                          
        joint_friction_range = [0.8, 1.2]
        # joint_friction_range = [1.5, 1.5]
        joint_1_friction_range = [0.9, 1.1]
        joint_2_friction_range = [0.9, 1.1]
        joint_3_friction_range = [0.9, 1.1]
        joint_4_friction_range = [0.9, 1.1]
        joint_5_friction_range = [0.9, 1.1]
        joint_6_friction_range = [0.9, 1.1]
        joint_7_friction_range = [0.9, 1.1]
        joint_8_friction_range = [0.9, 1.1]
        joint_9_friction_range = [0.9, 1.1]
        joint_10_friction_range = [0.9, 1.1]
        joint_11_friction_range = [0.9, 1.1]
        joint_12_friction_range = [0.9, 1.1]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = True
        # default_joint_damping = [40, 30, 3., 15, 6, 70,
        #                          40, 30, 3., 15, 6, 70] # new
        default_joint_damping = [20, 15, 1, 15, 5, 70,\
                                 30, 15, 1, 15, 5, 70]
        joint_damping_range = [0.8, 1.2]
        joint_1_damping_range = [0.9, 1.1]
        joint_2_damping_range = [0.9, 1.1]
        joint_3_damping_range = [0.9, 1.1]
        joint_4_damping_range = [0.9, 1.1]
        joint_5_damping_range = [0.9, 1.1]
        joint_6_damping_range = [0.9, 1.1]
        joint_7_damping_range = [0.9, 1.1]
        joint_8_damping_range = [0.9, 1.1]
        joint_9_damping_range = [0.9, 1.1]
        joint_10_damping_range = [0.9, 1.1]
        joint_11_damping_range = [0.9, 1.1]
        joint_12_damping_range = [0.9, 1.1]

        randomize_joint_armature = True   
        randomize_joint_armature_each_joint = False
        # default_joint_armature = [0.5, 0.5, 0.3, 0.5, 0.3, 0.5,\
        #                           0.5, 0.5, 0.3, 0.5, 0.3, 0.5] # new
        # joint_armature_range = [0.8, 1.2]     # Factor
        # joint_1_armature_range = [0.8, 1.2]
        # joint_2_armature_range = [0.8, 1.2]
        # joint_3_armature_range = [0.8, 1.2]
        # joint_4_armature_range = [0.8, 1.2]
        # joint_5_armature_range = [0.8, 1.2]
        # joint_6_armature_range = [0.8, 1.2]
        # joint_7_armature_range = [0.8, 1.2]
        # joint_8_armature_range = [0.8, 1.2]
        # joint_9_armature_range = [0.8, 1.2]
        # joint_10_armature_range = [0.8, 1.2]
        # joint_11_armature_range = [0.8, 1.2]
        # joint_12_armature_range = [0.8, 1.2]
        joint_armature_range = [0.5, 0.647]     # Factor
        joint_1_armature_range = [0.5, 0.647]
        joint_2_armature_range = [0.5, 0.647]
        joint_3_armature_range = [0.5, 0.647]
        joint_4_armature_range = [0.5, 0.647]
        joint_5_armature_range = [0.5, 0.647]
        joint_6_armature_range = [0.5, 0.647]
        joint_7_armature_range = [0.5, 0.647]
        joint_8_armature_range = [0.5, 0.647]
        joint_9_armature_range = [0.5, 0.647]
        joint_10_armature_range = [0.5, 0.647]
        joint_11_armature_range = [0.5, 0.647]
        joint_12_armature_range = [0.5, 0.647]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
                                # xxw True wh False
        class ranges:
            lin_vel_x = [-0.01,0.01]  # min max [m/s]
            # lin_vel_y = [-0.5, 0.5]   # min max [m/s]  0.02
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]  0.3 xxw 0.02 wh 1
            heading = [-3.14, 3.14]
            lin_vel_y = [-0.01, 0.01]   # min max [m/s]  0.02
            ang_vel_yaw = [-0.01, 0.01]    # min max [rad/s]  0.3 xxw 0.02 wh 1

    class rewards:
        base_height_target = 1.06         #wh 直立1.1   曲膝 1.06  pos = [0.0, 0.0, 1.36]  # 14 dof版本, 
        min_feet_dist = 0.24  # 原地踏步 
        max_feet_dist = 0.29  # 原地踏步
        min_knee_dist = 0.24  # 原地踏步
        max_knee_dist = 0.27  # 原地踏步
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.2   # rad=11.7度=0.2rad T=0.8
        target_feet_height = 0.04      # 77.7*cos(0.85rad) * (1-cos(0.2rad)) =  
        cycle_time = 1.2               # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True  # xxw  True  # wh False
        # tracking reward = exp(error*sigma)
        # tracking_sigma = 4    # vel = 0.5 对应 20; vel = 1 对应 4; vel = 1.5 对应 2; vel = 2 对应 1; vel = 3 对应 0.5
        tracking_sigma_vel_x = 20
        tracking_sigma_vel_y = 20
        tracking_sigma_ang_vel = 20
        max_contact_force = 800  # Forces above this value are penalized xxx 1400
        tracking_vel_enhance = True
        tracking_vel_hard = False    # 非常严苛
        class scales:
            # #  --------------- Humanoid Gym ----------------
            # # reference motion tracking
            joint_pos = 1.4     # 1.4 wende修改成2，原本是1.4; wh 4
            # # 防止roll外翻的惩罚项
            leg_roll_joint_pos_outside = -0.10
            leg_roll_joint_pos_inside = -0.10
            feet_clearance = 1 # 原本是1.0 ，wende改成0
            feet_contact_number = 1.6  # 1.2
            # # gait
            feet_air_time = 0.5   # xxw 0.5 # wh 2 wende改成3，原本是2
            foot_slip = -0.05   # xxw -0.05 # wh -0.4
            feet_distance = 0.4  #  xxw  本来是0.4， wende改成1
            knee_distance = 0.3   #  xxw  本来是0.3， wende改成1
            # # contact
            feet_contact_forces = -0.01   # xxw -0.01
            # # vel tracking
            # # tracking_lin_vel = 3.0  
            tracking_lin_vel_x = 1.2   # 分开进行vel_x和vel_y奖励计算，本来是3.0，wende改成3.5 ; wh 4
            tracking_lin_vel_y = 0.5
            tracking_ang_vel = 1.1
            # # track_vel_hard = 0.6
            # # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # # low_speed = 0.2     # 0.2 # wh 1
            # # base pos
            default_joint_pos = 0.4   
            orientation = 1.4  

            feet_rotation = 0.3
            # base_height = 0.2   
            base_acc = 0.3   
            # # feet_height_smoothness = -10
            stand_still_vel_penality = -0.3
            # stand_still_base_pos_penality = -1  # for only step at the origin point
            # stand_still_base_pos = 0.3    # for only step at the origin point
            # # energy
            # # real_action_smoothness = -0.05 # -1e-4
            action_smoothness = -0.05
            torques = -1e-5   #  -1e-5
            # # dof_vel = 1   # -5e-4
            # # dof_acc = 2  # xxw
            # # collision = -1.
            # #  --------------- Unitree Gym ----------------
            # # Unitree_termination = -0.0
            # # # Unitree_tracking_lin_vel = 10.0
            # # # Unitree_tracking_ang_vel = 5
            # # tracking_lin_vel_x = 10   # 分开进行vel_x和vel_y奖励计算，本来是3.0，wende改成3.5 ; wh 4
            # # tracking_lin_vel_y = 5
            # # tracking_ang_vel = 5
            Unitree_lin_vel_z = -0.1
            Unitree_ang_vel_xy = -0.4
            # Unitree_orientation = -0.5
            # # Unitree_base_height = -100.0
            Unitree_dof_acc = -1e-5
            Unitree_dof_vel = -1e-3 # -5e-4
            # # Unitree_feet_air_time = 1.0
            # # Unitree_collision = 0.0
            # # Unitree_action_rate = -0.01
            # # Unitree_torques = 0.0
            # # Unitree_dof_pos_limits = -10.0
            # # Unitree_feet_stumble = -0.0 
            # # Unitree_stand_still = -0.
            #  --------------- 悬挂实验 Gym ----------------
            # joint_pos = 1.4     # 1.4 wende修改成2，原本是1.4; wh 4
            # leg_roll_joint_pos_outside = -0.10
            # leg_roll_joint_pos_inside = -0.10
            # feet_distance = 0.4  #  xxw  本来是0.4， wende改成1
            # knee_distance = 0.3   #  xxw  本来是0.3， wende改成1
            # default_joint_pos = 0.4   
            # # feet_rotation = 0.3
            # action_smoothness = -0.05
            # torques = -1e-5   #  -1e-5
            # Unitree_dof_acc = -1e-5
            # Unitree_dof_vel = -1e-3 # -5e-4

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 1 # 0.05
            quat = 1.
            height_measurements = 5.0
            torques = 0.02
            dist = 4
        clip_observations = 100.
        clip_actions = 100.
        


class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2 ##2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa'
        run_name = 'ppo'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt
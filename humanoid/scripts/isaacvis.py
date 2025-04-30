import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import time
import sys
import select

def get_input(timeout=0.01):
    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    return None

def main():
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 1
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 10
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 0.001
    sim_params.gravity = gymapi.Vec3(0, 0, 0)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())



    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    # gym.add_ground(sim, plane_params)

    urdf_location = "/home/xiaxianxian/isaac_viz/cowa/urdf"
    urdf_file = "cowa_wheel_legged_rea.urdf"  
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True  
    asset = gym.load_asset(sim, urdf_location, urdf_file, asset_options)
    
    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)


    cam_pos = gymapi.Vec3(2, 0, 2)
    cam_target = gymapi.Vec3(0, 0, 2)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
    pose.r = gymapi.Quat(-0, 0.0, 0.0, 1)
    robot_handle = gym.create_actor(env, asset, pose, "robot", 1, 1)



    dof_props = gym.get_actor_dof_properties(env, robot_handle)
    dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(0.0)
    dof_props['damping'].fill(0.0)
    gym.set_actor_dof_properties(env, robot_handle, dof_props)

    # rigid_body_props = gym.get_actor_rigid_body_properties(env, robot_handle)
    # for body in rigid_body_props:
    #     body.mass = 0.0
    #     body.inertial = gymapi.Vec3(0.0, 0.0, 0.0)
    # gym.set_actor_rigid_body_properties(env, robot_handle, rigid_body_props)

    dof_states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL)
    dof_states['pos'][:] = 0.0
    dof_states['vel'][:] = 0.0
    gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)
    num_dofs = len(dof_states)
    print("关节数量：", num_dofs)

    target_positions = np.zeros(num_dofs)  

    print("输入关节位置：")
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.sync_frame_time(sim)
        gym.draw_viewer(viewer, sim, True)

        user_input = get_input()
        if user_input:
            try:
                input_values = [float(x) for x in user_input.split()]
                if len(input_values) != num_dofs:
                    print(f"输入的关节数应为 {num_dofs}，请重新输入！")
                else:
                    target_positions = np.array(input_values)
                    print(f"设置新目标关节位置：{target_positions}")
            except ValueError:
                print("输入无效，请输入有效的数字！")
        for _ in range(10):
            dof_states['pos'][:num_dofs] = target_positions
            gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_POS)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
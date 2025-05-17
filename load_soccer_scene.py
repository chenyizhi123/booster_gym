import os
import math
import numpy as np


from isaacgym import gymapi

from isaacgym import gymtorch
import torch
from isaacgym.torch_utils import to_torch, quat_from_euler_xyz

def main():
    # 初始化gym
    gym = gymapi.acquire_gym()

    # 创建模拟环境
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0  # 60Hz的仿真频率
    sim_params.substeps = 2
    
    # 物理参数设置
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # 配置PhysX
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1  # TGS solver
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        print("创建模拟器失败")
        return
    
    # 设置地面平面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    gym.add_ground(sim, plane_params)
    
    # 创建观察者相机
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    viewer = gym.create_viewer(sim, cam_props)
    
    if viewer is None:
        print("创建观察者失败")
        return
    
    # 加载URDF资源
    asset_root = "resources/T1/"
    
    # 1. 加载足球场
    field_file = "soccer_field.urdf"
    field_asset_options = gymapi.AssetOptions()
    field_asset_options.fix_base_link = True
    field_asset = gym.load_asset(sim, asset_root, field_file, field_asset_options)
    
    # 2. 加载机器人
    robot_file = "T1_locomotion.urdf"
    robot_asset_options = gymapi.AssetOptions()
    robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    robot_asset_options.fix_base_link = False
    robot_asset_options.collapse_fixed_joints = True
    robot_asset_options.angular_damping = 0.0
    robot_asset = gym.load_asset(sim, asset_root, robot_file, robot_asset_options)
    
    # 3. 加载足球
    ball_file = "soccer_ball.urdf"
    ball_asset_options = gymapi.AssetOptions()
    ball_asset_options.fix_base_link = False
    ball_asset_options.angular_damping = 0.1
    ball_asset_options.linear_damping = 0.1
    ball_asset = gym.load_asset(sim, asset_root, ball_file, ball_asset_options)
    
    # 获取机器人的DOF数量
    robot_dof_count = gym.get_asset_dof_count(robot_asset)
    robot_dof_props = gym.get_asset_dof_properties(robot_asset)
    
    # 将所有DOF设置为位置控制模式
    for i in range(robot_dof_count):
        robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        robot_dof_props['stiffness'][i] = 100.0
        robot_dof_props['damping'][i] = 1.0
    
    # 创建环境
    num_envs = 1
    spacing = 10.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    
    envs = []
    robot_handles = []
    ball_handles = []
    field_handles = []
    
    for i in range(num_envs):
        # 创建环境
        env = gym.create_env(sim, lower, upper, 1)
        envs.append(env)
        
        # 放置足球场
        field_pose = gymapi.Transform()
        field_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        field_pose.r = gymapi.Quat(0, 0, 0, 1)
        field_handle = gym.create_actor(env, field_asset, field_pose, "field", i, 0)
        field_handles.append(field_handle)
        
        # 放置机器人
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.72)  # 确保机器人在地面上方
        robot_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", i, 0)
        robot_handles.append(robot_handle)
        
        # 设置机器人DOF属性
        gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)
        
        # 设置机器人初始姿态
        robot_dof_states = np.zeros(robot_dof_count, gymapi.DofState.dtype)
        for j in range(robot_dof_count):
            robot_dof_states['pos'][j] = 0.0
        gym.set_actor_dof_states(env, robot_handle, robot_dof_states, gymapi.STATE_ALL)
        
        # 放置足球
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(2.0, 0.0, 0.11)  # 球在机器人前方2m处
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_handle = gym.create_actor(env, ball_asset, ball_pose, "ball", i, 0)
        ball_handles.append(ball_handle)
    
    # 设置相机位置
    cam_pos = gymapi.Vec3(0.0, -8.0, 5.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 模拟循环
    while not gym.query_viewer_has_closed(viewer):
        # 步进模拟
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 更新观察者
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # 等待
        gym.sync_frame_time(sim)
    
    # 清理
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main() 
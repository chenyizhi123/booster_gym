import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from .base_task import BaseTask

from utils.utils import apply_randomization


class T1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ball_pose = None  # 初始化足球姿态变量
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    
        # 加载足球场URDF
        if self.cfg["env"].get("enable_soccer_field", False):
            soccer_field_options = gymapi.AssetOptions()
            soccer_field_options.fix_base_link = True
            soccer_field_options.disable_gravity = True
            soccer_field_asset = self.gym.load_asset(self.sim,asset_root, "soccer_field.urdf", soccer_field_options)
        else:
            soccer_field_asset = None
        # 加载足球URDF
        if self.cfg["env"].get("enable_soccer_ball", False):
            soccer_ball_options = gymapi.AssetOptions()
            soccer_ball_options.density = 100  # 控制球的密度
            soccer_ball_options.restitution = 0.8  # 控制球的弹性
            soccer_ball_options.angular_damping = 0.2
            soccer_ball_options.linear_damping = 0.2
            soccer_ball_asset = self.gym.load_asset(self.sim,asset_root, "soccer_ball.urdf", soccer_ball_options)
        else:
            soccer_ball_asset = None
        
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = [] # 存储球的handles
        self.field_handles = [] # 存储足球场的handles
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            
            # 创建足球场
            if soccer_field_asset is not None:
                field_pose = gymapi.Transform()
                field_pose.p = gymapi.Vec3(*pos)
                field_pose.p.z = -0.025  # 调整足球场的位置，使其在地面上
                field_handle = self.gym.create_actor(env_handle, soccer_field_asset, field_pose, "soccer_field", i, 0, 1)
                self.field_handles.append(field_handle)
            else:
                self.field_handles.append(None)
            
            # 创建足球
            if soccer_ball_asset is not None:
                ball_pose = gymapi.Transform()  # 为每个环境创建新的球姿态
                ball_pose.p = gymapi.Vec3(*pos)  # 初始位置与环境原点相同
                
                ball_position_mode = self.cfg["env"].get("ball_position_mode")

                if ball_position_mode == "random" and self.cfg["env"].get("ball_position_range") is not None:
                    # 随机位置逻辑
                    pos_range = self.cfg["env"]["ball_position_range"]
                    rand_x = np.random.uniform(pos_range[0][0], pos_range[0][1])
                    rand_y = np.random.uniform(pos_range[1][0], pos_range[1][1])
                    rand_z = np.random.uniform(pos_range[2][0], pos_range[2][1])
                    
                    ball_pose.p.x += rand_x
                    ball_pose.p.y += rand_y
                    ball_pose.p.z = rand_z
                elif self.cfg["env"].get("ball_initial_position") is not None:
                    # 固定位置逻辑
                    ball_init_pos = self.cfg["env"]["ball_initial_position"]
                    ball_pose.p.x += ball_init_pos[0]
                    ball_pose.p.y += ball_init_pos[1]
                    ball_pose.p.z = ball_init_pos[2]
                else:
                    # 默认位置
                    ball_pose.p.x += 1.0
                    ball_pose.p.z = 0.11
                
                ball_handle = self.gym.create_actor(env_handle, soccer_ball_asset, ball_pose, "soccer_ball", i, 0, 2)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.ball_handles.append(ball_handle)
        
        # 在创建完所有环境后更新has_ball标志
        self.has_ball = hasattr(self, 'ball_handles') and len(self.ball_handles) > 0 and any(handle is not None for handle in self.ball_handles)
        if self.has_ball:
            # 足球通常只有一个刚体
            self.num_bodies += 1
    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        if self.has_ball:
            # 足球相关缓冲区
            self.ball_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.ball_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
            self.ball_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.ball_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            
            # 添加球门相关缓冲区
            # 右侧球门中心位置（根据URDF文件）
            self.right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # 左侧球门中心位置（根据URDF文件）
            self.left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # 目标球门中心相对于机器人的方向向量
            self.goal_dir_relative = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # 机器人前进方向与球门方向的夹角
            self.ball_to_goal_angle = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
            # 球到目标球门中心的向量
            self.ball_to_goal_vec = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            
            # 如果有足球，确保观察维度正确
            # 注意：原本47维基础 + 可获取的观测(相对位置3维+相对速度3维+相对方向3维+夹角1维+球到目标3维) = 60维
            # 特权信息：原本14维 + 难以获取的观测(世界坐标球位置3维+球速度3维) = 20维
            if self.num_obs < 60:
                print(f"警告：观察空间维度可能不足，当前为{self.num_obs}，添加可获取的观测需要至少60维")
            if self.num_privileged_obs < 20:
                print(f"警告：特权观察空间维度可能不足，当前为{self.num_privileged_obs}，添加特权信息需要至少20维")

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # 我们需要确保张量形状匹配的辅助函数
        def ensure_tensor_match(tensor, expected_batch_size):
            """确保张量的批处理维度与预期相匹配"""
            if tensor.shape[0] != expected_batch_size:
                if tensor.shape[0] > expected_batch_size and self.has_ball:
                    return tensor[:expected_batch_size]
                elif tensor.shape[0] < expected_batch_size:
                    # 如果张量太小，这种情况不应该发生
                    print(f"警告: 张量形状{tensor.shape}小于预期批量大小{expected_batch_size}")
            return tensor
        
        # 将辅助函数设为类方法以便在其他地方使用
        self.ensure_tensor_match = ensure_tensor_match

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 检查张量形状
        print(f"actor_root_state shape: {gymtorch.wrap_tensor(actor_root_state).shape}")
        print(f"body_state shape: {gymtorch.wrap_tensor(body_state).shape}")
        
        # 计算实际刚体总数，包括足球（如果存在）
        total_bodies_per_env = self.num_bodies
        if self.has_ball:
            total_bodies_per_env += 1
            print(f"检测到足球对象，每个环境的总刚体数：{total_bodies_per_env} (包含机器人刚体数 {self.num_bodies} + 足球)")
        else:
            print(f"未检测到足球对象，每个环境的总刚体数：{total_bodies_per_env}")

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        # 处理足球情况下的root_states
        if self.has_ball and self.root_states.shape[0] > self.num_envs:
            print(f"处理root_states：形状从{self.root_states.shape}调整为前{self.num_envs}个元素")
            self.root_states_robots = self.root_states[:self.num_envs].clone()
        else:
            self.root_states_robots = self.root_states
            
        # 使用机器人状态更新基本属性
        self.base_pos = self.root_states_robots[:, 0:3]
        self.base_quat = self.root_states_robots[:, 3:7]
            
        # 使用修正的总刚体数重塑body_states tensor
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, total_bodies_per_env, 13)
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states_robots[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 确保所有向量操作使用机器人状态
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states_robots[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states_robots[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        
        # 重置足球位置
        if self.has_ball and env_ids.shape[0] > 0:
            # 准备足球状态重置
            ball_states = torch.zeros((len(env_ids), 13), device=self.device)
            
            # 对于每个需要重置的环境，设置足球位置
            for i, env_id in enumerate(env_ids):
                if self.ball_handles[env_id] is not None:
                    # 获取环境原点
                    env_origin = self.env_origins[env_id].clone()
                    
                    # 设置位置
                    ball_states[i, 0:3] = env_origin.clone()  # 先设置为环境原点
                    
                    # 根据配置模式调整位置
                    ball_position_mode = self.cfg["env"].get("ball_position_mode")
                    
                    if ball_position_mode == "random" and self.cfg["env"].get("ball_position_range") is not None:
                        # 随机位置逻辑
                        pos_range = self.cfg["env"]["ball_position_range"]
                        rand_x = np.random.uniform(pos_range[0][0], pos_range[0][1])
                        rand_y = np.random.uniform(pos_range[1][0], pos_range[1][1])
                        rand_z = np.random.uniform(pos_range[2][0], pos_range[2][1])
                        
                        ball_states[i, 0] += rand_x
                        ball_states[i, 1] += rand_y
                        ball_states[i, 2] = rand_z
                    elif self.cfg["env"].get("ball_initial_position") is not None:
                        # 固定位置逻辑
                        ball_init_pos = self.cfg["env"]["ball_initial_position"]
                        ball_states[i, 0] += ball_init_pos[0]
                        ball_states[i, 1] += ball_init_pos[1]
                        ball_states[i, 2] = ball_init_pos[2]
                    else:
                        # 默认位置
                        ball_states[i, 0] += 1.0
                        ball_states[i, 2] = 0.11
                    
                    # 重置旋转为默认值（无旋转）
                    ball_states[i, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device)
                    
                    # 重置速度为0
                    ball_states[i, 7:13] = 0
            
            # 将球的状态应用到模拟中
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            
            # 为每个环境ID获取对应的球句柄
            ball_indices = torch.zeros(len(env_ids), dtype=torch.int32, device=self.device)
            for i, env_id in enumerate(env_ids):
                if self.ball_handles[env_id] is not None:
                    # 确保只重置有效的球
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(ball_states[i:i+1]),
                        gymtorch.unwrap_tensor(torch.tensor([env_id], device=self.device, dtype=torch.int32)),
                        1
                    )

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.root_states[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 处理足球情况下的root_states
        if self.has_ball and self.root_states.shape[0] > self.num_envs:
            self.root_states_robots = self.root_states[:self.num_envs].clone()
        else:
            self.root_states_robots = self.root_states
            
        # 使用机器人状态更新基本属性
        self.base_pos[:] = self.root_states_robots[:, 0:3]
        self.base_quat[:] = self.root_states_robots[:, 3:7]
        
        # 使用正确的张量计算速度和重力方向
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robots[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states_robots[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:10] = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        
        # 足球相关终止条件
        if self.has_ball:
            # 进球终止条件
            goal_x = 6.05  # 右球门x坐标
            goal_y_range = 1.25  # 球门宽度的一半
            goal_z_range = (0.0, 1.0)  # 球门高度范围
            
            # 检查球是否在球门范围内
            in_x_range = self.ball_pos[:, 0] > goal_x - 0.2
            in_y_range = torch.abs(self.ball_pos[:, 1]) < goal_y_range
            in_z_range = (self.ball_pos[:, 2] > goal_z_range[0]) & (self.ball_pos[:, 2] < goal_z_range[1])
            
            # 判断球是否进球
            scored = in_x_range & in_y_range & in_z_range
            
            # 球出界终止条件
            field_x_limit = 6.05  # 场地x轴边界
            field_y_limit = 4.05  # 场地y轴边界
            ball_out_of_bounds = (torch.abs(self.ball_pos[:, 0]) > field_x_limit) | (torch.abs(self.ball_pos[:, 1]) > field_y_limit)
            
            # 更新reset_buf和统计信息
            self.reset_buf |= scored | ball_out_of_bounds
            
            # 记录成功进球的次数（用于计算成功率）
            if not hasattr(self, 'goal_success_counter'):
                self.goal_success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self.total_episodes_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            
            # 对于当前帧需要重置的环境，更新计数器
            need_reset = self.reset_buf.clone()
            self.goal_success_counter += (scored & need_reset).long()
            self.total_episodes_counter += need_reset.long()
            
            # 计算成功率并保存到extras中
            success_rate = self.goal_success_counter.float() / torch.clamp(self.total_episodes_counter.float(), min=1.0)
            if 'metrics' not in self.extras:
                self.extras['metrics'] = {}
            self.extras['metrics']['success_rate'] = success_rate.mean().item()
        
        # 原有的超时条件
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )
        
        # 基础观察（适用于所有情况）
        base_obs = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        
        # 准备特权观察（基础部分）
        privileged_base_obs = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        
        # 如果有足球，添加足球相关的观察
        if self.has_ball:
            # 更新足球状态
            for i in range(self.num_envs):
                if self.ball_handles[i] is not None:
                    ball_state = self.gym.get_actor_rigid_body_states(self.envs[i], self.ball_handles[i], gymapi.STATE_ALL)
                    self.ball_pos[i, 0] = ball_state['pose']['p'][0][0]
                    self.ball_pos[i, 1] = ball_state['pose']['p'][0][1]
                    self.ball_pos[i, 2] = ball_state['pose']['p'][0][2]
                    self.ball_rot[i, 0] = ball_state['pose']['r'][0][0]
                    self.ball_rot[i, 1] = ball_state['pose']['r'][0][1]
                    self.ball_rot[i, 2] = ball_state['pose']['r'][0][2]
                    self.ball_rot[i, 3] = ball_state['pose']['r'][0][3]
                    self.ball_vel[i, 0] = ball_state['vel']['linear'][0][0]
                    self.ball_vel[i, 1] = ball_state['vel']['linear'][0][1]
                    self.ball_vel[i, 2] = ball_state['vel']['linear'][0][2]
                    self.ball_ang_vel[i, 0] = ball_state['vel']['angular'][0][0]
                    self.ball_ang_vel[i, 1] = ball_state['vel']['angular'][0][1]
                    self.ball_ang_vel[i, 2] = ball_state['vel']['angular'][0][2]
                    
                    # 设定球门位置 - 根据URDF文件的实际位置
                    # 右侧球门中心位置（x轴正方向）
                    self.right_goal_pos[i, 0] = 6.05  # 场地边界x=6.05, 球门后墙x=6.05，球门中心大约x=5.7
                    self.right_goal_pos[i, 1] = 0.0  # 球场中央
                    self.right_goal_pos[i, 2] = 0.3  # 球门高度中点大约0.3
                    
                    # 左侧球门中心位置（x轴负方向）
                    self.left_goal_pos[i, 0] = -6.0  # 场地边界x=-6.05, 球门中心大约x=-5.7
                    self.left_goal_pos[i, 1] = 0.0
                    self.left_goal_pos[i, 2] = 0.3
            
            # 计算足球相对于机器人的位置和速度
            ball_relative_pos = self.ball_pos - self.base_pos
            # 将相对位置转换到机器人本地坐标系
            ball_local_pos = quat_rotate_inverse(self.base_quat, ball_relative_pos)
            self.ball_local_pos = ball_local_pos  # 保存为类变量
            # 计算足球相对于机器人的速度
            ball_relative_vel = self.ball_vel - self.root_states[:, 7:10]
            # 将相对速度转换到机器人本地坐标系
            ball_local_vel = quat_rotate_inverse(self.base_quat, ball_relative_vel)
            self.ball_local_vel = ball_local_vel  # 保存为类变量
            
            # 使用右侧球门作为默认目标（可以根据需要调整）
            target_goal_pos = self.right_goal_pos
            
            # 计算球门方向相关的观察
            # 1. 球门中心相对于机器人躯干的3D方向
            goal_relative_pos = target_goal_pos - self.base_pos
            self.goal_dir_relative = quat_rotate_inverse(self.base_quat, goal_relative_pos)
            self.goal_dir_relative = self.goal_dir_relative / (torch.norm(self.goal_dir_relative, dim=1, keepdim=True) + 1e-6)  # 单位向量
            
            # 2. 机器人前进方向与球门方向的夹角
            forward_dir = torch.zeros_like(self.base_pos)
            forward_dir[:, 0] = 1.0  # 假设机器人的前进方向是局部坐标系的x轴
            world_forward = quat_rotate(self.base_quat, forward_dir)
            goal_dir = target_goal_pos - self.base_pos
            goal_dir = goal_dir / (torch.norm(goal_dir, dim=1, keepdim=True) + 1e-6)
            # 计算夹角的余弦值，然后转换为角度
            cos_angle = torch.sum(world_forward * goal_dir, dim=1, keepdim=True)
            self.ball_to_goal_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
            
            # 3. 球到球门中心的向量
            self.ball_to_goal_vec = target_goal_pos - self.ball_pos
            
            # 将可通过传感器获取的足球和球门信息添加到观察中
            self.obs_buf = torch.cat(
                (
                    base_obs,
                    ball_local_pos,  # 足球在机器人坐标系下的相对位置（3维）- 可通过机器人视觉系统获取
                    ball_local_vel,  # 足球在机器人坐标系下的相对速度（3维）- 可通过连续观测估计
                    self.goal_dir_relative,  # 球门中心相对于机器人躯干的3D方向（3维）- 可通过预设信息获取
                    self.ball_to_goal_angle, # 机器人前进方向与球门方向的夹角（1维）- 可通过计算获取
                    self.ball_to_goal_vec,   # 球到球门中心的向量（3维）- 可通过RGB-D相机获取
                ),
                dim=-1,
            )
            
            # 将难以在现实环境中获取的信息放入特权观察中
            self.privileged_obs_buf = torch.cat(
                (
                    privileged_base_obs,  # 基础特权信息（14维）
                    self.ball_pos,        # 足球在世界坐标系下的位置（3维）
                    self.ball_vel,        # 足球在世界坐标系下的速度（3维）
                ),
                dim=-1,
            )
        else:
            # 如果没有足球，使用基础观察
            self.obs_buf = base_obs
            self.privileged_obs_buf = privileged_base_obs
        
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()
    def _reward_approach_ball(self):
    # """靠近球的奖励 - 鼓励机器人接近足球  
    # 在Phase-2中启用，权重较高"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 计算机器人到球的距离（使用局部坐标系中的位置）
        dist_to_ball = torch.norm(self.ball_local_pos, dim=1)
        # 使用高斯函数将距离转换为奖励，距离越近奖励越高
        approach_sigma = self.cfg["rewards"].get("approach_sigma", 0.5)  # 可在yaml中配置
        # 最大奖励为1.0，随距离增加呈指数衰减
        return torch.exp(-torch.square(dist_to_ball) / approach_sigma)
        
    def _reward_face_ball(self):
        """面向球的奖励 - 鼓励机器人正面朝向球
        在Phase-2中启用，与approach_ball配合使用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 机器人前向方向在局部坐标系中是(1,0,0)
        # ball_local_pos已经是球在机器人局部坐标系中的位置
        forward_vec = torch.zeros_like(self.ball_local_pos)
        forward_vec[:, 0] = 1.0  # x轴正方向是机器人的前向
        
        # 计算球的方向向量（需要先归一化）
        ball_dir = self.ball_local_pos.clone()
        ball_dist = torch.norm(ball_dir, dim=1, keepdim=True) + 1e-6
        ball_dir = ball_dir / ball_dist
        # 计算前向向量与球方向向量的点积(余弦值)
        # 完全朝向球时为1，垂直时为0，背对时为-1
        cos_angle = torch.sum(forward_vec * ball_dir, dim=1)
        
        # 将余弦值裁剪到[0,1]范围，只奖励正面朝向球
        return torch.clamp(cos_angle, min=0.0)
        
    def _reward_align_goal(self):
        """球门对准奖励 - 鼓励机器人让球、自己和球门在一条直线上
        在Phase-3中启用，为踢球做准备"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 计算两个关键向量
        # 1. 机器人到球的方向（机器人应该面对的方向）
        robot_to_ball_dir = self.ball_local_pos.clone()
        robot_to_ball_dist = torch.norm(robot_to_ball_dir, dim=1, keepdim=True) + 1e-6
        robot_to_ball_dir = robot_to_ball_dir / robot_to_ball_dist
        
        # 2. 球到球门的方向（应该踢球的方向）
        # 我们已经在观测空间中有ball_to_goal_vec
        ball_to_goal_dir = self.ball_to_goal_vec.clone()
        ball_to_goal_dist = torch.norm(ball_to_goal_dir, dim=1, keepdim=True) + 1e-6
        ball_to_goal_dir = ball_to_goal_dir / ball_to_goal_dist
        
        # 计算这两个向量的夹角余弦值
        # 球、机器人、球门完全对齐时，余弦值为1
        cos_angle = torch.sum(robot_to_ball_dir * ball_to_goal_dir, dim=1)
        
        # 我们希望机器人站在球后面，朝向球门
        # 只有当角度小于90度时（余弦值>0）才给予奖励
        return torch.clamp(cos_angle, min=0.0)
        
    def _reward_kick_velocity(self):
        """踢球速度奖励 - 鼓励球朝向球门方向移动
        在Phase-3和Phase-4中启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 计算球的速度在球门方向上的投影
        ball_vel_world = self.ball_vel
        
        # 球到球门的方向（单位向量）
        ball_to_goal_dir = self.ball_to_goal_vec.clone()
        ball_to_goal_dist = torch.norm(ball_to_goal_dir, dim=1, keepdim=True) + 1e-6
        ball_to_goal_dir = ball_to_goal_dir / ball_to_goal_dist
        
        # 计算球速在球门方向上的投影分量
        vel_proj = torch.sum(ball_vel_world * ball_to_goal_dir, dim=1)
        
        # 只有当球朝向球门移动时才给予奖励（速度投影为正）
        # 且奖励与速度成正比，但设置上限
        max_velocity = 5.0  # 可在yaml中配置
        return torch.clamp(vel_proj, min=0.0, max=max_velocity) / max_velocity
        
    def _reward_goal_scored(self):
        """进球奖励 - 当球进入球门时给予高额奖励
        在Phase-4中启用，是最终的训练目标"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 定义球门的位置范围
        goal_x = 6.05  # 右球门x坐标
        goal_y_range = 1.25  # 球门宽度的一半，从URDF得知
        goal_z_range = (0.0, 1.0)  # 球门高度范围
        
        # 检查球是否在球门范围内
        in_x_range = self.ball_pos[:, 0] > goal_x - 0.2  # 稍微宽松一点
        in_y_range = torch.abs(self.ball_pos[:, 1]) < goal_y_range
        in_z_range = (self.ball_pos[:, 2] > goal_z_range[0]) & (self.ball_pos[:, 2] < goal_z_range[1])
        
        # 判断球是否进球
        scored = in_x_range & in_y_range & in_z_range
        
        # 进球给予固定奖励1.0，未进球为0
        # 注意：实际训练时，可以在T1.yaml中设置很大的系数(如30)
        return scored.float()
        
    def _reward_dribbling(self):
        """带球奖励 - 鼓励机器人在控制球的同时移动
        可选奖励，在Phase-4中启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 定义控制范围 - 球在机器人前方适当距离内
        control_dist_min = 0.3
        control_dist_max = 0.8
        
        # 计算球到机器人的距离
        ball_dist = torch.norm(self.ball_local_pos, dim=1)
        
        # 球在控制范围内的mask
        in_control = (ball_dist > control_dist_min) & (ball_dist < control_dist_max)
        
        # 机器人和球都在移动的情况(使用x方向速度作为指标)
        robot_moving = torch.abs(self.base_lin_vel[:, 0]) > 0.3  # 机器人在移动
        ball_moving = torch.abs(self.ball_vel[:, 0]) > 0.2  # 球在移动
        
        # 带球移动的奖励
        dribbling = in_control & robot_moving & ball_moving
        
        return dribbling.float()
        
    def _reward_ball_position_z(self):
        """球高度惩罚 - 惩罚球过高，鼓励低平射门
        辅助奖励，根据需要启用"""
        if not self.has_ball:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        # 理想高度是球半径
        ball_radius = 0.2  # 从soccer_ball.urdf获得
        ideal_height = ball_radius
        
        # 计算球高度与理想高度的差距，并惩罚
        height_diff = torch.abs(self.ball_pos[:, 2] - ideal_height)
        
        # 转换为[0,1]范围的惩罚，差距越大惩罚越大
        height_sigma = 0.5  # 可在yaml中配置
        # 注意：这里返回的是惩罚，T1.yaml中应设为负权重
        return torch.exp(-torch.square(height_diff) / height_sigma)
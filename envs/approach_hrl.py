import os
import torch
import torch.nn as nn
import numpy as np
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
from .base_task import BaseTask
from utils.utils import apply_randomization, check_tensor

class approach_hrl(BaseTask):
    """HRL版本的approach任务"""

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 初始化HRL组件
        self._init_hrl_components()
        
        # 继续原有初始化流程
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _init_hrl_components(self):
        """初始化HRL组件"""
        self.hrl_cfg = self.cfg.get("hrl", {})
        self.hrl_enabled = self.hrl_cfg.get("enabled", True)
        # 高层策略由训练器创建，环境不需要创建
        # 决策控制参数
        self.decision_interval = self.hrl_cfg["decision_interval"]
        # 动作空间参数
        self.command_limits = torch.tensor(
            self.hrl_cfg["action_space"]["command_limits"], 
            device=self.device
        )
        self.delta_limits = torch.tensor(
            self.hrl_cfg["action_space"]["delta_limits"], 
            device=self.device
        )
        
        
        # 课程学习配置
        self.curriculum_cfg = self.hrl_cfg.get("curriculum", {})
        self.curriculum_enabled = self.curriculum_cfg.get("enabled", False)
        
        # 加载训练好的低层网络（t1.py的网络）
        self.trained_locomotion_policy = self._load_trained_locomotion_policy()
        if self.trained_locomotion_policy is not None:
            self.trained_locomotion_policy.eval()  # 设置为评估模式
            # 冻结低层网络参数
            for param in self.trained_locomotion_policy.parameters():
                param.requires_grad = False



    def _load_trained_locomotion_policy(self):
        """
        加载训练好的t1.py网络
        
        从HRL配置中读取t1模型路径并加载
        """
        try:
            import glob
            import os
            
            # 从HRL配置中获取t1模型路径
            t1_model_config = self.hrl_cfg.get("t1_model", {})
            model_path = t1_model_config.get("checkpoint_path", None)
            
            if model_path is None:
                print("⚠️  警告: 未指定t1模型路径，请在HRL配置中设置t1_model.checkpoint_path")
                return None
            
            # 如果路径是-1，自动寻找最新的t1模型
            if model_path == "-1":
                # 寻找logs目录下最新的t1模型
                t1_checkpoints = glob.glob(os.path.join("logs", "**/T1_*.pth"), recursive=True)
                if not t1_checkpoints:
                    # 如果没有找到T1_开头的，寻找包含T1的
                    t1_checkpoints = glob.glob(os.path.join("logs", "**/*T1*.pth"), recursive=True)
                
                if t1_checkpoints:
                    model_path = sorted(t1_checkpoints, key=os.path.getmtime)[-1]
                    print(f"📂 自动找到最新的t1模型: {model_path}")
                else:
                    print("❌ 未找到t1模型文件，请确保已训练t1模型")
                    return None
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                print(f"❌ t1模型文件不存在: {model_path}")
                return None
            
            print(f"📂 加载t1模型: {model_path}")
            
            # 加载模型检查点
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 创建ActorCritic网络（与t1.py相同的结构） 
            from utils.model import ActorCritic
            
            # t1.py的网络参数（从T1.yaml获取）
            t1_num_actions = 12  # 12个关节
            t1_num_obs = 47      # t1.py的观察维度
            t1_num_privileged_obs = 14  # 特权观察维度
            
            # 创建完整的ActorCritic网络
            full_model = ActorCritic(t1_num_actions, t1_num_obs, t1_num_privileged_obs).to(self.device)
            
            # 加载权重
            full_model.load_state_dict(checkpoint["model"])
            
            # 只返回actor部分（策略网络）
            locomotion_policy = full_model.actor
            
            print("✅ t1模型加载成功")
            return locomotion_policy
            
        except Exception as e:
            print(f"❌ 加载t1模型失败: {e}")
            print("请检查:")
            print("1. t1模型是否已训练完成")
            print("2. 模型路径是否正确")
            print("3. 模型文件是否完整")
            return None

    def _create_envs(self):
        """创建环境，与原approach.py保持一致"""
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

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        # 创建足球
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.1
        ball_options.linear_damping = 0.1
        ball_asset = self.gym.load_asset(self.sim, asset_root, "soccer_ball.urdf", ball_options)
        "=========下面是创建足球场地的部分=========" 
        field_options = gymapi.AssetOptions()
        field_options.disable_gravity = True
        field_options.fix_base_link = True
        field_asset = self.gym.load_asset(self.sim, asset_root, "soccer_field_half.urdf", field_options) 
        self.field_num_dofs = self.gym.get_asset_dof_count(field_asset)
        # DOF属性设置
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

        # 身体名称和接触索引
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        # 足部索引
        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        # 基础初始状态
        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + 
            self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []
        self.field_handles = [] 
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            position_range=[[-4,4],[-3,3]]  
            # 机器人位置
            robot_pose = gymapi.Transform()
            robot_pose.p.x = pos[0] + np.random.uniform(position_range[0][0], position_range[0][1])
            robot_pose.p.y = pos[1] + np.random.uniform(position_range[1][0], position_range[1][1])
            robot_pose.p.z = pos[2]
            
            # 场地位置
            field_pose = gymapi.Transform()
            field_pose.p.x = pos[0]
            field_pose.p.y = pos[1]
            field_pose.p.z = pos[2]
            
            # 球位置
            ball_pose = gymapi.Transform()
            ball_pose.p.x = robot_pose.p.x + 0.15
            ball_pose.p.y = robot_pose.p.y + 0.15
            ball_pose.p.z = robot_pose.p.z+0.11
            "=======================下面处理机器人======================="
            actor_handle = self.gym.create_actor(env_handle, robot_asset, robot_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            "=======================下面处理足球======================="
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 1, 0)
            field_handle = self.gym.create_actor(env_handle, field_asset, field_pose, "field", i, 1, 0)
            self.ball_handles.append(ball_handle)
            self.field_handles.append(field_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_rigid_body_props(self, props, i):
        """处理刚体属性"""
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
        """处理刚体形状属性"""
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        """获取环境原点"""
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _init_buffers(self):
        """初始化缓冲区，包含HRL特定的缓冲区"""
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        # 球相关观察
        self.ball_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_local_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_local_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_angular_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # 球门相关
        self.goal_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.goal_position[:, 0] = self.env_origins[:, 0]
        self.goal_position[:, 1] = self.env_origins[:, 1] + 4.5
        self.goal_position[:, 2] = self.env_origins[:, 2] 
        self.goal_dir_relative = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_to_goal_vec = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.heading_angle = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.goal_width=2.35
        self.goal_height=0.8
        # HRL特定缓冲区
        self.decision_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_delta_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_potential = torch.zeros(self.num_envs, device=self.device)
        
        # 低层网络需要的额外状态（来自t1.py）
        self.last_low_level_actions = torch.zeros(self.num_envs, 12, device=self.device)  # 12维关节动作
        self.gait_process = torch.zeros(self.num_envs, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)
        # 初始化步态频率为合理值
        self.gait_frequency[:] = 3.0  # 默认步态频率
        
        # 课程学习相关
        if self.curriculum_enabled:
            self.curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            window_size = self.curriculum_cfg["window_size"]
            self.success_history = torch.zeros(self.num_envs, window_size, dtype=torch.bool, device=self.device)
            self.history_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 标准缓冲区
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # 获取gym状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 机器人和球的索引
        self.num_actors = 3
        self.robot_indices = torch.arange(0, self.num_actors * self.num_envs, self.num_actors, device=self.device)
        self.soccer_indices = torch.arange(1, self.num_actors * self.num_envs, self.num_actors, device=self.device)
        self.goal_indices = torch.arange(2, self.num_actors * self.num_envs, self.num_actors, device=self.device)

        # 创建状态张量包装器
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # ===== 分离机器人和球场的DOF状态 =====
        total_dofs_per_env = self.num_dofs + self.field_num_dofs
        # 重新整形DOF状态张量：[num_envs, total_dofs_per_env, 2]
        dof_state_reshaped = self.dof_state.view(self.num_envs, total_dofs_per_env, 2)
        # 提取机器人的DOF状态（前num_dofs个）
        robot_dof_state = dof_state_reshaped[:, :self.num_dofs, :]
        self.dof_pos = robot_dof_state[..., 0]  # 机器人关节位置
        self.dof_vel = robot_dof_state[..., 1]  # 机器人关节速度
        
        # 针对机器人和足球的特定处理，注意这个变量只读，只能由gym更新
        self.robot_root_states=self.root_states[self.robot_indices]
        self.soccer_root_states=self.root_states[self.soccer_indices]
        #============================cyz changed it================================
        self.all_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.contact_forces=self.all_forces[:,:self.num_bodies,:]
        self.all_body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, -1, 13)
        self.body_states=self.all_body_states[:,:self.num_bodies,:]
        self.soccer_body_states = self.all_body_states[:, self.num_bodies, :]
        self.base_pos = self.robot_root_states[:, 0:3]
        self.base_quat = self.robot_root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # 初始化其他变量
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        
        # 机器人运动状态
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 低层网络需要的其他状态
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 任务成功标志
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # self.success_distance_threshold = self.cfg['rewards']['success_distance_threshold']
        # self.success_angle_threshold = self.cfg['rewards']['success_angle_threshold']

        # 默认关节位置
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
        """准备奖励函数"""
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def step(self, actions):
        """
        HRL版本的step函数
        
        频率控制：
        - 高层策略：5Hz (每10帧更新一次)
        - 低层控制：50Hz (每帧执行)
        
        Args:
            actions: 高层策略的3维输出 [ΔVx, ΔVy, ΔW_ang]
        """
        
        # 1. 高层决策更新 (5Hz - 每10帧更新)
        self._update_high_level_commands(actions)
        
        # 2. 低层执行 (50Hz - 每帧执行)
        self._execute_low_level_control()
        
        # 3. 仿真步进
        self._simulate_physics()
        
        # 4. 状态更新
        self._post_physics_step()
        
        
        # 5. 奖励计算
        self._compute_reward()
        
        # 6. 检查终止和重置
        self._check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
        
        # 7. 更新观察
        self._update_ball_observations()
        self._compute_observations()
        
        # 8. 更新历史状态（与t1.py保持一致）
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.robot_root_states[:, 7:13]
        
        # 9. 更新步数计数器
        self.common_step_counter += 1
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_high_level_commands(self, delta_commands):
        """
        更新高层命令 (5Hz频率)
        
        频率控制逻辑：
        - 每10帧更新一次高层策略 (50Hz/5Hz = 10)
        - 其他时间保持上一次的指令不变
        """
        # 增加决策计数器
        self.decision_counter += 1
        
        # 检查是否需要更新高层指令 (每10帧)
        update_mask = (self.decision_counter % self.decision_interval == 0)
        
        if update_mask.any():
            env_ids = torch.where(update_mask)[0]
            
            # 限制增量范围
            limited_delta = torch.clamp(delta_commands[env_ids], -self.delta_limits, self.delta_limits)
            
            # 应用增量更新
            self.current_commands[env_ids] = self.last_commands[env_ids] + limited_delta
            
            # 限制绝对速度范围
            self.current_commands[env_ids] = torch.clamp(
                self.current_commands[env_ids], -self.command_limits, self.command_limits
            )
            
            # 更新历史
            self.last_commands[env_ids] = self.current_commands[env_ids]
            self.last_delta_commands[env_ids] = limited_delta
        
        # 将高层命令设置为低层的速度指令 (每帧都执行)
        self.commands[:, :3] = self.current_commands

    def _execute_low_level_control(self):
        """执行低层控制 - 使用训练好的t1.py网络"""
        
        # 1. 构建t1.py需要的47维观察 (包含速度指令)
        # 注意：self.commands已经在_update_high_level_commands中更新了
        low_level_obs = self._build_low_level_observations()
        
        # 2. 使用训练好的t1.py网络推理关节动作
        if self.trained_locomotion_policy is not None:
            with torch.no_grad():  # 冻结低层网络参数
                joint_actions = self.trained_locomotion_policy(low_level_obs)
        else:
            # 如果没有加载低层网络，使用零动作
            joint_actions = torch.zeros(self.num_envs, 12, device=self.device)
        
        # 3. 限制动作范围 (与t1.py保持一致)
        joint_actions = torch.clip(joint_actions, 
                                  -self.cfg["normalization"]["clip_actions"], 
                                   self.cfg["normalization"]["clip_actions"])
        
        # 4. 执行关节控制
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * joint_actions
        
        # 5. 执行控制循环（与t1.py相同）
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.minimum(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        
        # 6. 更新历史动作
        self.last_low_level_actions[:] = joint_actions

    def _build_low_level_observations(self):
        """构建低层网络（t1.py）需要的47维观察"""
        # 根据t1.py的观察构建方式
        commands_scale = torch.tensor([
            self.cfg["normalization"]["lin_vel"], 
            self.cfg["normalization"]["lin_vel"], 
            self.cfg["normalization"]["ang_vel"]
        ], device=self.device)
        # 构建47维观察（与t1.py保持一致）
        low_level_obs = torch.cat([
            apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
            apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
            self.commands[:, :3] * commands_scale,
            (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
            (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
            apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
            apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
            self.last_low_level_actions,
        ], dim=-1)

        return low_level_obs

    def _simulate_physics(self):
        """仿真物理步进"""
        self.render()

    def _post_physics_step(self):
        """物理步进后的状态更新"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 更新机器人和球的状态
        self.robot_root_states = self.root_states[self.robot_indices]
        self.soccer_root_states = self.root_states[self.soccer_indices]
        self.base_pos[:] = self.robot_root_states[:, 0:3]
        self.base_quat[:] = self.robot_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 更新低层网络需要的状态
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        
        # 更新步态过程
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        
        self.episode_length_buf += 1

    def _compute_reward(self):
        """Compute rewards - 使用标准奖励框架
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

    # ------------ 大道至简的奖励函数 ----------------
    def _reward_approach_ball(self):
        """简单奖励：靠近球 - 优化远距离动力"""
        ball_distance = torch.norm(self.ball_local_position[:, :2], dim=1)
        return torch.exp(-ball_distance * 0.5)

    def _reward_behind_ball(self):
        """机器人在球的后方（相对于球门）- 只给正奖励"""
        ball_distance = torch.norm(self.ball_local_position[:, :2], dim=1)
        
        # 扩大激活范围，让机器人更早考虑位置
        close_mask = ball_distance <= 1.2
        if not close_mask.any():
            return torch.zeros(self.num_envs, device=self.device)
        
        # 使用世界坐标系计算，避免坐标系混乱
        # 机器人位置
        robot_pos = self.base_pos[:, :2]  # 世界坐标系中的机器人位置
        ball_pos = self.ball_position[:, :2]  # 世界坐标系中的球位置
        goal_pos = self.goal_position[:, :2]  # 世界坐标系中的球门位置
        
        # 球到球门的方向
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_norm = torch.norm(ball_to_goal, dim=1, keepdim=True) + 1e-8
        ball_to_goal_unit = ball_to_goal / ball_to_goal_norm
        
        # 机器人到球的方向
        robot_to_ball = ball_pos - robot_pos
        robot_to_ball_norm = torch.norm(robot_to_ball, dim=1, keepdim=True) + 1e-8
        robot_to_ball_unit = robot_to_ball / robot_to_ball_norm
        
        # 计算对齐度：机器人到球的方向应该与球到球门的方向相同
        alignment = torch.sum(robot_to_ball_unit * ball_to_goal_unit, dim=1)
        # 新增：机器人朝向检查
        robot_forward = torch.tensor([1.0, 0.0], device=self.device)
        goal_direction = self.goal_dir_relative[:, :2]  # 机器人局部坐标系中的球门方向
        goal_direction_norm = torch.norm(goal_direction, dim=1, keepdim=True) + 1e-8
        goal_direction_unit = goal_direction / goal_direction_norm
        # 机器人是否面向球门
        facing_alignment = torch.sum(robot_forward * goal_direction_unit, dim=1)
        # 只给正奖励，避免负奖励陷阱
        positive_alignment = torch.clamp(alignment + facing_alignment, 0.0, 2.0)

        return positive_alignment * close_mask.float()

    def _reward_kick_to_goal(self):
        """简单粗暴：球朝球门移动就给奖励"""
        # 球到球门的向量
        ball_to_goal_vec = self.ball_to_goal_vec[:, :2]
        ball_to_goal_distance = torch.norm(ball_to_goal_vec, dim=1, keepdim=True)
        ball_to_goal_unit = ball_to_goal_vec / (ball_to_goal_distance + 1e-8)
        
        # 球的真实世界速度（不是相对速度）
        ball_velocity = self.ball_velocity[:, :2].clone()
        
        # 球朝球门移动的速度分量
        ball_goal_velocity = torch.sum(ball_velocity * ball_to_goal_unit, dim=1)
        
        # 只奖励朝球门的速度，不奖励远离球门的速度
        return torch.clamp(ball_goal_velocity * 3.0, 0.0, 100.0)

    def _reward_success(self):
        """任务成功奖励"""
        return self.task_success.float() * 100.0

    def _reward_time_penalty(self):
        """时间惩罚 - 每一帧都扣分，鼓励尽快完成任务"""
        return -torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_final_sprint(self):
        """爆发时刻触发：检测关键时机全力冲刺"""
        ball_dist = torch.norm(self.ball_to_goal_vec[:, :2], dim=1)
        robot_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        robot_ball_dist = torch.norm(self.ball_local_position[:, :2], dim=1)
        
        # 爆发时机判断
        explosive_moment = (
            (ball_dist < 3.0) &                    # 球接近球门
            (robot_ball_dist < 0.8) &              # 机器人接近球
            (robot_speed > 0.5)                    # 机器人已在移动
        )
        
        # 爆发状态下的机器人冲刺奖励
        sprint_power = torch.square(robot_speed) * 3.0  # 平方增长，鼓励高速
        
        # 持续冲刺奖励（防止急停）
        momentum_bonus = torch.tanh(robot_speed * 2.0) * 2.0
        
        # 只在爆发时刻给予奖励
        final_reward = explosive_moment.float() * (sprint_power + momentum_bonus)
        
        return final_reward 

        # ------------ 分阶段奖励系统 ----------------
    def _reward_approach_and_align(self):
        """智能接近和对准 - 分阶段处理"""
        ball_distance = torch.norm(self.ball_local_position[:, :2], dim=1)
        ball_to_goal_distance = torch.norm(self.ball_to_goal_vec[:, :2], dim=1)
        
        # 排除射门阶段
        shoot_stage = (ball_distance <= 0.3) & (ball_to_goal_distance <= 1.0)
        stage_mask = ~shoot_stage
        
        if not stage_mask.any():
            return torch.zeros(self.num_envs, device=self.device)
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 阶段1: 距离超过0.5米 - 只考虑向球走
        far_mask = (ball_distance > 0.5) & stage_mask
        if far_mask.any():
            # 简单的距离奖励
            distance_reward = torch.exp(-ball_distance * 0.8)
            reward += distance_reward * far_mask.float()
        
        # 阶段2: 0.5米到0.3米 - 边转向边校准位置
        mid_mask = (ball_distance <= 0.5) & (ball_distance > 0.3) & stage_mask
        if mid_mask.any():
            robot_forward = torch.tensor([1.0, 0.0], device=self.device).expand(self.num_envs, -1)
            
            # 球到球门方向（局部坐标系）
            ball_to_goal_world = self.ball_to_goal_vec[:, :2]
            ball_to_goal_local = quat_rotate_inverse(
                self.base_quat, 
                torch.cat([ball_to_goal_world, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
            )[:, :2]
            ball_to_goal_norm = torch.norm(ball_to_goal_local, dim=1, keepdim=True) + 1e-8
            ball_to_goal_unit = ball_to_goal_local / ball_to_goal_norm
            
            # 机器人朝向球门
            orientation_reward = torch.clamp(torch.sum(robot_forward * ball_to_goal_unit, dim=1), 0.0, 1.0)
            
            # 机器人在球后方
            robot_to_ball = self.ball_local_position[:, :2]
            robot_to_ball_norm = torch.norm(robot_to_ball, dim=1, keepdim=True) + 1e-8
            robot_to_ball_unit = robot_to_ball / robot_to_ball_norm
            
            behind_ball_reward = torch.clamp(torch.sum(robot_to_ball_unit * ball_to_goal_unit, dim=1), 0.0, 1.0)
            
            # 组合奖励：距离 + 对准
            distance_reward = torch.exp(-ball_distance * 1.0)
            alignment_reward = orientation_reward * 0.5 + behind_ball_reward * 0.7
            
            total_mid_reward = distance_reward * 0.5 + alignment_reward * 0.5
            reward += total_mid_reward * mid_mask.float()
        
        return reward

    def _reward_dribble(self):
        """推球奖励 - 小于0.3米开始把球往球门带"""
        ball_distance = torch.norm(self.ball_local_position[:, :2], dim=1)
        ball_to_goal_distance = torch.norm(self.ball_to_goal_vec[:, :2], dim=1)
        
        # 高速推球阶段条件
        high_speed_stage = (ball_distance <= 0.3) & (ball_to_goal_distance <= 1.0)
        
        # 普通推球：小于0.3米但球离球门远
        normal_dribble_mask = (ball_distance <= 0.3) & (ball_to_goal_distance > 1.0)
        
        if not (normal_dribble_mask.any() or high_speed_stage.any()):
            return torch.zeros(self.num_envs, device=self.device)
        
        # 球的真实世界速度朝球门移动
        ball_to_goal_vec = self.ball_to_goal_vec[:, :2]
        ball_to_goal_distance_vec = torch.norm(ball_to_goal_vec, dim=1, keepdim=True)
        ball_to_goal_unit = ball_to_goal_vec / (ball_to_goal_distance_vec + 1e-8)
        
        ball_velocity = self.ball_velocity[:, :2]
        ball_goal_velocity = torch.sum(ball_velocity * ball_to_goal_unit, dim=1)
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 普通推球奖励
        if normal_dribble_mask.any():
            normal_reward = torch.clamp(ball_goal_velocity * 3.0, 0.0, 10.0)
            reward += normal_reward * normal_dribble_mask.float()
        
        return reward

    def _reward_high_speed_push(self):
        """高速推球奖励 - 小于0.3米且球离球门小于1米时激活"""
        ball_distance = torch.norm(self.ball_local_position[:, :2], dim=1)
        ball_to_goal_distance = torch.norm(self.ball_to_goal_vec[:, :2], dim=1)
        
        # 高速推球阶段条件
        high_speed_stage = (ball_distance <= 0.3) & (ball_to_goal_distance <= 1.0)
        
        if not high_speed_stage.any():
            return torch.zeros(self.num_envs, device=self.device)
        
        # 球的真实世界速度
        ball_velocity = self.ball_velocity[:, :2]
        ball_speed = torch.norm(ball_velocity, dim=1)
        
        # 球门方向
        ball_to_goal_vec = self.ball_to_goal_vec[:, :2]
        ball_to_goal_distance_vec = torch.norm(ball_to_goal_vec, dim=1, keepdim=True)
        ball_to_goal_unit = ball_to_goal_vec / (ball_to_goal_distance_vec + 1e-8)
        
        # 球朝球门的速度分量
        ball_goal_velocity = torch.sum(ball_velocity * ball_to_goal_unit, dim=1)
        
        # 机器人自身速度奖励（鼓励快速推球）
        robot_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        robot_speed_reward = torch.clamp(robot_speed * 2.0, 0.0, 5.0)
        
        # 球高速奖励（更高的系数）
        ball_speed_reward = torch.clamp(ball_speed * 4.0, 0.0, 15.0)
        
        # 方向奖励
        ball_vel_norm = torch.norm(ball_velocity, dim=1, keepdim=True) + 1e-8
        ball_vel_unit = ball_velocity / ball_vel_norm
        direction_reward = torch.clamp(torch.sum(ball_vel_unit * ball_to_goal_unit, dim=1), 0.0, 1.0) * 5.0
        
        # 组合奖励
        total_reward = robot_speed_reward + ball_speed_reward + direction_reward
        
        return total_reward * high_speed_stage.float()

    def _check_termination(self):
        """检查终止条件"""
        # 基础终止条件
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf = self.root_states[self.robot_indices, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] < self.cfg["rewards"]["terminate_height"]
        
        # 任务成功检查 - 球进门才算成功
        ball_world_pos = self.soccer_root_states[:, 0:3]
        ball_relative_pos = ball_world_pos - self.env_origins
        
        # 进球条件：
        # 1. 球越过球门线 (y >= goal_line_y)
        # 2. 球在球门宽度内 (x在[-goal_width_half, goal_width_half]范围内)
        # 3. 球在球门高度内 (z在[goal_height_min, goal_height_max]范围内)
        goal_line_y = self.cfg["rewards"]["goal_line_y"]
        goal_width_half = self.cfg["rewards"]["goal_width_half"]
        goal_height_min = self.cfg["rewards"]["goal_height_min"]
        goal_height_max = self.cfg["rewards"]["goal_height_max"]
        
        goal_line_crossed = ball_relative_pos[:, 1] >= goal_line_y
        within_goal_width = (ball_relative_pos[:, 0] >= -goal_width_half) & (ball_relative_pos[:, 0] <= goal_width_half)
        within_goal_height = (ball_relative_pos[:, 2] >= goal_height_min) & (ball_relative_pos[:, 2] <= goal_height_max)
        
        success_condition = goal_line_crossed & within_goal_width & within_goal_height
        self.task_success[:] = success_condition
        
        # 成功时给予奖励并重置 (通过标准奖励系统处理)
        # success_reward 将通过 _reward_success() 函数和标准奖励框架处理
        self.reset_buf |= success_condition
        "=========下面是球出界检查的部分=========" 
        # 球出界检查
        ball_world_pos = self.soccer_root_states[:, 0:3]
        ball_relative_pos = ball_world_pos - self.env_origins
        ball_out_of_bounds = (
            (ball_relative_pos[:, 0] < -6.0) | (ball_relative_pos[:, 0] > 6.0) |
            (ball_relative_pos[:, 1] < -4.5) | (ball_relative_pos[:, 1] > 4.5)
        )
        self.reset_buf |= ball_out_of_bounds
        
        # 超时检查
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf

    def _compute_observations(self):
        """计算HRL观察"""
        # 高层策略观察 (14维) - 仅包含真实部署可获取的信息
        # 机器人水平面运动状态 (3维): [Vx, Vy, Wz]
        robot_planar_velocity = torch.cat([
            self.base_lin_vel[:, :2],        # 水平线速度 (Vx, Vy) - shape: [num_envs, 2]
            self.base_ang_vel[:, 2:3]        # 绕Z轴角速度 (Wz) - shape: [num_envs, 1]
        ], dim=-1)  # 3维
        
        # 先对位置施加噪声，然后计算距离
        ball_position_noisy = check_tensor(apply_randomization(self.ball_local_position, self.cfg["noise"].get("vision_position")), device=self.device)
        goal_dir_noisy = check_tensor(apply_randomization(self.goal_dir_relative, self.cfg["noise"].get("vision_direction")), device=self.device)
        
        # 基于带噪声的位置计算距离
        ball_distance = torch.norm(ball_position_noisy[:, :2], dim=1, keepdim=True)  # 1维 - 基于噪声位置计算
        goal_distance = torch.norm(goal_dir_noisy[:, :2], dim=1, keepdim=True)  # 1维 - 基于噪声方向计算
        
        self.obs_buf = torch.cat([
            ball_position_noisy,           # 3维 - 通过视觉可获取（带噪声）
            goal_dir_noisy,                # 3维 - 通过定位系统可获取（带噪声）
            # check_tensor(apply_randomization(robot_planar_velocity, self.cfg["noise"].get("imu_velocity")), device=self.device),         # 3维 - 通过IMU可获取 (Vx, Vy, Wz) - 有噪声
            # check_tensor(self.last_commands),
            ball_distance,                 # 1维 - 基于噪声位置计算的距离
            goal_distance,                 # 1维 - 基于噪声方向计算的距离
            self.last_commands             # 3维 - 内部状态 (Vx_cmd, Vy_cmd, Wz_cmd)
        ], dim=-1)
        
        # 特权观察（包含真实部署难以获取的信息，用于更好的critic训练）
        self.privileged_obs_buf = torch.cat([
            self.obs_buf,                  # 14维 - 基础观察
            self.ball_local_velocity,      # 3维 - 球速度（真实中难以精确获取）
            self.ball_velocity,            # 3维 - 球全局速度
        ], dim=-1)
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def _update_ball_observations(self):
        """更新足球相关的观察变量"""
        # 更新足球状态
        self.ball_position[:] = self.soccer_root_states[:, 0:3]
        self.ball_velocity[:] = self.soccer_root_states[:, 7:10]
        self.ball_angular_velocity[:] = self.soccer_root_states[:, 10:13]
        
        # 计算球在机器人局部坐标系中的位置
        ball_relative_world = self.ball_position - self.base_pos
        self.ball_local_position[:] = quat_rotate_inverse(self.base_quat, ball_relative_world)
        
        # 计算球相对于机器人的速度在局部坐标系中
        ball_relative_vel_world = self.ball_velocity - self.robot_root_states[:, 7:10]
        self.ball_local_velocity[:] = quat_rotate_inverse(self.base_quat, ball_relative_vel_world)
        
        # 计算球门方向
        goal_direction_world = self.goal_position - self.base_pos
        self.goal_dir_relative[:] = quat_rotate_inverse(self.base_quat, goal_direction_world)
        
        # 计算球到球门的向量
        self.ball_to_goal_vec[:] = self.goal_position - self.ball_position
        
        # 计算机器人前进方向与球门方向的夹角
        robot_forward_world = quat_rotate(self.base_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        goal_dir_world = goal_direction_world / (torch.norm(goal_direction_world, dim=1, keepdim=True) + 1e-8)
        cos_angle = torch.sum(robot_forward_world * goal_dir_world, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        self.heading_angle[:] = torch.acos(cos_angle).unsqueeze(1)

    def reset(self):
        """重置所有环境"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.commands[:] = 0.0  # HRL模式下初始化为零速度指令
        self._update_ball_observations()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        """重置指定环境"""
        if len(env_ids) == 0:
            return

        # 更新课程学习
        if self.curriculum_enabled:
            self._update_curriculum(env_ids)

        # 重置DOF和根状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # 重置HRL相关状态
        if self.hrl_enabled:
            self.current_commands[env_ids] = 0
            self.last_commands[env_ids] = 0
            self.last_delta_commands[env_ids] = 0
            self.decision_counter[env_ids] = 0
            self.last_potential[env_ids] = 0
            self.last_low_level_actions[env_ids] = 0
            self.gait_process[env_ids] = 0
            self.filtered_lin_vel[env_ids] = 0
            self.filtered_ang_vel[env_ids] = 0

        # 重置其他状态
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.robot_root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        robot_indices = self.robot_indices[env_ids_int32].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )

    def _reset_root_states(self, env_ids):
        """重置根状态，参考逻辑但保持随机初始化"""
        # 重置机器人状态
        robot_actor_indices = self.robot_indices[env_ids]
        self.root_states[robot_actor_indices] = self.base_init_state
        self.root_states[robot_actor_indices, :2] += self.env_origins[env_ids, :2]
        self.root_states[robot_actor_indices, :2] = apply_randomization(
            self.root_states[robot_actor_indices, :2], self.cfg["randomization"].get("init_base_pos_xy")
        )
        
        # 地形高度处理（兼容平面和复杂地形）
        if hasattr(self, 'terrain') and self.terrain is not None:
            self.root_states[robot_actor_indices, 2] += self.terrain.terrain_heights(self.root_states[robot_actor_indices, :2])
        else:
            self.root_states[robot_actor_indices, 2] += 0.0  # 平面地形
            
        # 设置初始线速度
        self.root_states[robot_actor_indices, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        
        # 随机设置机器人朝向（不面向球或球门，通过奖励学习转向）
        self.root_states[robot_actor_indices, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )

        # 重置球状态（基于课程学习）
        soccer_actor_indices = self.soccer_indices[env_ids]
        if self.curriculum_enabled:
            self._reset_ball_with_curriculum(env_ids, soccer_actor_indices)
        else:
            self._reset_ball_random(env_ids, soccer_actor_indices)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _reset_ball_with_curriculum(self, env_ids, soccer_actor_indices):
        """基于课程学习重置球位置"""
        for i, env_id in enumerate(env_ids):
            level = self.curriculum_level[env_id].item() if hasattr(self, 'curriculum_level') else 0
            level_config = self.curriculum_cfg["levels"].get(level, self.curriculum_cfg["levels"][0])
            
            # 根据课程等级设置球的初始位置
            ball_dist_range = level_config["ball_distance_range"]
            ball_angle_range = level_config["ball_angle_range"]
            
            # 随机距离和角度
            distance = torch.rand(1, device=self.device) * (ball_dist_range[1] - ball_dist_range[0]) + ball_dist_range[0]
            angle = torch.rand(1, device=self.device) * (ball_angle_range[1] - ball_angle_range[0]) + ball_angle_range[0]
            
            # 计算球的位置
            ball_x = distance * torch.cos(angle)
            ball_y = distance * torch.sin(angle)
            
            # 限制在场地范围内
            ball_x = torch.clamp(ball_x, -6.0, 6.0)
            ball_y = torch.clamp(ball_y, -4.5, 4.5)
            
            # 设置球位置
            self.root_states[soccer_actor_indices[i], 0] = self.env_origins[env_id, 0] + ball_x
            self.root_states[soccer_actor_indices[i], 1] = self.env_origins[env_id, 1] + ball_y
            self.root_states[soccer_actor_indices[i], 2] = self.env_origins[env_id, 2] + 0.11

        # 重置球的速度和旋转
        self.root_states[soccer_actor_indices, 3:7] = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.root_states[soccer_actor_indices, 7:13] = 0.0

    def _reset_ball_random(self, env_ids, soccer_actor_indices):
        """随机重置球位置"""
        for i, env_id in enumerate(env_ids):
            ball_x = torch.rand(1, device=self.device) * 12.0 - 6.0  # [-5, 5] 缩小X范围
            ball_y = torch.rand(1, device=self.device) * 8.0 - 4.0   # [-4, 4] 更多在前方
            
            self.root_states[soccer_actor_indices[i], 0] = self.env_origins[env_id, 0] + ball_x
            self.root_states[soccer_actor_indices[i], 1] = self.env_origins[env_id, 1] + ball_y
            self.root_states[soccer_actor_indices[i], 2] = self.env_origins[env_id, 2] + 0.11

        # 重置球的速度和旋转
        self.root_states[soccer_actor_indices, 3:7] = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.root_states[soccer_actor_indices, 7:13] = 0.0

    def _update_curriculum(self, env_ids):
        """更新课程学习进度"""
        if not self.curriculum_enabled:
            return

        # 记录成功/失败
        success_mask = self.task_success[env_ids]

        # 更新成功历史
        for i, env_id in enumerate(env_ids):
            ptr = self.history_ptr[env_id]
            self.success_history[env_id, ptr] = success_mask[i]
            self.history_ptr[env_id] = (ptr + 1) % self.curriculum_cfg["window_size"]

        # 检查晋级条件
        for env_id in env_ids:
            current_level = self.curriculum_level[env_id]
            max_level = len(self.curriculum_cfg["levels"]) - 1

            if current_level < max_level:
                success_rate = self.success_history[env_id].float().mean()
                threshold = self.curriculum_cfg["success_threshold"]

                if success_rate >= threshold:
                    self.curriculum_level[env_id] += 1
                    # 重置成功历史
                    self.success_history[env_id].fill_(False)
                    self.history_ptr[env_id] = 0

        # 重置任务成功标志
        self.task_success[env_ids] = False


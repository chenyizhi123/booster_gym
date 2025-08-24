#!/usr/bin/env python3
"""
T1机器人足球模型演示脚本
仿照legged_gym的标准实现，简化版本
"""

import os
import yaml
import argparse
import numpy as np
import torch
import glob

# Isaac Gym必须在其他模块之前导入
import isaacgym

from envs.kick_2D import kick_2D
from rsl_rl.modules.actor_critic import ActorCritic

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='T1机器人足球演示')
    parser.add_argument("--task", type=str, default="kick_2D", help="任务名称")
    parser.add_argument("--load_run", type=str, default="-1", help="模型路径，-1表示最新模型")
    parser.add_argument("--checkpoint", type=str, default="-1", help="检查点编号，-1表示最新")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    
    return parser.parse_args()

def play(args):
    """主演示函数"""
    
    # 加载配置
    env_cfg_path = os.path.join("envs", f"{args.task}.yaml")
    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    
    # 演示环境配置调整
    env_cfg['env']['num_envs'] = min(args.num_envs, 50)  # 限制演示环境数量
    env_cfg['basic']['headless'] = args.headless
    
    # 禁用一些训练时的随机化设置
    if 'randomization' in env_cfg:
        for key in env_cfg['randomization'].keys():
            if key.startswith('init_') or key in ['push_robots', 'kick_interval_s']:
                env_cfg['randomization'][key] = {'range': [0., 0.], 'operation': 'additive', 'distribution': 'uniform'}
    
    # 禁用域随机化
    if 'noise' in env_cfg:
        for key in env_cfg['noise'].keys():
            env_cfg['noise'][key]['range'] = [0., 0.]
    
    # 禁用AMP初始化（演示时不需要）
    if 'env' in env_cfg and 'reference_state_initialization' in env_cfg['env']:
        env_cfg['env']['reference_state_initialization'] = False
    
    print(f"创建环境: {args.task}")
    env = kick_2D(env_cfg)
    obs_buf, privileged_obs_buf = env.reset()
    
    # 查找并加载模型
    checkpoint_path = find_model_path(args)
    print(f"加载模型: {checkpoint_path}")
    
    # 创建策略网络
    policy_cfg = env_cfg.get('policy', {})
    policy = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=policy_cfg.get('actor_hidden_dims', [512, 256, 128]),
        critic_hidden_dims=policy_cfg.get('critic_hidden_dims', [512, 256, 128]),
        activation=policy_cfg.get('activation', 'elu'),
        init_noise_std=policy_cfg.get('init_noise_std', 1.0),
    ).to(env.device)
    
    # 加载模型参数
    load_model(policy, checkpoint_path, env.device)
    policy.eval()
    
    # 演示参数
    robot_index = 0  # 用于记录的机器人索引
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = torch.zeros(env.num_envs, device=env.device)
    current_episode_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)
    
    print("开始演示...")
    print("按 Ctrl+C 停止")
    print("-" * 50)
    
    try:
        for i in range(10 * int(env.max_episode_length)):
            # 策略推理
            with torch.no_grad():
                actions = policy.act_inference(obs_buf)
            
            # 环境步进
            obs_buf, privileged_obs_buf, rewards, dones, infos, _, _ = env.step(actions)
            
            # 累积统计
            current_episode_reward += rewards
            current_episode_length += 1
            
            # 处理episode结束
            done_indices = dones.nonzero(as_tuple=False).flatten()
            if len(done_indices) > 0:
                for idx in done_indices:
                    episode_rewards.append(current_episode_reward[idx].item())
                    episode_lengths.append(current_episode_length[idx].item())
                    
                    if len(episode_rewards) % 10 == 0:  # 每10个episode打印一次
                        avg_reward = np.mean(episode_rewards[-10:])
                        avg_length = np.mean(episode_lengths[-10:])
                        print(f"Episodes {len(episode_rewards)-9}-{len(episode_rewards)}: "
                              f"平均奖励 = {avg_reward:.2f}, 平均长度 = {avg_length:.1f}")
                
                current_episode_reward[done_indices] = 0
                current_episode_length[done_indices] = 0
            
            # 记录一些状态信息（可选）
            if i % 100 == 0 and i > 0:
                if env.num_envs > robot_index:
                    base_pos = env.base_pos[robot_index].cpu().numpy()
                    base_vel = env.base_lin_vel[robot_index].cpu().numpy()
                    print(f"步数 {i}: 位置 [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}], "
                          f"速度 [{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}]")
    
    except KeyboardInterrupt:
        print("\n演示被用户停止")
    
    # 打印最终统计
    if episode_rewards:
        print("\n" + "="*50)
        print("演示统计:")
        print(f"总Episode数: {len(episode_rewards)}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
        print(f"最高奖励: {np.max(episode_rewards):.2f}")
        print(f"最低奖励: {np.min(episode_rewards):.2f}")
        print(f"平均长度: {np.mean(episode_lengths):.1f} steps")
        print("="*50)

def find_model_path(args):
    """查找模型文件路径"""
    if args.load_run != "-1" and os.path.exists(args.load_run):
        return args.load_run
    
    # 查找最新的模型文件
    patterns = [
        "logs/*/model_*.pt",
        "logs/**/*.pt", 
        "checkpoints/**/*.pt",
        "models/**/*.pt",
        "*.pt"
    ]
    
    for pattern in patterns:
        model_files = glob.glob(pattern, recursive=True)
        if model_files:
            # 按修改时间排序，返回最新的
            latest_model = max(model_files, key=os.path.getmtime)
            return latest_model
    
    raise FileNotFoundError("找不到任何模型文件，请检查logs目录")

def load_model(policy, checkpoint_path, device):
    """加载模型参数"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 尝试不同的键名
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ac_parameters' in checkpoint:
        state_dict = checkpoint['ac_parameters']
    elif 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载参数
    try:
        policy.load_state_dict(state_dict, strict=False)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"警告: {e}")
        # 兼容性加载
        policy_dict = policy.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if k in policy_dict and v.size() == policy_dict[k].size()}
        policy_dict.update(filtered_dict)
        policy.load_state_dict(policy_dict)
        print(f"✓ 加载了 {len(filtered_dict)}/{len(policy_dict)} 个参数")

if __name__ == '__main__':
    # 可选功能开关
    EXPORT_POLICY = False  # 是否导出策略
    RECORD_FRAMES = False  # 是否录制帧
    MOVE_CAMERA = False    # 是否移动相机
    
    args = get_args()
    play(args)

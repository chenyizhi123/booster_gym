#!/usr/bin/env python3
"""
T1机器人足球模型演示脚本
基于utils/runner.py的简化版本
"""

import os
import yaml
import argparse
import numpy as np
import torch
import glob

# Isaac Gym必须在其他模块之前导入
import isaacgym

from rsl_rl.modules.actor_critic import ActorCritic
from envs import *

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='T1机器人足球演示')
    parser.add_argument("--task", type=str, default="kick_2D", help="任务名称")
    parser.add_argument("--checkpoint", type=str, default="-1", help="模型路径，-1表示最新模型")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    
    return parser.parse_args()

def play(args):
    """主演示函数"""
    
    # 加载配置文件
    cfg_file = os.path.join("envs", f"{args.task}.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 更新配置
    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    if args.headless:
        cfg["basic"]["headless"] = args.headless
    if args.seed:
        cfg["basic"]["seed"] = args.seed
    
    # 禁用视频录制（演示时不需要）
    cfg["viewer"]["record_video"] = False
    
    print(f"创建环境: {args.task}")
    
    # 动态创建环境
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    
    device = cfg["basic"]["rl_device"]
    
    # 创建模型
    model = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation='elu',
        init_noise_std=1.0,
    ).to(device)
    
    # 加载模型
    checkpoint_path = find_checkpoint(args.checkpoint)
    print(f"加载模型: {checkpoint_path}")
    load_model(model, checkpoint_path, device)
    model.eval()
    
    # 重置环境
    obs, infos = env.reset()
    obs = obs.to(device)
    
    print("开始演示...")
    print("按 Ctrl+C 停止")
    print("-" * 50)
    
    step_count = 0
    episode_count = 0
    episode_rewards = []
    current_episode_reward = torch.zeros(env.num_envs, device=device)
    
    try:
        while True:
            with torch.no_grad():
                # 使用模型推理动作，直接使用均值（不采样）
                actions = model.act_inference(obs)
                
                # 环境步进
                obs, rewards, dones, infos = env.step(actions)
                obs = obs.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)
                
                current_episode_reward += rewards
                step_count += 1
                
                # 处理episode结束
                done_indices = dones.nonzero(as_tuple=False).flatten()
                if len(done_indices) > 0:
                    for idx in done_indices:
                        episode_rewards.append(current_episode_reward[idx].item())
                        episode_count += 1
                        print(f"Episode {episode_count} 结束，奖励: {current_episode_reward[idx].item():.2f}")
                    
                    current_episode_reward[done_indices] = 0
                
                # 每100步打印一次状态
                if step_count % 100 == 0:
                    if hasattr(env, 'base_pos') and env.num_envs > 0:
                        pos = env.base_pos[0].cpu().numpy()
                        print(f"步数 {step_count}: 位置 [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    else:
                        print(f"步数 {step_count}")
    
    except KeyboardInterrupt:
        print("\n演示被用户停止")
    
    # 打印统计信息
    if episode_rewards:
        print("\n" + "="*50)
        print("演示统计:")
        print(f"总Episode数: {len(episode_rewards)}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
        print(f"最高奖励: {np.max(episode_rewards):.2f}")
        print(f"最低奖励: {np.min(episode_rewards):.2f}")
        print("="*50)

def find_checkpoint(checkpoint_arg):
    """查找检查点文件"""
    if checkpoint_arg != "-1" and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    
    # 查找最新的模型文件
    patterns = [
        "logs/**/*.pth",
        "logs/**/*.pt", 
        "checkpoints/**/*.pth",
        "models/**/*.pth",
        "*.pth"
    ]
    
    for pattern in patterns:
        model_files = glob.glob(pattern, recursive=True)
        if model_files:
            # 按修改时间排序，返回最新的
            latest_model = max(model_files, key=os.path.getmtime)
            return latest_model
    
    raise FileNotFoundError("找不到任何模型文件，请检查logs目录")

def load_model(model, checkpoint_path, device):
    """加载模型参数"""
    print(f"从 {checkpoint_path} 加载模型...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 尝试不同的键名
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

if __name__ == '__main__':
    args = get_args()
    play(args)

#!/usr/bin/env python3
"""
T1机器人足球训练 - 直接使用AMPOnPolicyRunner

这个脚本直接集成了AMP训练功能，无需单独的AMP训练脚本
支持普通PPO和AMP两种训练模式的自动切换
"""

# Isaac Gym必须在其他模块之前导入
import isaacgym
import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch


# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加rsl_rl模块路径
rsl_rl_path = os.path.join(current_dir, "rsl_rl")
if os.path.exists(rsl_rl_path):
    sys.path.insert(0, rsl_rl_path)

# 导入环境和训练器
from envs.kick_2D import kick_2D
from rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from utils.utils import set_seed

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='T1机器人足球训练')
    parser.add_argument("--task", type=str, default="kick_2D", 
                       help="任务名称 (默认: kick_2D)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型检查点路径")
    parser.add_argument("--num_envs", type=int, default=None,
                       help="环境数量")
    parser.add_argument("--headless", action="store_true",
                       help="无头模式运行")
    parser.add_argument("--sim_device", type=str, default=None,
                       help="仿真设备")
    parser.add_argument("--rl_device", type=str, default=None,
                       help="RL算法设备")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")
    parser.add_argument("--max_iterations", type=int, default=None,
                       help="最大训练迭代次数")
    parser.add_argument("--use_amp", default=True,
                       help="使用AMP训练")
    parser.add_argument("--amp_config", type=str, default=None,
                       help="AMP配置文件路径")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="日志目录")
    
    return parser.parse_args()

def load_config(task_name):
    """加载配置文件"""
    config_path = os.path.join("envs", f"{task_name}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    print(f"✓ 配置文件加载完成: {config_path}")
    return cfg

def update_config_from_args(cfg, args):
    """用命令行参数更新配置"""
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint
    if args.headless:
        cfg["basic"]["headless"] = True
    if args.sim_device is not None:
        cfg["basic"]["sim_device"] = args.sim_device
    if args.rl_device is not None:
        cfg["basic"]["rl_device"] = args.rl_device
    if args.seed is not None:
        cfg["basic"]["seed"] = args.seed
    if args.max_iterations is not None:
        cfg["basic"]["max_iterations"] = args.max_iterations
    
    return cfg

def setup_amp_config(cfg, use_amp):
    """设置AMP相关配置"""
    if use_amp:
        # 启用AMP训练 - 修改配置文件中的关键设置
        if 'env' not in cfg:
            cfg['env'] = {}
        cfg['env']['reference_state_initialization'] = True
        
        if 'runner' not in cfg:
            cfg['runner'] = {}
        cfg['runner']['algorithm_class_name'] = 'AMPPPO'
        
        if 'algorithm' not in cfg:
            cfg['algorithm'] = {}
        cfg['algorithm']['class_name'] = 'AMPPPO'
        
        print("✓ AMP配置已启用")
        print("  - 环境初始化: reference_state_initialization = true")
        print("  - 训练算法: AMPPPO")
        
        # 检查专家数据文件
        amp_files = cfg.get('env', {}).get('amp_motion_files', [])
        if not amp_files or not any(os.path.exists(f) for f in amp_files):
            print("⚠️  警告: 未找到有效的专家数据文件")
            print("   请先运行: python create_expert_data.py --method manual")
            
    else:
        # 普通PPO训练 - 确保禁用AMP
        if 'env' not in cfg:
            cfg['env'] = {}
        cfg['env']['reference_state_initialization'] = False
        
        if 'runner' not in cfg:
            cfg['runner'] = {}
        cfg['runner']['algorithm_class_name'] = 'PPO'
        
        if 'algorithm' not in cfg:
            cfg['algorithm'] = {}
        cfg['algorithm']['class_name'] = 'PPO'
        
        print("✓ 普通PPO配置已启用")
        print("  - 训练算法: PPO")
    
    return cfg

def create_env(cfg):
    """创建环境"""
    print(f"正在创建环境...")
    env = kick_2D(cfg)
    
    print(f"✓ 环境创建成功")
    print(f"  - 环境数量: {env.num_envs}")
    print(f"  - 观察维度: {env.num_obs}")
    print(f"  - 动作维度: {env.num_actions}")
    if hasattr(env, 'num_privileged_obs'):
        print(f"  - 特权观察维度: {env.num_privileged_obs}")
    if hasattr(env, 'num_amp_obs'):
        print(f"  - AMP观察维度: {env.num_amp_obs}")
    
    return env

def create_runner(env, cfg, use_amp, log_dir):
    """创建训练运行器"""
    device = cfg.get('basic', {}).get('rl_device', 'cuda:0')
    
    # 准备训练配置
    train_cfg = {
        'runner': cfg.get('runner', {}),
        'amp_algorithm': cfg.get('amp_algorithm', {}),
        'policy': cfg.get('policy', {}),
        'device': device,
        'env': cfg.get('env', {})
    }
    
    if use_amp:
        print("正在初始化AMP训练运行器...")
        
        # 检查AMP配置
        if not train_cfg['runner'].get('amp_motion_files'):
            print("⚠️  警告: 未指定专家动作数据文件")
            print("   如果没有专家数据，AMP将无法正常工作")
        
        runner = AMPOnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device
        )
        print("✓ AMP训练运行器创建成功")
    else:
        print("正在初始化PPO训练运行器...")
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device
        )
        print("✓ PPO训练运行器创建成功")
    
    return runner

def main():
    """主函数"""
    print("="*60)
    print("T1机器人足球训练系统")
    print("="*60)
    
    # 解析命令行参数
    args = parse_args()
    
    # 自动检测是否使用AMP
    use_amp = args.use_amp
    
    print(f"🤖 训练模式: {'AMP (对抗运动先验)' if use_amp else 'PPO (近端策略优化)'}")
    
    # 加载配置文件
    print(f"\n📖 加载配置文件...")
    try:
        cfg = load_config(args.task)
        cfg = update_config_from_args(cfg, args)
        cfg = setup_amp_config(cfg, use_amp)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return
    
    # 设置随机种子
    seed = cfg['basic'].get('seed', 42)
    if seed == -1:
        seed = np.random.randint(0, 10000)
        cfg['basic']['seed'] = seed
    set_seed(seed)
    print(f"🎲 随机种子: {seed}")
    
    # 设置日志目录
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = f"logs/{args.task}_{'amp' if use_amp else 'ppo'}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置到日志目录
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"📝 配置已保存到: {config_path}")
    
    # 创建环境
    try:
        env = create_env(cfg)
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建训练运行器
    try:
        runner = create_runner(env, cfg, use_amp, log_dir)
    except Exception as e:
        print(f"❌ 训练运行器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载检查点（如果指定）
    if cfg['basic'].get('checkpoint'):
        try:
            print(f"\n📂 加载检查点: {cfg['basic']['checkpoint']}")
            runner.load(cfg['basic']['checkpoint'])
            print("✓ 检查点加载成功")
        except Exception as e:
            print(f"⚠️  检查点加载失败: {e}")
    
    # 开始训练
    max_iterations = cfg['basic'].get('max_iterations', 30000)
    print(f"\n🚀 开始训练...")
    print(f"   最大迭代次数: {max_iterations}")
    print(f"   日志目录: {log_dir}")
    print(f"   设备: {cfg['basic'].get('rl_device', 'cuda:0')}")
    print(f"   TensorBoard: tensorboard --logdir={log_dir}")
    print("-"*60)
    
    try:
        start_time = time.time()
        runner.learn(num_learning_iterations=max_iterations)
        end_time = time.time()
        
        print("-"*60)
        print(f"✅ 训练完成！")
        print(f"   总耗时: {end_time - start_time:.2f} 秒")
        print(f"   模型保存位置: {log_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  训练被用户中断")
        print(f"   当前状态已保存到: {log_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if hasattr(env, 'gym') and env.gym is not None:
            env.gym.destroy_sim(env.sim)
            print("✓ 仿真资源已清理")

if __name__ == "__main__":
    main()

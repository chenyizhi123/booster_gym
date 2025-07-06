#!/usr/bin/env python3
"""
HRL训练脚本
使用标准ActorCritic网络训练高层策略
"""

import argparse
from utils.hrl_runner import HRLRunner

def main():
    parser = argparse.ArgumentParser(description='HRL训练脚本')
    parser.add_argument('--task', type=str, default='approach_hrl', 
                       help='任务名称 (默认: approach_hrl)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径')
    parser.add_argument('--num_envs', type=int, default=512,
                       help='环境数量')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行')
    parser.add_argument('--sim_device', type=str, default='cuda:0',
                       help='仿真设备')
    parser.add_argument('--rl_device', type=str, default='cuda:0',
                       help='强化学习设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_iterations', type=int, default=10000,
                       help='最大训练迭代次数')
    
    args = parser.parse_args()
    
    # 创建HRL训练器
    runner = HRLRunner(test=False)
    
    print("🚀 开始HRL训练...")
    print(f"📊 任务: {args.task}")
    print(f"📊 环境数量: {args.num_envs}")
    print(f"📊 最大迭代次数: {args.max_iterations}")
    print(f"🔧 仿真设备: {args.sim_device}")
    print(f"🔧 强化学习设备: {args.rl_device}")
    print("-" * 60)
    
    # 开始训练
    runner.train()

if __name__ == "__main__":
    main() 
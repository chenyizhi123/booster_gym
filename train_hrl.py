#!/usr/bin/env python3
"""
HRL训练脚本
使用标准ActorCritic网络训练高层策略
"""

from utils.hrl_runner import HRLRunner

def main():
    # 创建HRL训练器，直接读取命令行参数
    runner = HRLRunner(test=False)
    
    print("🚀 开始HRL训练...")
    print(f"📊 任务: {runner.cfg['basic']['task']}")
    print(f"📊 环境数量: {runner.cfg['env']['num_envs']}")
    print(f"📊 最大迭代次数: {runner.cfg['basic']['max_iterations']}")
    print(f"🔧 仿真设备: {runner.cfg['basic']['sim_device']}")
    print(f"🔧 强化学习设备: {runner.cfg['basic']['rl_device']}")
    print("-" * 60)
    
    # 开始训练
    runner.train()

if __name__ == "__main__":
    main() 
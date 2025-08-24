#!/usr/bin/env python3
"""
T1机器人足球训练快速启动脚本

这是一个便捷的训练启动脚本，提供了常用的训练场景
现在使用统一的配置文件 envs/kick_2D.yaml
"""

import subprocess
import sys
import os

def run_command(cmd):
    """运行命令并实时显示输出"""
    print(f"执行命令: {cmd}")
    print("-" * 60)
    
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # 实时打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        return rc == 0
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return False

def main():
    print("="*60)
    print("T1机器人足球训练快速启动")
    print("="*60)
    
    print("\n请选择操作模式:")
    print("1. 普通PPO训练 (快速开始)")
    print("2. AMP训练 (需要先生成专家数据)")
    print("3. 生成专家数据 (为AMP训练准备)")
    print("4. 继续训练 (从检查点恢复)")
    print("5. 模型演示 (加载训练好的模型)")
    print("6. 自定义参数训练")
    print("0. 退出")
    
    choice = input("\n请输入选择 (0-6): ").strip()
    
    if choice == "0":
        print("退出系统")
        return
    
    elif choice == "1":
        print("\n🚀 启动普通PPO训练...")
        cmd = "python train.py --task kick_2D --num_envs 2048 --max_iterations 30000"
        run_command(cmd)
    
    elif choice == "2":
        print("\n📋 启动AMP训练...")
        # 检查是否存在专家数据
        data_file = "data/expert_kick_data.yaml"
        if not os.path.exists(data_file):
            print(f"⚠️  专家数据文件不存在: {data_file}")
            print("请先选择选项3生成专家数据，或手动创建数据文件")
            return
        
        cmd = "python train.py --task kick_2D --use_amp --num_envs 2048 --max_iterations 50000"
        run_command(cmd)
    
    elif choice == "3":
        print("\n📝 生成专家数据...")
        os.makedirs("data", exist_ok=True)
        cmd = "python create_expert_data.py --method manual --output data/expert_kick_data.yaml"
        if run_command(cmd):
            print("\n✅ 专家数据生成完成!")
            print("现在可以选择选项2开始AMP训练")
    
    elif choice == "4":
        print("\n📂 继续训练...")
        checkpoint = input("请输入检查点文件路径 (或按回车使用最新的): ").strip()
        if not checkpoint:
            checkpoint = "-1"  # 使用最新检查点
        
        use_amp = input("是否使用AMP模式? (y/n): ").strip().lower() == 'y'
        
        cmd = f"python train.py --task kick_2D --checkpoint {checkpoint}"
        if use_amp:
            cmd += " --use_amp"
        
        run_command(cmd)
    
    elif choice == "5":
        print("\n🎬 模型演示...")
        
        # 基础参数
        checkpoint = input("模型文件路径 (或按回车使用最新的): ").strip()
        if not checkpoint:
            checkpoint = "-1"  # 使用最新模型
        
        # 演示选项
        num_envs = input("演示环境数量 (默认: 1): ").strip() or "1"
        headless = input("无头模式运行? (y/n): ").strip().lower() == 'y'
        
        # 构建命令
        cmd = f"python play.py --load_run {checkpoint} --num_envs {num_envs}"
        
        if headless:
            cmd += " --headless"
        
        print(f"\n将执行: {cmd}")
        confirm = input("确认执行? (y/n): ").strip().lower()
        if confirm == 'y':
            run_command(cmd)
        else:
            print("取消执行")
    
    elif choice == "6":
        print("\n⚙️  自定义参数训练...")
        
        # 基础参数
        task = input(f"任务名称 (默认: kick_2D): ").strip() or "kick_2D"
        num_envs = input(f"环境数量 (默认: 2048): ").strip() or "2048"
        max_iter = input(f"最大迭代次数 (默认: 30000): ").strip() or "30000"
        
        # AMP选项
        use_amp = input("是否使用AMP? (y/n): ").strip().lower() == 'y'
        
        # 其他选项
        headless = input("无头模式? (y/n): ").strip().lower() == 'y'
        device = input("设备 (默认: cuda:0): ").strip() or "cuda:0"
        seed = input("随机种子 (默认: 42): ").strip() or "42"
        
        # 构建命令
        cmd = f"python train.py --task {task} --num_envs {num_envs} --max_iterations {max_iter} --rl_device {device} --seed {seed}"
        
        if use_amp:
            cmd += " --use_amp"
        if headless:
            cmd += " --headless"
        
        print(f"\n将执行: {cmd}")
        confirm = input("确认执行? (y/n): ").strip().lower()
        if confirm == 'y':
            run_command(cmd)
        else:
            print("取消执行")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()

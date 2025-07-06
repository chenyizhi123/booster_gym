#!/usr/bin/env python3
"""
HRL版本的测试脚本

使用方法:
python play_hrl.py --task approach_hrl --checkpoint path/to/checkpoint.pth

特点:
- 加载训练好的HRL模型
- 可视化高层策略和低层控制的协同工作
- 支持录制视频
"""

import isaacgym
from utils.hrl_runner import HRLRunner

if __name__ == "__main__":
    print("🎮 启动HRL测试系统")
    print("=" * 50)
    print("📋 系统配置:")
    print("  - 高层策略频率: 5Hz")
    print("  - 低层控制频率: 50Hz")
    print("  - 模式: 测试/演示")
    print("  - 可视化: 启用")
    print("=" * 50)
    
    runner = HRLRunner(test=True)
    runner.play() 
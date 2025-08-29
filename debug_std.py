#!/usr/bin/env python3

import torch
import yaml

# 加载配置文件
with open('envs/kick_2D.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 模拟关节限制范围 (以弧度为单位，T1机器人的大概范围)
# 基于URDF文件，大多数关节范围大约是 [-1.57, 1.57] (±90度)
dof_pos_limits = torch.tensor([
    [-1.57, 1.57],  # 示例关节1
    [-1.57, 1.57],  # 示例关节2  
    [-3.14, 3.14],  # 示例关节3 (一些关节可能有更大的范围)
    [-1.0, 1.0],    # 示例关节4
    [-2.0, 2.0],    # 示例关节5
] * 5)  # 假设有25个关节

# 获取min_normalized_std配置
min_normalized_std = config['runner']['min_normalized_std'][0]
print(f"配置的 min_normalized_std: {min_normalized_std}")

# 计算实际的min_std
min_std = (
    torch.tensor([min_normalized_std]) *
    (torch.abs(dof_pos_limits[:, 1] - dof_pos_limits[:, 0]))
)

print(f"关节范围示例:")
for i in range(min(5, len(dof_pos_limits))):
    joint_range = dof_pos_limits[i, 1] - dof_pos_limits[i, 0]
    actual_min_std = min_normalized_std * joint_range.item()
    print(f"  关节{i+1}: 范围={joint_range:.3f}, 实际min_std={actual_min_std:.6f}")

print(f"\n计算得到的 min_std 值:")
print(f"  最小值: {min_std.min().item():.6f}")
print(f"  最大值: {min_std.max().item():.6f}")
print(f"  平均值: {min_std.mean().item():.6f}")

# 检查0.1318是否在这个范围内
problem_std = 0.1318
print(f"\n问题分析:")
print(f"  观察到的固定std值: {problem_std}")
print(f"  是否在min_std范围内: {min_std.min().item() <= problem_std <= min_std.max().item()}")

if min_std.min().item() > problem_std:
    print(f"  ❌ min_std太大! std被强制提升到 {min_std.min().item():.6f}")
elif min_std.max().item() < problem_std:
    print(f"  ✅ min_std合理，问题可能在别处")
else:
    print(f"  🤔 部分关节的min_std影响了探索")

print(f"\n建议:")
print(f"  当前 min_normalized_std = {min_normalized_std}")
print(f"  建议降低到 0.005-0.01 范围")
recommended_std = 0.008
recommended_min_std = recommended_std * (dof_pos_limits[:, 1] - dof_pos_limits[:, 0])
print(f"  如果设为 {recommended_std}: min_std范围 = [{recommended_min_std.min().item():.6f}, {recommended_min_std.max().item():.6f}]")

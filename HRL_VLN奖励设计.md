# HRL-VLN奖励设计说明

## 设计理念

将足球机器人的approach任务建模为**VLN（Vision-Language Navigation）**问题：
- 🎯 **目标导向**: 分阶段完成复杂任务
- 🚀 **增量决策**: 专注于速度增量（加速度）的合理性
- 📍 **空间感知**: 基于距离自动切换任务阶段

## 任务分解

### 🏃 阶段1: 接近球 (距离 > 1.0m)
**目标**: 快速向球移动
```python
# 评估速度增量方向是否指向球
direction_alignment = dot(delta_velocity_unit, ball_direction_unit)
approach_reward = direction_alignment * weight["approach"] * stage_mask
```
**关键点**:
- 速度增量 `[ΔVx, ΔVy]` 应该指向球
- 距离远时鼓励大的速度增量
- 不关心角度，专注快速接近

### 🎯 阶段2: 对齐球门 (0.5m < 距离 ≤ 1.0m)
**目标**: 调整角度，使机器人-球-球门三点对齐
```python
# 评估角速度增量是否有助于对齐
cos_angle = dot(robot_to_ball_unit, ball_to_goal_unit)
desired_angular_delta = clamp(acos(cos_angle) * 0.5, -0.1, 0.1)
angular_error = abs(actual_angular_delta - desired_angular_delta)
alignment_reward = (cos_angle - angular_error) * weight["alignment"] * stage_mask
```
**关键点**:
- 角速度增量 `ΔWz` 应该帮助对齐
- 线速度增量继续向球移动但减速
- 对齐度越高奖励越大

### ⚽ 阶段3: 带球前进 (0.3m < 距离 ≤ 0.5m)
**目标**: 保持控球，向球门方向移动
```python
# 评估是否向球门方向带球
direction_alignment = dot(delta_velocity_unit, ball_to_goal_unit)
speed_penalty = where(delta_speed > 0.1, (delta_speed - 0.1) * 2.0, 0.0)
dribble_reward = (direction_alignment - speed_penalty) * weight["dribble"] * stage_mask
```
**关键点**:
- 速度增量应该指向球门
- 控制速度，避免失控（速度增量不宜过大）
- 保持与球的接触

### 🚀 阶段4: 加速射门 (距离 ≤ 0.3m)
**目标**: 快速加速，将球踢向球门
```python
# 评估射门的方向和力度
direction_alignment = dot(delta_velocity_unit, ball_to_goal_unit)
speed_reward = clamp(delta_speed * 5.0, 0.0, 2.0)  # 鼓励大速度增量
shoot_reward = (direction_alignment + speed_reward) * weight["shoot"] * stage_mask
```
**关键点**:
- 速度增量应该大且指向球门
- 鼓励加速（与阶段3相反）
- 方向准确性至关重要

### ⚡ 增量质量评估 (全阶段)
**目标**: 确保速度增量的合理性
```python
# 避免过大变化和无意义微调
smoothness_penalty = where(delta_magnitude > 0.2, (delta_magnitude - 0.2) * 2.0, 0.0)
action_penalty = where(delta_magnitude < 0.01, 0.1, 0.0)
delta_reward = -(smoothness_penalty + action_penalty) * weight["delta_quality"]
```
**关键点**:
- 惩罚过大的速度变化（> 0.2）
- 惩罚无意义的微小变化（< 0.01）
- 鼓励有意义的适度调整

## 阶段切换逻辑

```python
def get_current_stage(ball_distance):
    if ball_distance > 1.0:
        return "approach"     # 接近球
    elif ball_distance > 0.5:
        return "alignment"    # 对齐球门
    elif ball_distance > 0.3:
        return "dribble"      # 带球前进
    else:
        return "shoot"        # 加速射门
```

## 奖励权重设计

```yaml
reward:
  approach: 1.0      # 基础接近奖励
  alignment: 2.0     # 对齐最重要（决定成功率）
  dribble: 1.5       # 控球技巧
  shoot: 3.0         # 射门最高奖励
  delta_quality: 0.5 # 增量质量保证
```

**权重原理**:
- `shoot > alignment > dribble > approach`: 后期阶段更重要
- `delta_quality`: 较小权重，避免过度约束

## 与传统方法对比

### 传统方法
```python
# ❌ 混合所有阶段的奖励
reward = distance_reward + alignment_reward + velocity_reward + ...
```
**问题**: 不同阶段的目标冲突，策略难以学习

### VLN方法（本设计）
```python
# ✅ 基于阶段的清晰奖励
if stage == "approach":
    reward = approach_reward + delta_reward
elif stage == "alignment":
    reward = alignment_reward + delta_reward
# ...
```
**优势**: 每个阶段目标明确，学习效率高

## 训练策略

### 课程学习
1. **初期**: 主要在阶段1-2训练（距离较远）
2. **中期**: 增加阶段3训练（带球控制）
3. **后期**: 重点训练阶段4（射门精度）

### 监控指标
- `approach_reward`: 应在远距离时激活
- `alignment_reward`: 应在中距离时激活
- `dribble_reward`: 应在近距离时激活
- `shoot_reward`: 应在最近距离时激活
- `delta_quality`: 应始终保持合理范围

### 调试技巧
```python
# 可视化当前阶段
current_stage = get_current_stage(ball_distance)
print(f"Stage: {current_stage}, Distance: {ball_distance:.2f}")

# 监控奖励分布
print(f"Rewards - Approach: {approach_reward.mean():.3f}, "
      f"Alignment: {alignment_reward.mean():.3f}, "
      f"Dribble: {dribble_reward.mean():.3f}, "
      f"Shoot: {shoot_reward.mean():.3f}")
```

## 预期行为

1. **远距离**: 机器人快速向球移动，不关心方向
2. **中距离**: 机器人调整角度，与球门对齐
3. **近距离**: 机器人小心控球，向球门移动
4. **最近距离**: 机器人加速踢球，力求准确

这种设计模拟了真实足球运动员的决策过程，确保每个阶段的行为都是合理且高效的。 
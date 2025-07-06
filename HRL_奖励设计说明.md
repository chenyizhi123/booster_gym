# HRL奖励设计说明

## 设计理念

HRL架构中，高层策略专注于**速度指令决策**，而非底层运动控制。因此奖励函数应该：
- ✅ 鼓励正确的速度指令选择
- ✅ 评估速度指令的合理性
- ❌ 避免底层运动控制的奖励（如扭矩、关节速度等）

## 奖励组成

### 🎯 高层决策奖励 (主要)

#### 1. 接近球奖励 (`approach_ball: 1.0`)
```python
approach_reward = -ball_distance * self.reward_weights["approach_ball"]
```
- **目的**: 鼓励机器人向球移动
- **原理**: 距离越近，奖励越高

#### 2. 对准球门奖励 (`alignment: 2.0`)
```python
alignment_reward = cos_angle * self.reward_weights["alignment"]
```
- **目的**: 鼓励从正确角度接近球
- **原理**: 机器人→球方向 与 球→球门方向 越一致，奖励越高

#### 3. 速度方向奖励 (`velocity_direction: 1.5`)
```python
direction_reward = direction_alignment * self.reward_weights["velocity_direction"]
```
- **目的**: 鼓励速度指令指向球
- **原理**: 速度方向与球方向越一致，奖励越高

#### 4. 速度大小奖励 (`velocity_magnitude: 0.5`)
```python
desired_speed = torch.clamp(ball_dist * 0.5, 0.1, 1.0)
speed_reward = -speed_error * self.reward_weights["velocity_magnitude"]
```
- **目的**: 鼓励合理的速度大小
- **原理**: 距离远→高速度，距离近→低速度

#### 5. 角速度合理性 (`angular_velocity: 0.3`)
```python
desired_ang_vel = torch.clamp(angle_error * 2.0, 0.0, 1.0)
ang_vel_reward = -ang_vel_error * self.reward_weights["angular_velocity"]
```
- **目的**: 鼓励合理的角速度指令
- **原理**: 需要转向时使用角速度，不需要时保持为0

#### 6. 指令平滑性 (`command_smoothness: 0.1`)
```python
smoothness_penalty = -torch.sum(torch.square(self.last_delta_commands), dim=-1)
```
- **目的**: 避免剧烈的指令变化
- **原理**: 平滑的指令变化有利于底层执行

#### 7. 成功奖励 (`success: 10.0`)
```python
success_reward = 到达理想踢球位置时的高奖励
```
- **目的**: 鼓励到达合适的踢球位置
- **原理**: 在理想距离(0.3m)附近给予高奖励

### 🛡️ 安全性奖励 (辅助)

#### 1. 姿态保护 (`orientation: -0.2`)
- **目的**: 防止机器人翻倒
- **原理**: 惩罚过大的姿态角度

#### 2. 高度保护 (`base_height: -0.5`)
- **目的**: 保持合适的身体高度
- **原理**: 偏离目标高度时给予惩罚

#### 3. 碰撞保护 (`collision: -1.0`)
- **目的**: 避免不当碰撞
- **原理**: 检测到碰撞时给予惩罚

#### 4. 关节限位保护 (`dof_pos_limits: -10.0`)
- **目的**: 保护关节不超出限位
- **原理**: 关节位置超限时给予强惩罚

## 与传统方法的对比

### 传统方法 (不适用于HRL)
```python
# ❌ 这些奖励不适用于高层策略
torques_penalty = -torch.sum(torch.square(self.torques), dim=-1)
dof_vel_penalty = -torch.sum(torch.square(self.dof_vel), dim=-1)
action_rate_penalty = -torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
```
- **问题**: 高层策略无法直接控制关节力矩和速度
- **后果**: 奖励信号与实际控制能力不匹配

### HRL方法 (本设计)
```python
# ✅ 这些奖励专注于速度指令决策
velocity_direction_reward = 鼓励正确的速度方向
velocity_magnitude_reward = 鼓励合理的速度大小
angular_velocity_reward = 鼓励合理的角速度
```
- **优势**: 奖励信号直接对应高层策略的控制能力
- **效果**: 策略学习更高效，收敛更快

## 参数调优建议

### 奖励权重优先级
1. **alignment (2.0)** - 最重要，决定踢球成功率
2. **velocity_direction (1.5)** - 次重要，决定移动效率
3. **approach_ball (1.0)** - 基础奖励，保证向球移动
4. **velocity_magnitude (0.5)** - 精细调节，优化速度选择
5. **angular_velocity (0.3)** - 辅助调节，优化转向
6. **command_smoothness (0.1)** - 最小权重，避免过度约束

### 调优策略
- **训练初期**: 提高`approach_ball`权重，确保基本行为
- **训练中期**: 提高`alignment`权重，优化策略质量
- **训练后期**: 调节`velocity_*`权重，精细化控制

## 验证方法

### 训练指标
- `approach_reward`: 应该逐渐增加
- `alignment_reward`: 应该逐渐增加
- `velocity_direction`: 应该逐渐增加
- `smoothness_penalty`: 应该逐渐减少

### 行为观察
- 机器人应该向球移动
- 接近球时速度应该减慢
- 应该从合适角度接近球
- 指令变化应该平滑

这种设计确保高层策略专注于"如何给出好的速度指令"，而不是"如何控制关节"。 
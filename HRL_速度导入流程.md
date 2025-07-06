# HRL速度导入流程详解

## 概述

本文档详细说明了HRL系统中高层策略的速度增量如何导入到底层t1.py模型中的完整流程。

## 🔄 数据流程图

```
高层策略网络 → 3维速度增量 → 速度指令更新 → 低层观察构建 → t1.py网络 → 关节动作
     ↓              ↓              ↓              ↓             ↓           ↓
  [ΔVx,ΔVy,ΔW]  增量累积    [Vx,Vy,W_ang]    47维观察     12维动作    物理仿真
```

## 📋 详细步骤

### 步骤1: 高层策略输出 (5Hz)

**文件**: `utils/hrl_runner.py` 第180-185行

```python
# 高层策略网络推理
dist = self.model.act(obs)  # 输入19维观察
act = dist.sample()         # 输出3维速度增量 [ΔVx, ΔVy, ΔW_ang]
act = torch.clamp(act, -1.0, 1.0)  # 限制范围
```

### 步骤2: 速度指令更新 (5Hz)

**文件**: `envs/approach_hrl.py` 第476-504行

```python
def _update_high_level_commands(self, delta_commands):
    # 1. 增加决策计数器
    self.decision_counter += 1
    
    # 2. 检查是否需要更新 (每10帧)
    update_mask = (self.decision_counter % self.decision_interval == 0)
    
    if update_mask.any():
        # 3. 限制增量范围
        limited_delta = torch.clamp(delta_commands[env_ids], -self.delta_limits, self.delta_limits)
        
        # 4. 应用增量更新 (关键步骤!)
        self.current_commands[env_ids] = self.last_commands[env_ids] + limited_delta
        
        # 5. 限制绝对速度范围
        self.current_commands[env_ids] = torch.clamp(
            self.current_commands[env_ids], -self.command_limits, self.command_limits
        )
    
    # 6. 将高层命令设置为低层的速度指令 (每帧都执行)
    self.commands[:, :3] = self.current_commands  # 这里是关键！
```

**关键点**: `self.commands[:, :3]` 就是传递给t1.py网络的速度指令！

### 步骤3: 低层观察构建 (50Hz)

**文件**: `envs/approach_hrl.py` 第545-575行

```python
def _build_low_level_observations(self):
    # 速度指令归一化
    commands_scale = torch.tensor([
        self.cfg["normalization"]["lin_vel"],   # Vx缩放
        self.cfg["normalization"]["lin_vel"],   # Vy缩放  
        self.cfg["normalization"]["ang_vel"]    # W_ang缩放
    ], device=self.device)
    
    # 构建47维观察
    low_level_obs = torch.cat([
        # 重力投影 (3维)
        apply_randomization(self.projected_gravity, ...) * self.cfg["normalization"]["gravity"],
        # 角速度 (3维)
        apply_randomization(self.base_ang_vel, ...) * self.cfg["normalization"]["ang_vel"],
        # 速度指令 (3维) - 这里使用高层策略的输出！
        self.commands[:, :3] * commands_scale,  # 关键！速度指令在这里导入
        # 步态信息 (2维)
        (torch.cos(2 * torch.pi * self.gait_process) * ...).unsqueeze(-1),
        (torch.sin(2 * torch.pi * self.gait_process) * ...).unsqueeze(-1),
        # 关节位置 (12维)
        apply_randomization(self.dof_pos - self.default_dof_pos, ...) * ...,
        # 关节速度 (12维)
        apply_randomization(self.dof_vel, ...) * ...,
        # 上一步动作 (12维)
        self.last_low_level_actions,
    ], dim=-1)
    
    return low_level_obs  # 返回47维观察
```

**关键点**: `self.commands[:, :3] * commands_scale` 将速度指令导入到第6-8维观察中！

### 步骤4: t1.py网络推理 (50Hz)

**文件**: `envs/approach_hrl.py` 第510-520行

```python
def _execute_low_level_control(self):
    # 1. 构建t1.py需要的47维观察
    low_level_obs = self._build_low_level_observations()
    
    # 2. 使用训练好的t1.py网络推理关节动作
    if self.trained_locomotion_policy is not None:
        with torch.no_grad():  # 冻结低层网络参数
            joint_actions = self.trained_locomotion_policy(low_level_obs)  # 47维 → 12维
    else:
        joint_actions = torch.zeros(self.num_envs, 12, device=self.device)
```

**关键点**: `self.trained_locomotion_policy(low_level_obs)` 将包含速度指令的47维观察转换为12维关节动作！

### 步骤5: t1.py模型加载

**文件**: `envs/approach_hrl.py` 第87-148行

```python
def _load_trained_locomotion_policy(self):
    # 1. 获取t1模型路径
    t1_model_config = self.hrl_cfg.get("t1_model", {})
    model_path = t1_model_config.get("checkpoint_path", None)
    
    # 2. 自动寻找最新模型
    if model_path == "-1":
        t1_checkpoints = glob.glob(os.path.join("logs", "**/T1_*.pth"), recursive=True)
        model_path = sorted(t1_checkpoints, key=os.path.getmtime)[-1]
    
    # 3. 加载模型检查点
    checkpoint = torch.load(model_path, map_location=self.device)
    
    # 4. 创建ActorCritic网络
    from utils.model import ActorCritic
    full_model = ActorCritic(12, 47, 47).to(self.device)
    
    # 5. 加载权重
    full_model.load_state_dict(checkpoint["model"])
    
    # 6. 只返回actor部分
    locomotion_policy = full_model.actor
    
    return locomotion_policy
```

## 🔧 配置设置

### HRL配置 (approach_hrl.yaml)

```yaml
hrl:
  # t1模型配置
  t1_model:
    checkpoint_path: "-1"  # 自动寻找最新模型
    # 或指定具体路径: "logs/your_t1_model.pth"
  
  # 动作空间限制
  action_space:
    command_limits: [2.0, 2.0, 2.0]  # 速度指令上限 [Vx, Vy, W_ang]
    delta_limits: [0.5, 0.5, 0.5]    # 速度增量上限 [ΔVx, ΔVy, ΔW_ang]
```

## 📊 观察维度对应

### t1.py网络的47维观察结构

| 维度 | 内容 | 来源 |
|------|------|------|
| 0-2  | 重力投影 | 机器人姿态 |
| 3-5  | 角速度 | 机器人运动状态 |
| **6-8**  | **速度指令** | **高层策略输出** |
| 9-10 | 步态信息 | 步态生成器 |
| 11-22 | 关节位置 | 机器人关节状态 |
| 23-34 | 关节速度 | 机器人关节状态 |
| 35-46 | 上一步动作 | 历史动作 |

**关键**: 第6-8维就是高层策略的速度指令！

## 🎯 速度指令的物理含义

- **Vx**: 机器人局部坐标系前后方向的期望速度 (m/s)
- **Vy**: 机器人局部坐标系左右方向的期望速度 (m/s)
- **W_ang**: 机器人绕Z轴的期望角速度 (rad/s)

## 🔄 完整的数据流

```
1. 高层策略网络
   输入: 19维观察 (球位置、球门方向等)
   输出: 3维速度增量 [ΔVx, ΔVy, ΔW_ang]

2. 速度指令更新 (每10帧)
   新指令 = 旧指令 + 速度增量
   self.commands[:, :3] = [Vx, Vy, W_ang]

3. 低层观察构建 (每帧)
   47维观察[6:9] = self.commands[:, :3] * 归一化系数

4. t1.py网络推理 (每帧)
   输入: 47维观察 (包含速度指令)
   输出: 12维关节动作

5. 物理仿真
   关节动作 → 关节力矩 → 机器人运动
```

## 🚀 使用示例

### 1. 准备t1.py模型

```bash
# 训练t1.py模型
python train.py --task T1

# 检查模型文件
ls logs/  # 应该有T1相关的.pth文件
```

### 2. 配置HRL

```yaml
# 在approach_hrl.yaml中设置
hrl:
  t1_model:
    checkpoint_path: "logs/your_t1_model.pth"  # 或使用"-1"自动寻找
```

### 3. 开始HRL训练

```bash
python train_hrl.py --task approach_hrl
```

### 4. 验证速度导入

```bash
# 运行测试脚本
python test_hrl_system.py

# 查看输出，确认t1模型加载成功
# ✅ t1模型加载成功
```

## 🐛 常见问题

### Q1: t1模型加载失败

**原因**: 找不到t1模型文件
**解决**: 
```bash
# 检查logs目录
find logs -name "*T1*.pth"

# 或手动指定路径
# approach_hrl.yaml中设置具体路径
```

### Q2: 速度指令没有传递

**原因**: 观察构建错误
**检查**: 
```python
# 在_build_low_level_observations中添加调试
print(f"Speed commands: {self.commands[:, :3]}")
print(f"Low level obs[6:9]: {low_level_obs[:, 6:9]}")
```

### Q3: 机器人不动

**原因**: 速度指令为零或t1模型未正确加载
**检查**:
```python
# 在_update_high_level_commands中添加调试
print(f"Current commands: {self.current_commands}")
print(f"Delta commands: {delta_commands}")
```

## 📈 性能监控

在TensorBoard中监控：
- `curriculum/level_*_count`: 课程学习进度
- `reward/potential`: 势能奖励变化
- `reward/smoothness`: 动作平滑性

## 🎉 总结

HRL系统的速度导入流程：

1. **高层策略** (5Hz) → 生成速度增量
2. **增量累积** → 更新速度指令
3. **观察构建** → 将速度指令嵌入47维观察
4. **t1.py推理** → 将观察转换为关节动作
5. **物理仿真** → 执行关节动作

关键在于 `self.commands[:, :3]` 这个变量，它连接了高层策略和低层控制！ 
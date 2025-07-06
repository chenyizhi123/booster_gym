# HRL系统 - 标准ActorCritic架构

## 🏗️ 系统架构

### 架构概览
```
环境状态 (14维)
       ↓
┌─────────────────────────────────────┐
│     高层策略 (需要训练)                │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Actor     │  │   Critic    │   │
│  │ (生成增量)   │  │ (评估价值)   │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
       ↓ (3维速度增量)
   速度指令合成 (47维低层观察)
       ↓
┌─────────────────────────────────────┐
│     低层控制 (已训练好)               │
│  ┌─────────────┐                    │
│  │   Actor     │  (无Critic)        │
│  │ (生成关节)   │                    │
│  └─────────────┘                    │
└─────────────────────────────────────┘
       ↓ (12维关节动作)
    机器人执行动作
```

### 网络结构
HRL系统使用**分层网络架构**：

```python
# 高层策略网络 (需要训练 - 完整ActorCritic)
high_level_model = ActorCritic(
    num_act=3,          # 3维速度增量 [ΔVx, ΔVy, ΔWz]
    num_obs=14,         # 14维高层观察
    num_privileged_obs=20  # 20维特权观察
)
# ✅ 包含Actor + Critic，用于PPO训练

# 低层控制网络 (已训练好 - 只需Actor)
locomotion_policy = t1_model.actor  # 只使用Actor部分
# ✅ 只需Actor，相当于play模式，直接推理
```

### 分层控制
- **高层策略 (5Hz)**：ActorCritic网络，学习生成速度增量指令
- **低层控制 (50Hz)**：Actor网络，执行具体关节动作（不需要训练）

## 🧠 网络架构详解

### 高层策略网络 (完整ActorCritic)

**Actor网络** - 生成速度增量：
```python
self.actor = torch.nn.Sequential(
    torch.nn.Linear(14, 256),           # 高层观察 -> 256
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),          # 256 -> 128
    torch.nn.ELU(),
    torch.nn.Linear(128, 128),          # 128 -> 128
    torch.nn.ELU(),
    torch.nn.Linear(128, 3),            # 128 -> 3维速度增量
)
```

**Critic网络** - 评估状态价值：
```python
self.critic = torch.nn.Sequential(
    torch.nn.Linear(34, 256),           # 14+20 -> 256 (观察+特权)
    torch.nn.ELU(),
    torch.nn.Linear(256, 256),          # 256 -> 256
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),          # 256 -> 128
    torch.nn.ELU(),
    torch.nn.Linear(128, 1),            # 128 -> 1 (价值)
)
```

### 低层控制网络 (仅Actor)

**Actor网络** - 生成关节动作：
```python
# 使用预训练t1.py模型的Actor部分
locomotion_policy = t1_full_model.actor
# 输入: 47维低层观察 (包含速度指令)
# 输出: 12维关节动作
```

### 动作分布
```python
# 对数标准差参数
self.logstd = torch.nn.parameter.Parameter(
    torch.full((1, num_act), fill_value=-2.0), requires_grad=True
)

# 生成正态分布
def act(self, obs):
    action_mean = self.actor(obs)
    action_std = torch.exp(self.logstd).expand_as(action_mean)
    return torch.distributions.Normal(action_mean, action_std)
```

## 🚀 使用方法

### 1. 测试系统
```bash
python test_hrl.py
```

### 2. 开始训练
```bash
python train_hrl.py --task approach_hrl --num_envs 512
```

### 3. 继续训练
```bash
python train_hrl.py --task approach_hrl --checkpoint logs/approach_hrl_xxx.pth
```

## 📊 训练参数

### 网络参数
- **学习率**: 3e-4 (与t1.py一致)
- **批次大小**: 512个环境
- **经验长度**: 24步
- **mini-epochs**: 4次更新

### HRL参数
- **决策频率**: 每10帧 (5Hz)
- **速度增量范围**: [-1.0, 1.0]
- **绝对速度限制**: [-2.0, 2.0]

## 🎯 奖励系统

采用标准奖励框架，自动发现和调用奖励函数：

### 任务奖励
- `approach`: 接近球的奖励
- `alignment`: 对齐球门的奖励  
- `dribble`: 带球前进的奖励
- `shoot`: 射门的奖励
- `delta_quality`: 增量质量奖励

### 安全奖励
- `orientation`: 姿态保持
- `base_height`: 高度保持
- `collision`: 碰撞惩罚
- `dof_pos_limits`: 关节限制

## 🔧 技术细节

### 数据流详解
```python
# 🎯 完整的HRL数据流
def step(self, actions):  # actions是高层策略的3维输出
    # 1. 高层决策更新 (5Hz)
    self._update_high_level_commands(actions)
    
    # 2. 低层执行 (50Hz)  
    self._execute_low_level_control()

def _update_high_level_commands(self, delta_commands):
    """每10帧更新一次高层指令"""
    self.decision_counter += 1
    update_mask = (self.decision_counter % 10 == 0)
    
    if update_mask.any():
        # 更新速度增量
        env_ids = torch.where(update_mask)[0]
        self.current_commands[env_ids] = self.last_commands[env_ids] + delta_commands[env_ids]
    
    # 🔑 关键：将高层指令传递给低层
    self.commands[:, :3] = self.current_commands
```

### 低层执行
```python
def _execute_low_level_control(self):
    """每帧执行低层控制"""
    # 1. 构建低层观察 (47维) - 包含速度指令！
    low_level_obs = self._build_low_level_observations()
    
    # 2. 使用t1.py网络生成关节动作
    with torch.no_grad():
        joint_actions = self.trained_locomotion_policy(low_level_obs)
    
    # 3. 限制动作范围 (与t1.py保持一致)
    joint_actions = torch.clip(joint_actions, 
                              -self.cfg["normalization"]["clip_actions"], 
                               self.cfg["normalization"]["clip_actions"])
    
    # 4. 应用关节动作
    self._apply_joint_actions(joint_actions)

def _build_low_level_observations(self):
    """构建包含速度指令的47维观察"""
    return torch.cat([
        self.projected_gravity,           # 重力 (3维)
        self.base_ang_vel,               # 角速度 (3维)  
        self.commands[:, :3],            # 速度指令 (3维) ← 关键！
        self.gait_cos, self.gait_sin,    # 步态 (2维)
        self.dof_pos, self.dof_vel,      # 关节 (24维)
        self.last_actions,               # 历史动作 (12维)
    ], dim=-1)  # 总计47维
```

## 📈 监控指标

### 训练指标
- `value_loss`: 价值函数损失
- `actor_loss`: 策略损失  
- `entropy`: 动作熵
- `kl_mean`: KL散度

### 任务指标
- `rew_terms/approach`: 接近奖励
- `rew_terms/alignment`: 对齐奖励
- `rew_terms/dribble`: 带球奖励
- `rew_terms/shoot`: 射门奖励

## 🔒 安全机制

### 动作限制
HRL系统包含两层动作限制：

1. **高层动作限制**：
```python
# 高层策略输出的速度增量
action = torch.clamp(action, -1.0, 1.0)  # 限制在[-1, 1]
```

2. **低层动作限制**（关键！）：
```python
# 低层网络输出的关节动作
joint_actions = torch.clip(joint_actions, 
                          -self.cfg["normalization"]["clip_actions"], 
                           self.cfg["normalization"]["clip_actions"])
```

⚠️ **重要**：低层动作限制确保关节动作在安全范围内，防止机器人损坏。

## 🐛 调试建议

### 1. 检查网络维度
```python
print(f"Actor输入: {env.num_obs}")           # 应该是14
print(f"Critic输入: {env.num_obs + env.num_privileged_obs}")  # 应该是34
print(f"动作输出: {env.num_actions}")         # 应该是3
```

### 2. 检查t1模型加载
```python
if env.trained_locomotion_policy is not None:
    print("✅ t1模型加载成功")
else:
    print("❌ t1模型未加载，请检查路径")
```

### 3. 监控动作范围
```python
# 高层动作应该在[-1, 1]范围内
high_level_action = torch.clamp(high_level_action, -1.0, 1.0)

# 低层动作应该在clip_actions范围内
print(f"关节动作范围: {joint_actions.min():.3f} ~ {joint_actions.max():.3f}")
print(f"限制范围: ±{env.cfg['normalization']['clip_actions']}")
```

## 🎉 优势

1. **标准化**: 使用与t1.py相同的ActorCritic结构
2. **模块化**: 高层策略和低层控制分离
3. **可复用**: 利用预训练的运动控制模型
4. **可扩展**: 标准奖励框架便于添加新奖励
5. **可调试**: 完整的监控和日志系统

这个架构确保了HRL系统的稳定性和可维护性，同时保持了与现有代码的兼容性。 
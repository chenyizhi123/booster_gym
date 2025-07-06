# HRL (分层强化学习) 系统使用说明

## 概述

本HRL系统实现了足球机器人的分层控制架构，将复杂的足球接近任务分解为：
- **高层策略（指挥官）**：5Hz频率，负责生成速度指令
- **低层控制（执行者）**：50Hz频率，负责执行具体的关节动作

## 系统架构

```
球/球门状态 → 19维观察 → 高层策略网络 → 3维速度增量 → 速度指令
                                                              ↓
机器人状态 → 47维观察 → t1.py网络(冻结) → 12维关节动作 → 物理仿真
```

## 频率控制

- **高层策略**：每10帧更新一次 (50Hz ÷ 5Hz = 10)
- **低层控制**：每帧执行 (50Hz)
- **决策间隔**：在非更新帧保持上一次的指令不变

## 文件结构

```
├── envs/
│   ├── approach_hrl.py       # HRL环境实现
│   ├── approach_hrl.yaml     # HRL配置文件
│   └── approach_hrl_example.py # 使用示例
├── utils/
│   └── hrl_runner.py         # HRL专用训练器
├── train_hrl.py              # HRL训练脚本
├── play_hrl.py               # HRL测试脚本
└── test_hrl_system.py        # 系统测试脚本
```

## 使用方法

### 1. 系统测试

首先测试HRL系统是否正常工作：

```bash
python test_hrl_system.py
```

### 2. 开始训练

```bash
python train_hrl.py --task approach_hrl
```

### 3. 测试训练结果

```bash
python play_hrl.py --task approach_hrl --checkpoint path/to/checkpoint.pth
```

### 4. 监控训练进度

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开 http://localhost:6006

## 配置说明

### 关键配置项 (approach_hrl.yaml)

```yaml
# 环境配置
env:
  num_observations: 19      # 高层策略观察维度
  num_actions: 3           # 高层策略动作维度 [ΔVx, ΔVy, ΔW_ang]

# HRL配置
hrl:
  enabled: true
  decision_interval: 10    # 高层策略决策间隔（帧）
  
  # 高层策略网络
  high_level_policy:
    obs_dim: 19
    action_dim: 3
  
  # 动作空间限制
  action_space:
    command_limits: [2.0, 2.0, 2.0]  # 速度指令上限
    delta_limits: [0.5, 0.5, 0.5]    # 速度增量上限
  
  # 奖励权重
  reward_weights:
    distance_potential: 1.0
    alignment_potential: 0.5
    action_smoothness: -0.01
    survival: -0.01
    success: 10.0
```

## 观察空间 (19维)

1. **球相对位置** (3维): 球在机器人局部坐标系中的位置
2. **球相对速度** (3维): 球相对于机器人的速度
3. **球门相对方向** (3维): 球门在机器人局部坐标系中的方向
4. **机器人速度** (6维): 线速度(3) + 角速度(3)
5. **球距离** (1维): 机器人到球的距离
6. **球门距离** (1维): 机器人到球门的距离
7. **上一轮指令** (3维): 上一次的速度指令

## 动作空间 (3维)

高层策略输出3维速度增量：
- **ΔVx**: 前后方向速度增量
- **ΔVy**: 左右方向速度增量  
- **ΔW_ang**: 旋转速度增量

## 奖励函数

### 主要奖励项

1. **势能奖励**: 基于距离和对准的势能差
2. **平滑性惩罚**: 惩罚大的速度增量变化
3. **生存奖励**: 每步的基础奖励
4. **成功奖励**: 成功接近球并对准球门时的大奖励

### 奖励计算

```python
total_reward = potential_reward + smoothness_penalty + survival_reward + success_reward
```

## 课程学习

系统支持3级自动晋级课程：

1. **入门级** (0.3-0.8米): 球距离较近，角度范围小
2. **进阶级** (0.8-2.5米): 中等距离和角度
3. **高级** (2.5-5.0米): 远距离，全角度范围

## 训练技巧

### 1. 确保t1.py模型可用

HRL系统依赖训练好的基础运动控制模型，确保：
- t1.py模型已经训练完成
- 模型文件路径正确
- 模型参数被正确冻结

### 2. 调整学习率

高层策略网络相对简单，可以使用较高的学习率：

```yaml
algorithm:
  learning_rate: 3.0e-4
```

### 3. 监控关键指标

重点关注：
- 势能奖励的变化趋势
- 成功率的提升
- 动作平滑性

### 4. 课程学习调优

根据训练进度调整课程学习参数：

```yaml
curriculum:
  success_threshold: 0.8    # 晋级成功率阈值
  window_size: 100          # 成功率统计窗口
```

## 常见问题

### Q1: 训练不收敛怎么办？

1. 检查t1.py模型是否正确加载
2. 降低动作空间限制
3. 调整奖励权重
4. 增加课程学习的训练时间

### Q2: 频率控制不正常？

1. 检查decision_interval设置
2. 确认决策计数器逻辑
3. 运行test_hrl_system.py验证

### Q3: 机器人行为异常？

1. 检查观察空间的坐标变换
2. 确认速度指令的范围设置
3. 验证低层网络的输入构建

## 性能优化

### 1. 并行环境数量

根据GPU内存调整环境数量：

```yaml
env:
  num_envs: 4096  # 根据GPU内存调整
```

### 2. 网络架构优化

高层策略网络相对简单，可以适当减少网络层数：

```python
# 在HRLActorCritic中调整网络结构
self.actor = torch.nn.Sequential(
    torch.nn.Linear(num_obs, 128),  # 减少隐藏层大小
    torch.nn.ELU(),
    torch.nn.Linear(128, 64),
    torch.nn.ELU(),
    torch.nn.Linear(64, num_act),
    torch.nn.Tanh()
)
```

## 扩展功能

### 1. 多任务支持

可以扩展支持kick任务：

```python
# 在approach_hrl.py中添加kick相关逻辑
if self.task_type == "kick":
    # kick特定的观察和奖励
```

### 2. 注意力机制

为高层策略添加注意力机制：

```python
class AttentionHRLPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        # 实现注意力机制
```

### 3. 元学习

支持快速适应新场景：

```python
class MetaHRLPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        # 实现元学习能力
```

## 联系支持

如有问题，请检查：
1. 系统测试脚本输出
2. 训练日志
3. TensorBoard可视化
4. 配置文件设置

祝训练顺利！🚀 
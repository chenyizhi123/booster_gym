# 🎯 机器人位置订阅系统使用说明

## 📋 系统概述

现在机器人控制系统已经集成了位置订阅功能，可以：
1. 📡 通过ROS2订阅足球和球门位置
2. 🎯 自动更新HRL策略所需的变量
3. 🤖 基于位置信息做出智能决策

## 🔧 系统架构

```
外部视觉系统 → ROS2话题 → 位置订阅器 → 控制器 → HRL策略 → 机器人动作
     ↓             ↓          ↓        ↓        ↓
  识别球门位置   发布位置     接收处理   更新变量   生成命令
```

## 🚀 使用方法

### 方法1: 使用ROS2 (推荐)

1. **启动机器人控制器**
```bash
cd /home/cyz/Workspace/booster_gym/deploy
python deploy.py --config T1.yaml
```

2. **在另一个终端启动测试发布器**
```bash
cd /home/cyz/Workspace/booster_gym/deploy
python test_publisher.py
```

3. **或者手动发布位置数据**
```bash
# 发布球位置
ros2 topic pub /ball_position geometry_msgs/msg/PointStamped '{
  "header": {"frame_id": "map"}, 
  "point": {"x": 2.0, "y": 1.0, "z": 0.1}
}'

# 发布球门位置
ros2 topic pub /goal_position geometry_msgs/msg/PointStamped '{
  "header": {"frame_id": "map"}, 
  "point": {"x": 5.0, "y": 0.0, "z": 0.5}
}'
```

### 方法2: 模拟模式 (无需ROS2)

如果没有ROS2，系统会自动切换到模拟模式，生成移动的球位置。

## 📊 HRL变量更新

系统会自动更新以下HRL策略所需的变量：

1. **`ball_position`** - 球的3D位置 [x, y, z]
2. **`goal_dir`** - 从机器人到球门的归一化方向向量
3. **`ball_distance`** - 机器人到球的2D距离
4. **`goal_distance`** - 机器人到球门的2D距离
5. **`last_commands`** - 上一次的速度命令 [vx, vy, vyaw]

## 🛠️ 配置选项

在 `configs/T1.yaml` 中可以调整：

```yaml
policy:
  policy_path_hrl: "./models/T1_hrl.pt"  # HRL策略模型路径
  num_observations_hrl: 11               # HRL观察空间维度
  normalization:
    clip_actions_hrl: 0.2                # HRL动作限制
```

## 🔍 调试信息

系统会每100个控制循环打印一次位置信息：
- 🏀 球位置
- 🥅 门位置  
- 📏 球距离
- 📏 门距离

## 📡 ROS2话题

- **输入话题:**
  - `/ball_position` (geometry_msgs/PointStamped)
  - `/goal_position` (geometry_msgs/PointStamped)

- **监控命令:**
```bash
# 查看话题列表
ros2 topic list

# 监控位置数据
ros2 topic echo /ball_position
ros2 topic echo /goal_position

# 检查发布频率
ros2 topic hz /ball_position
```

## 🎮 控制模式

系统支持三种控制模式：

1. **🚨 紧急停止** - 按空格键，所有运动停止
2. **🎮 手动控制** - 使用键盘/手柄控制
3. **🤖 HRL控制** - 基于位置信息的自主决策

## 🔧 故障排除

### ROS2相关问题
```bash
# 安装ROS2
sudo apt install ros-humble-desktop ros-humble-geometry-msgs

# 设置环境
source /opt/ros/humble/setup.bash

# 检查ROS2
ros2 --version
```

### 没有位置数据
1. 检查发布器是否运行
2. 确认话题名称正确
3. 使用 `ros2 topic echo` 验证数据

### HRL策略问题
1. 确认模型文件存在
2. 检查观察空间维度配置
3. 验证动作限制设置

## 📈 性能建议

- 位置更新频率: 10-30 Hz
- HRL推理频率: 每10个控制循环一次
- 确保ROS2通信延迟 < 10ms

## 🎯 下一步

1. 集成真实的视觉系统
2. 添加机器人位置估计
3. 优化HRL策略参数
4. 实现多球/多目标支持

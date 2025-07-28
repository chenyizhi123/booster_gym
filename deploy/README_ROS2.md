# ROS2 位置订阅使用说明

## 概述

位置订阅器现在支持通过ROS2接收足球和球门位置数据。系统会自动订阅指定的ROS2话题，并将位置信息传递给机器人控制逻辑。

## 依赖安装

### 1. 安装ROS2
```bash
# Ubuntu 22.04 (推荐)
sudo apt update
sudo apt install ros-humble-desktop

# 或者使用其他ROS2发行版
```

### 2. 安装必要的ROS2包
```bash
sudo apt install ros-humble-geometry-msgs
```

### 3. 配置环境
```bash
source /opt/ros/humble/setup.bash
# 添加到 ~/.bashrc 中以永久生效
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## 配置文件设置

在 `configs/T1.yaml` 中配置：

```yaml
position_subscriber:
  communication_type: "ros2"  # 使用ROS2通信
  ball_topic: "ball_position"  # 足球位置话题
  goal_topic: "goal_position"  # 球门位置话题
  connection_timeout: 5.0
```

## 消息格式

位置订阅器期望接收 `geometry_msgs/msg/PointStamped` 消息格式：

```yaml
# geometry_msgs/msg/PointStamped
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: "map"
point:
  x: 1.0  # 米
  y: 2.0  # 米
  z: 0.1  # 米
```

## 测试使用

### 1. 启动位置发布器（测试用）
```bash
cd /home/cyz/Workspace/booster_gym/deploy
python test_ros2_publisher.py

# 可选参数：
python test_ros2_publisher.py --ball-topic ball_position --goal-topic goal_position --rate 10.0
```

### 2. 测试位置订阅器
```bash
cd /home/cyz/Workspace/booster_gym/deploy
python test_ros2_subscriber.py
```

### 3. 手动发布位置消息
```bash
# 发布足球位置
ros2 topic pub /ball_position geometry_msgs/msg/PointStamped '{
  "header": {"frame_id": "map"}, 
  "point": {"x": 1.0, "y": 2.0, "z": 0.1}
}'

# 发布球门位置
ros2 topic pub /goal_position geometry_msgs/msg/PointStamped '{
  "header": {"frame_id": "map"}, 
  "point": {"x": 5.0, "y": 0.0, "z": 0.5}
}'
```

## 运行机器人控制器

```bash
cd /home/cyz/Workspace/booster_gym/deploy
python deploy.py --config T1.yaml
```

控制器将自动：
1. 启动ROS2位置订阅器
2. 接收足球和球门位置
3. 计算距离和方向
4. 将数据传递给HRL策略

## 话题监控

查看可用话题：
```bash
ros2 topic list
```

监控位置数据：
```bash
# 监控足球位置
ros2 topic echo /ball_position

# 监控球门位置
ros2 topic echo /goal_position
```

检查话题信息：
```bash
ros2 topic info /ball_position
ros2 topic hz /ball_position  # 检查发布频率
```

## 故障排除

### 1. ROS2未找到
```
ImportError: No module named 'rclpy'
```
解决方案：
- 确保已安装ROS2
- 执行 `source /opt/ros/humble/setup.bash`
- 安装rclpy: `sudo apt install ros-humble-rclpy`

### 2. 消息类型未找到
```
ImportError: No module named 'geometry_msgs.msg'
```
解决方案：
- 安装geometry_msgs: `sudo apt install ros-humble-geometry-msgs`

### 3. 没有接收到数据
- 检查话题名称是否正确
- 确认发布器正在运行
- 使用 `ros2 topic list` 查看可用话题
- 使用 `ros2 topic echo /ball_position` 确认数据发布

### 4. 自动降级到Socket通信
如果ROS2不可用，系统会自动使用UDP socket通信作为备用方案。

## 自定义话题

如果您的系统使用不同的话题名称，可以在配置文件中修改：

```yaml
position_subscriber:
  communication_type: "ros2"
  ball_topic: "your_ball_topic"      # 自定义足球话题
  goal_topic: "your_goal_topic"      # 自定义球门话题
```

## 性能建议

- 建议位置更新频率：10-30 Hz
- 确保网络延迟较低
- 考虑使用局域网进行ROS2通信
- 监控CPU使用率，避免过高的发布频率

## 与其他通信方式的比较

| 通信方式 | 优点 | 缺点 | 使用场景 |
|---------|------|------|----------|
| ROS2 | 标准化、丰富的工具、易于集成 | 需要安装ROS2 | 机器人生态系统 |
| UDP | 简单、快速、无依赖 | 可能丢包 | 简单测试 |
| TCP | 可靠传输 | 较慢 | 需要可靠性 |

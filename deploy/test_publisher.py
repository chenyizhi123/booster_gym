#!/usr/bin/env python3
"""
简单的ROS2位置发布器 - 用于测试
发布移动的球和固定的球门位置
"""
import time
import math

# 尝试导入ROS2
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("❌ ROS2不可用，无法运行发布器")
    exit(1)


class PositionPublisher(Node):
    """位置发布器节点"""
    
    def __init__(self):
        super().__init__('position_publisher')
        
        # 创建发布器
        self.ball_publisher = self.create_publisher(PointStamped, 'ball_position', 10)
        self.goal_publisher = self.create_publisher(PointStamped, 'goal_position', 10)
        
        # 创建定时器 (10Hz)
        self.timer = self.create_timer(0.1, self.publish_positions)
        
        self.start_time = time.time()
        self.counter = 0
        
        self.get_logger().info('🚀 位置发布器已启动')
        self.get_logger().info('📡 发布话题: ball_position, goal_position')
    
    def publish_positions(self):
        """发布位置数据"""
        current_time = time.time() - self.start_time
        
        # 创建移动的球
        ball_x = 2.0 + 1.5 * math.sin(current_time * 0.5)
        ball_y = 1.0 * math.cos(current_time * 0.3)
        ball_z = 0.1
        
        # 发布球位置
        ball_msg = PointStamped()
        ball_msg.header.stamp = self.get_clock().now().to_msg()
        ball_msg.header.frame_id = "map"
        ball_msg.point.x = float(ball_x)
        ball_msg.point.y = float(ball_y)
        ball_msg.point.z = float(ball_z)
        self.ball_publisher.publish(ball_msg)
        
        # 发布固定的球门位置
        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.point.x = 5.0
        goal_msg.point.y = 0.0
        goal_msg.point.z = 0.5
        self.goal_publisher.publish(goal_msg)
        
        self.counter += 1
        
        # 每50次(5秒)打印一次
        if self.counter % 50 == 0:
            self.get_logger().info(
                f'📍 已发布 {self.counter} 条消息 - '
                f'球: [{ball_x:.2f}, {ball_y:.2f}, {ball_z:.2f}]'
            )


def main():
    """主函数"""
    if not ROS2_AVAILABLE:
        return
    
    rclpy.init()
    
    try:
        publisher = PositionPublisher()
        print("🎯 开始发布位置数据...")
        print("📋 球位置: 在圆形轨道上移动")
        print("📋 门位置: 固定在 (5.0, 0.0, 0.5)")
        print("🛑 按 Ctrl+C 停止")
        
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        if 'publisher' in locals():
            publisher.destroy_node()
        rclpy.shutdown()
        print("✅ 发布器已停止")


if __name__ == '__main__':
    main()

"""
简单的位置订阅器 - 订阅ROS2话题获取球和球门位置
"""
import threading
import time
import logging
import math

# 导入ROS2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class PositionSubscriber:
    """位置订阅器类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 位置数据 (使用列表而不是numpy)
        self.ball_position = [0.0, 0.0, 0.0]      # 球位置 [x, y, z]
        self.goal_position = [5.0, 0.0, 0.5]      # 球门位置 [x, y, z]
        
        # 线程安全
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread = None
        
        # ROS2相关
        self.ros2_node = None
        
        self.logger.info("✅ 位置订阅器初始化完成")
    
    def start(self):
        """启动订阅器"""
        if self.running:
            return
        
        self.running = True
        self._start_ros2()
        self.logger.info("🚀 位置订阅器已启动")
    
    def stop(self):
        """停止订阅器"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        if self.ros2_node:
            self.ros2_node.destroy_node()
            self.ros2_node = None
        
        self.logger.info("🛑 位置订阅器已停止")
    
    def _start_ros2(self):
        """启动ROS2订阅"""
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.ros2_node = PositionNode(self._update_positions)
            
            # 在单独线程中运行ROS2节点
            self.worker_thread = threading.Thread(target=self._run_ros2)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            self.logger.info("📡 ROS2订阅器已启动")
        except Exception as e:
            self.logger.error(f"❌ ROS2启动失败: {e}")
            raise
    
    def _run_ros2(self):
        """运行ROS2节点"""
        try:
            rclpy.spin(self.ros2_node)
        except Exception as e:
            if self.running:
                self.logger.error(f"ROS2运行错误: {e}")
    
    def _update_positions(self, ball_pos, goal_pos):
        """更新位置数据的回调函数"""
        with self.lock:
            if ball_pos:
                self.ball_position = ball_pos
            if goal_pos:
                self.goal_position = goal_pos
    
    def get_ball_position(self):
        """获取球位置"""
        with self.lock:
            return self.ball_position.copy()
    
    def get_goal_position(self):
        """获取球门位置"""
        with self.lock:
            return self.goal_position.copy()
    
    def get_ball_distance(self):
        """计算球到机器人的距离（机器人在原点）"""
        with self.lock:
            dx = self.ball_position[0]
            dy = self.ball_position[1]
            return math.sqrt(dx*dx + dy*dy)
    
    def get_goal_distance(self):
        """计算球门到机器人的距离（机器人在原点）"""
        with self.lock:
            dx = self.goal_position[0]
            dy = self.goal_position[1]
            return math.sqrt(dx*dx + dy*dy)
    
    def get_goal_direction(self):
        """计算球门方向（归一化向量，机器人在原点）"""
        with self.lock:
            dx = self.goal_position[0]
            dy = self.goal_position[1]
            dz = self.goal_position[2]
            
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            if length > 1e-6:
                return [dx/length, dy/length, dz/length]
            else:
                return [1.0, 0.0, 0.0]  # 默认向前


class PositionNode(Node):
    """ROS2节点类"""
    
    def __init__(self, callback):
        super().__init__('position_subscriber_node')
        self.callback = callback
        
        # 存储最新位置
        self.latest_ball = None
        self.latest_goal = None
        
        # 创建订阅器
        self.ball_sub = self.create_subscription(
            PointStamped,
            'ball_position',
            self._ball_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            PointStamped,
            'goal_position', 
            self._goal_callback,
            10
        )
        
        self.get_logger().info('📡 ROS2位置节点已创建')
    
    def _ball_callback(self, msg):
        """球位置回调"""
        self.latest_ball = [msg.point.x, msg.point.y, msg.point.z]
        self.callback(self.latest_ball, self.latest_goal)
        self.get_logger().debug(f'收到球位置: {self.latest_ball}')
    
    def _goal_callback(self, msg):
        """球门位置回调"""
        self.latest_goal = [msg.point.x, msg.point.y, msg.point.z]
        self.callback(self.latest_ball, self.latest_goal)
        self.get_logger().debug(f'收到球门位置: {self.latest_goal}')


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 测试位置订阅器")
    
    subscriber = PositionSubscriber()
    subscriber.start()
    
    try:
        for i in range(10):
            time.sleep(1)
            
            ball_pos = subscriber.get_ball_position()
            goal_pos = subscriber.get_goal_position()
            ball_dist = subscriber.get_ball_distance()
            goal_dist = subscriber.get_goal_distance()
            goal_dir = subscriber.get_goal_direction()
            
            print(f"⏰ {i+1}秒:")
            print(f"  🏀 球位置: {ball_pos}")
            print(f"  🥅 门位置: {goal_pos}")
            print(f"  📏 球距离: {ball_dist:.2f}")
            print(f"  📏 门距离: {goal_dist:.2f}")
            print(f"  🧭 门方向: {goal_dir}")
            print()
    
    except KeyboardInterrupt:
        print("🛑 用户中断")
    finally:
        subscriber.stop()
        print("✅ 测试完成")

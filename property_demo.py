import torch

class TeleportDemo:
    def __init__(self):
        self.root_states = torch.tensor([
            [0.0, 0.0, 0.0],  # 机器人0
            [1.0, 1.0, 1.0],  # 球0  
            [5.0, 5.0, 5.0],  # 机器人1
            [6.0, 6.0, 6.0],  # 球1
        ])
        self.robot_indices = torch.tensor([0, 2])  # 机器人在第0和第2行
        
    @property
    def robot_root_states(self):
        print("  -> getter 被调用")
        return self.root_states[self.robot_indices]
    
    @robot_root_states.setter
    def robot_root_states(self, value):
        print("  -> setter 被调用")
        self.root_states[self.robot_indices] = value

print("初始状态:")
demo = TeleportDemo()
print("root_states:", demo.root_states)
print("robot_root_states:", demo.robot_root_states)
print()

# 模拟传送操作
print("=== 方法1: 使用 property + 复合索引 (错误方法) ===")
out_x_min = torch.tensor([True, False])  # 只传送第0个机器人
offset = 10.0
print(f"执行: robot_root_states[{out_x_min}, 0] += {offset}")
demo.robot_root_states[out_x_min, 0] += offset
print("修改后 root_states:", demo.root_states)  # 没有变化！
print()

print("=== 方法2: 直接操作 root_states (正确方法) ===")
print("执行: root_states[robot_indices[out_x_min], 0] += offset")
demo.root_states[demo.robot_indices[out_x_min], 0] += offset
print("修改后 root_states:", demo.root_states)  # 正确变化！
print()

print("=== 方法3: 通过 setter 的正确方式 ===")
print("读取 -> 修改 -> 重新赋值")
temp = demo.robot_root_states  # 调用 getter
temp[out_x_min, 0] += 5.0      # 修改副本
demo.robot_root_states = temp  # 调用 setter
print("修改后 root_states:", demo.root_states) 
import torch

class PropertyDemo:
    def __init__(self):
        print("=== 初始化开始 ===")
        # 这时候 robot_root_states property 已经存在，但还不能使用
        # 因为依赖的数据还没创建
        
        # 创建依赖数据
        self.root_states = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.robot_indices = torch.tensor([0, 2])  # 选择第0和第2行
        print("依赖数据创建完成")
        
        # 现在可以使用 property 了
        print("第一次访问 robot_root_states:")
        print(self.robot_root_states)
        print("=== 初始化结束 ===\n")
    
    @property
    def robot_root_states(self):
        """这个方法在每次访问时都会被调用"""
        print("  -> property getter 被调用了!")
        return self.root_states[self.robot_indices]
    
    @robot_root_states.setter
    def robot_root_states(self, value):
        print("  -> property setter 被调用了!")
        self.root_states[self.robot_indices] = value

# 演示
print("=== 创建类实例 ===")
demo = PropertyDemo()

print("=== 修改 root_states ===")
demo.root_states[0] = torch.tensor([10, 20, 30])
print("修改后再次访问 robot_root_states:")
print(demo.robot_root_states)

print("\n=== 通过 property setter 修改 ===")
demo.robot_root_states = torch.tensor([[100, 200, 300], [700, 800, 900]])
print("修改后的 root_states:")
print(demo.root_states) 
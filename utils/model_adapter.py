import torch
import torch.nn as nn
from typing import Dict, Optional

class ModelAdapter:
    """处理不同输入维度模型之间的权重迁移"""
    
    @staticmethod
    def adapt_actor_weights(old_checkpoint: Dict, 
                          old_obs_dim: int = 47, 
                          new_obs_dim: int = 60,
                          device: str = 'cuda') -> Dict:
        """
        将旧的actor模型权重适配到新的输入维度
        
        Args:
            old_checkpoint: 包含旧模型权重的检查点
            old_obs_dim: 旧模型的观察维度
            new_obs_dim: 新模型的观察维度
            device: 设备
            
        Returns:
            适配后的检查点
        """
        new_checkpoint = old_checkpoint.copy()
        
        if 'actor' in old_checkpoint:
            actor_state = old_checkpoint['actor']
            
            # 找到第一层（通常是 'mlp.0.weight' 或类似的名称）
            first_layer_keys = [k for k in actor_state.keys() if 'weight' in k and '0' in k]
            
            for key in first_layer_keys:
                if actor_state[key].shape[1] == old_obs_dim:
                    # 这是输入层，需要适配
                    old_weight = actor_state[key]  # shape: [hidden_dim, old_obs_dim]
                    
                    # 创建新的权重矩阵
                    new_weight = torch.zeros(old_weight.shape[0], new_obs_dim, device=device)
                    
                    # 复制旧的权重到前47维
                    new_weight[:, :old_obs_dim] = old_weight
                    
                    # 对新增的13维使用小的随机初始化
                    # 这样不会立即影响已学习的行为
                    new_weight[:, old_obs_dim:] = torch.randn(
                        old_weight.shape[0], new_obs_dim - old_obs_dim, device=device
                    ) * 0.01
                    
                    # 更新权重
                    actor_state[key] = new_weight
                    print(f"适配层 {key}: {old_weight.shape} -> {new_weight.shape}")
        
        return new_checkpoint
    
    @staticmethod
    def create_adapter_layer(old_obs_dim: int = 47, 
                           new_obs_dim: int = 60) -> nn.Module:
        """
        创建一个适配层，将新观察映射到旧观察空间
        这是另一种方法：不修改原模型，而是添加一个预处理层
        """
        class ObservationAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                # 可学习的投影矩阵
                self.projection = nn.Linear(new_obs_dim, old_obs_dim)
                
                # 初始化：前47维是恒等映射，后13维映射到0
                with torch.no_grad():
                    self.projection.weight.zero_()
                    self.projection.weight[:, :old_obs_dim] = torch.eye(old_obs_dim)
                    self.projection.bias.zero_()
            
            def forward(self, obs):
                return self.projection(obs)
        
        return ObservationAdapter()

    @staticmethod
    def adapt_critic_weights(old_checkpoint: Dict, 
                          old_obs_dim: int = 47,
                          new_obs_dim: int = 60,
                          old_privileged_obs_dim: int = 14, 
                          new_privileged_obs_dim: int = 20,
                          device: str = 'cuda') -> Dict:
        """
        将旧的critic模型权重适配到新的输入维度
        注意：Critic的输入是观察维度+特权观察维度
        
        Args:
            old_checkpoint: 包含旧模型权重的检查点
            old_obs_dim: 旧模型的观察维度
            new_obs_dim: 新模型的观察维度
            old_privileged_obs_dim: 旧模型的特权观察维度
            new_privileged_obs_dim: 新模型的特权观察维度
            device: 设备
            
        Returns:
            适配后的检查点
        """
        if 'critic' in old_checkpoint:
            critic_state = old_checkpoint['critic']
            
            # Critic的输入维度是obs + privileged_obs
            old_total_dim = old_obs_dim + old_privileged_obs_dim  # 47 + 14 = 61
            new_total_dim = new_obs_dim + new_privileged_obs_dim  # 60 + 20 = 80
            
            # 找到第一层（通常是 'mlp.0.weight' 或类似的名称）
            first_layer_keys = [k for k in critic_state.keys() if 'weight' in k and '0' in k]
            
            for key in first_layer_keys:
                if critic_state[key].shape[1] == old_total_dim:
                    # 这是输入层，需要适配
                    old_weight = critic_state[key]  # shape: [hidden_dim, old_total_dim]
                    
                    # 创建新的权重矩阵
                    new_weight = torch.zeros(old_weight.shape[0], new_total_dim, device=device)
                    
                    # 复制旧的观察部分权重到前old_obs_dim维
                    new_weight[:, :old_obs_dim] = old_weight[:, :old_obs_dim]
                    
                    # 对新增的观察维度（47-60之间的13维）使用小的随机初始化
                    new_weight[:, old_obs_dim:new_obs_dim] = torch.randn(
                        old_weight.shape[0], new_obs_dim - old_obs_dim, device=device
                    ) * 0.01
                    
                    # 复制旧的特权观察部分权重
                    new_weight[:, new_obs_dim:new_obs_dim+old_privileged_obs_dim] = \
                        old_weight[:, old_obs_dim:old_obs_dim+old_privileged_obs_dim]
                    
                    # 对新增的特权观察维度使用小的随机初始化
                    if new_privileged_obs_dim > old_privileged_obs_dim:
                        new_weight[:, new_obs_dim+old_privileged_obs_dim:] = torch.randn(
                            old_weight.shape[0], 
                            new_privileged_obs_dim - old_privileged_obs_dim, 
                            device=device
                        ) * 0.01
                    
                    # 更新权重
                    critic_state[key] = new_weight
                    print(f"适配critic层 {key}: {old_weight.shape} -> {new_weight.shape}")
        
        return old_checkpoint

def load_and_adapt_checkpoint(checkpoint_path: str,
                            old_obs_dim: int = 47,
                            new_obs_dim: int = 60,
                            old_privileged_obs_dim: int = 14,
                            new_privileged_obs_dim: int = 20,
                            device: str = 'cuda') -> Dict:
    """
    加载并适配检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        old_obs_dim: 旧观察维度
        new_obs_dim: 新观察维度
        old_privileged_obs_dim: 旧特权观察维度
        new_privileged_obs_dim: 新特权观察维度
        device: 设备
        
    Returns:
        适配后的检查点
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 适配actor权重
    adapted_checkpoint = ModelAdapter.adapt_actor_weights(
        checkpoint, old_obs_dim, new_obs_dim, device
    )
    
    # 适配critic权重
    adapted_checkpoint = ModelAdapter.adapt_critic_weights(
        adapted_checkpoint, old_obs_dim, new_obs_dim, old_privileged_obs_dim, new_privileged_obs_dim, device
    )
    
    print(f"成功适配检查点：")
    print(f"  Actor: {old_obs_dim}维 -> {new_obs_dim}维")
    print(f"  Critic: {old_obs_dim}维 + {old_privileged_obs_dim}维 -> {new_obs_dim}维 + {new_privileged_obs_dim}维")
    
    return adapted_checkpoint 
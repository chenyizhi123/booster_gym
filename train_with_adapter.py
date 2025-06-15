#!/usr/bin/env python3
"""
模型维度扩展脚本
将 model_approach_best.pth 的维度进行扩展：
- Actor网络第一层：60维 -> 64维
- Critic网络第一层：80维 -> 84维
- 保留原有参数，随机初始化新增参数
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional

class ModelDimensionExpander:
    """模型维度扩展器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def expand_model_dimensions(self, 
                              checkpoint_path: str,
                              new_actor_input_dim: int = 64,
                              new_critic_input_dim: int = 84,
                              output_path: Optional[str] = None) -> Dict:
        """
        扩展模型维度
        
        Args:
            checkpoint_path: 原始模型检查点路径
            new_actor_input_dim: Actor网络新的输入维度
            new_critic_input_dim: Critic网络新的输入维度  
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            扩展后的检查点字典
        """
        print(f"正在加载模型: {checkpoint_path}")
        
        # 加载原始检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model' not in checkpoint:
            raise ValueError("检查点中未找到'model'键")
        
        model_dict = checkpoint['model']
        
        # 检查当前维度
        print("\n=== 当前模型维度 ===")
        actor_keys = [k for k in model_dict.keys() if 'actor.0.weight' in k]
        critic_keys = [k for k in model_dict.keys() if 'critic.0.weight' in k]
        
        if not actor_keys or not critic_keys:
            raise ValueError("未找到actor或critic的第一层权重")
        
        actor_key = actor_keys[0]
        critic_key = critic_keys[0]
        
        current_actor_dim = model_dict[actor_key].shape[1]
        current_critic_dim = model_dict[critic_key].shape[1]
        
        print(f"Actor第一层当前维度: {current_actor_dim}")
        print(f"Critic第一层当前维度: {current_critic_dim}")
        
        # 扩展Actor权重
        print(f"\n=== 扩展Actor维度: {current_actor_dim} -> {new_actor_input_dim} ===")
        self._expand_layer_weights(model_dict, actor_key, new_actor_input_dim)
        
        # 同时处理对应的bias（如果存在）
        actor_bias_key = actor_key.replace('.weight', '.bias')
        if actor_bias_key in model_dict:
            print(f"Actor bias维度保持不变: {model_dict[actor_bias_key].shape}")
        
        # 扩展Critic权重
        print(f"\n=== 扩展Critic维度: {current_critic_dim} -> {new_critic_input_dim} ===")
        self._expand_layer_weights(model_dict, critic_key, new_critic_input_dim)
        
        # 同时处理对应的bias（如果存在）
        critic_bias_key = critic_key.replace('.weight', '.bias')
        if critic_bias_key in model_dict:
            print(f"Critic bias维度保持不变: {model_dict[critic_bias_key].shape}")
        
        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(checkpoint_path)[0]
            output_path = f"{base_name}_expanded_{new_actor_input_dim}d_{new_critic_input_dim}d.pth"
        
        # 保存扩展后的模型
        torch.save(checkpoint, output_path)
        print(f"\n=== 保存成功 ===")
        print(f"扩展后的模型已保存到: {output_path}")
        
        # 验证扩展结果
        self._verify_expansion(output_path, new_actor_input_dim, new_critic_input_dim)
        
        return checkpoint
    
    def _expand_layer_weights(self, model_dict: Dict, weight_key: str, new_input_dim: int):
        """
        扩展指定层的权重维度
        
        Args:
            model_dict: 模型权重字典
            weight_key: 权重键名
            new_input_dim: 新的输入维度
        """
        old_weight = model_dict[weight_key]
        current_input_dim = old_weight.shape[1]
        output_dim = old_weight.shape[0]
        
        if new_input_dim <= current_input_dim:
            print(f"警告: 新维度 {new_input_dim} 不大于当前维度 {current_input_dim}，跳过扩展")
            return
        
        # 创建新的权重矩阵
        new_weight = torch.zeros(output_dim, new_input_dim, dtype=old_weight.dtype, device=self.device)
        
        # 复制原有权重
        new_weight[:, :current_input_dim] = old_weight
        
        # 为新增维度随机初始化参数
        additional_dims = new_input_dim - current_input_dim
        
        # 使用小的随机值初始化新增参数，标准差基于原有权重的标准差
        original_std = old_weight.std().item()
        init_std = min(original_std * 0.1, 0.01)  # 使用较小的初始化标准差
        
        new_weight[:, current_input_dim:] = torch.randn(
            output_dim, additional_dims, dtype=old_weight.dtype, device=self.device
        ) * init_std
        
        # 更新模型字典
        model_dict[weight_key] = new_weight
        
        print(f"✓ 权重扩展: {weight_key}")
        print(f"  形状: {old_weight.shape} -> {new_weight.shape}")
        print(f"  新增参数初始化标准差: {init_std:.6f}")
        print(f"  原有参数标准差: {original_std:.6f}")
    
    def _verify_expansion(self, checkpoint_path: str, expected_actor_dim: int, expected_critic_dim: int):
        """验证扩展结果"""
        print(f"\n=== 验证扩展结果 ===")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = checkpoint['model']
        
        # 检查Actor维度
        actor_keys = [k for k in model_dict.keys() if 'actor.0.weight' in k]
        if actor_keys:
            actual_actor_dim = model_dict[actor_keys[0]].shape[1]
            status = "✓" if actual_actor_dim == expected_actor_dim else "✗"
            print(f"{status} Actor维度: {actual_actor_dim} (期望: {expected_actor_dim})")
        
        # 检查Critic维度
        critic_keys = [k for k in model_dict.keys() if 'critic.0.weight' in k]
        if critic_keys:
            actual_critic_dim = model_dict[critic_keys[0]].shape[1]
            status = "✓" if actual_critic_dim == expected_critic_dim else "✗"
            print(f"{status} Critic维度: {actual_critic_dim} (期望: {expected_critic_dim})")
    
    def show_model_info(self, checkpoint_path: str):
        """显示模型信息"""
        print(f"\n=== 模型信息: {checkpoint_path} ===")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = checkpoint['model']
        
        print("Actor网络层:")
        for key in sorted(model_dict.keys()):
            if 'actor' in key:
                print(f"  {key}: {model_dict[key].shape}")
        
        print("\nCritic网络层:")
        for key in sorted(model_dict.keys()):
            if 'critic' in key:
                print(f"  {key}: {model_dict[key].shape}")
        
        # 统计参数总数
        total_params = sum(p.numel() for p in model_dict.values())
        print(f"\n总参数数量: {total_params:,}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='扩展模型维度')
    parser.add_argument('--input', '-i', type=str, default='model_approach_best.pth',
                      help='输入模型文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='输出模型文件路径（自动生成如果未指定）')
    parser.add_argument('--actor-dim', type=int, default=64,
                      help='Actor网络新的输入维度')
    parser.add_argument('--critic-dim', type=int, default=84,
                      help='Critic网络新的输入维度')
    parser.add_argument('--device', type=str, default='cpu',
                      help='计算设备')
    parser.add_argument('--show-info', action='store_true',
                      help='只显示模型信息不进行扩展')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    # 创建扩展器
    expander = ModelDimensionExpander(device=args.device)
    
    if args.show_info:
        # 只显示模型信息
        expander.show_model_info(args.input)
    else:
        # 执行维度扩展
        print("=== 模型维度扩展工具 ===")
        print(f"输入文件: {args.input}")
        print(f"目标Actor维度: {args.actor_dim}")
        print(f"目标Critic维度: {args.critic_dim}")
        print(f"计算设备: {args.device}")
        
        try:
            expander.expand_model_dimensions(
                checkpoint_path=args.input,
                new_actor_input_dim=args.actor_dim,
                new_critic_input_dim=args.critic_dim,
                output_path=args.output
            )
            print("\n✓ 维度扩展完成！")
        except Exception as e:
            print(f"\n✗ 扩展失败: {e}")

if __name__ == "__main__":
    main() 
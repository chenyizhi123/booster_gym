import os
import torch
import argparse
from utils.model import ActorCritic

def migrate_model(source_checkpoint, target_checkpoint, 
                 source_obs_dim=47, source_priv_obs_dim=14,
                 target_obs_dim=60, target_priv_obs_dim=20,
                 num_actions=12):
    """
    将一个Phase训练的模型迁移到另一个Phase，处理观察空间维度的变化。
    主要是保留权重参数，但调整输入层大小。
    
    参数:
    - source_checkpoint: 源模型检查点路径
    - target_checkpoint: 目标模型保存路径
    - source_obs_dim: 源模型的观察空间维度
    - source_priv_obs_dim: 源模型的特权观察空间维度
    - target_obs_dim: 目标模型的观察空间维度
    - target_priv_obs_dim: 目标模型的特权观察空间维度
    - num_actions: 动作空间维度
    """
    print(f"正在将模型从维度 ({source_obs_dim}, {source_priv_obs_dim}) 迁移到 ({target_obs_dim}, {target_priv_obs_dim})")
    
    # 加载源模型
    print(f"加载源模型: {source_checkpoint}")
    checkpoint = torch.load(source_checkpoint, map_location='cpu')
    source_state_dict = checkpoint["model"]
    
    # 创建目标模型
    target_model = ActorCritic(num_actions, target_obs_dim, target_priv_obs_dim)
    target_state_dict = target_model.state_dict()
    
    # 创建新的权重字典
    new_state_dict = {}
    
    # 复制参数，排除第一层（输入层）
    # 输入层需要重新初始化因为维度不同
    skip_layers = ['actor.0.weight', 'actor.0.bias', 
                  'critic.0.weight', 'critic.0.bias']
    
    print("迁移参数...")
    for name, param in source_state_dict.items():
        if name in skip_layers:
            print(f"跳过输入层: {name}")
            # 保持目标模型的默认初始化值
            new_state_dict[name] = target_state_dict[name]
            continue
        
        # 复制其他所有层的参数
        print(f"复制参数: {name}, 形状: {param.shape}")
        new_state_dict[name] = param
    
    # 使用新的状态字典更新目标模型
    target_model.load_state_dict(new_state_dict)
    
    # 准备新的检查点
    new_checkpoint = {
        "model": target_model.state_dict(),
        "optimizer": checkpoint.get("optimizer", None),  # 可能不需要优化器
        "curriculum": checkpoint.get("curriculum", None)
    }
    
    # 保存新模型
    print(f"保存迁移后的模型到: {target_checkpoint}")
    torch.save(new_checkpoint, target_checkpoint)
    print("迁移完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型迁移工具 - 处理不同阶段间的观察空间变化")
    parser.add_argument("--source", required=True, help="源模型检查点路径")
    parser.add_argument("--target", required=True, help="目标模型保存路径")
    parser.add_argument("--source_obs_dim", type=int, default=47, help="源模型的观察空间维度")
    parser.add_argument("--source_priv_obs_dim", type=int, default=14, help="源模型的特权观察空间维度")
    parser.add_argument("--target_obs_dim", type=int, default=60, help="目标模型的观察空间维度")
    parser.add_argument("--target_priv_obs_dim", type=int, default=20, help="目标模型的特权观察空间维度")
    parser.add_argument("--num_actions", type=int, default=12, help="动作空间维度")
    
    args = parser.parse_args()
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(args.target), exist_ok=True)
    
    migrate_model(
        args.source, 
        args.target,
        args.source_obs_dim,
        args.source_priv_obs_dim,
        args.target_obs_dim,
        args.target_priv_obs_dim,
        args.num_actions
    ) 
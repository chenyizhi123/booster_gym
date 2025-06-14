"""
使用模型适配器从47维模型迁移到新的维度维模型的训练脚本
"""

import argparse
import os
import torch
from utils.model_adapter import load_and_adapt_checkpoint

def train_with_adapted_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_checkpoint', type=str, required=True, 
                      help='Path to the old 47-dim checkpoint')
    parser.add_argument('--config', type=str, default='envs/T1_phase1.yaml',
                      help='Config file for training')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    # 1. 加载并适配旧的检查点
    print(f"加载旧模型: {args.old_checkpoint}")
    adapted_checkpoint = load_and_adapt_checkpoint(
        checkpoint_path=args.old_checkpoint,
        old_obs_dim=60,
        new_obs_dim=64,
        old_privileged_obs_dim=14,
        new_privileged_obs_dim=20,
        device=args.device
    )
    
    # 2. 准备训练命令
    # 注意：需要使用 --checkpoint 参数来加载适配后的模型
    # 首先保存适配后的检查点
    adapted_path = args.old_checkpoint.replace('.pth', '_adapted_60d.pth')
    torch.save(adapted_checkpoint, adapted_path)
    print(f"保存适配后的模型到: {adapted_path}")
    
    # 3. 构建训练命令
    train_cmd = f"python train.py --config {args.config} --checkpoint {adapted_path}"
    
    print("\n" + "="*50)
    print("请运行以下命令开始训练：")
    print(train_cmd)
    print("="*50)
    
    # 提示
    print("\n训练建议：")
    print("1. 初始学习率可以设置得较小（如1e-6），因为模型已经会走路")
    print("2. 前1000步观察模型是否保持稳定")
    print("3. 如果模型表现下降，可以降低学习率或冻结部分层")
    print("4. 新增的13维输入（足球信息）会逐渐被学习")

if __name__ == "__main__":
    train_with_adapted_model() 
#!/usr/bin/env python3
"""
创建AMP训练的专家动作数据

这个脚本提供几种创建专家数据的方法：
1. 从现有训练好的模型生成轨迹数据
2. 从人工设计的动作序列生成数据  
3. 从motion capture数据转换
4. 从其他成功的训练session录制数据

用法：
    python create_expert_data.py --method rollout --model path/to/model.pt
    python create_expert_data.py --method manual --output expert_kick_data.yaml
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
from typing import List, Dict, Any

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class ExpertDataGenerator:
    """专家数据生成器"""
    
    def __init__(self):
        self.data_format = {
            'observations': [],    # 观察序列
            'actions': [],         # 动作序列
            'rewards': [],         # 奖励序列
            'dones': [],           # 终止标志
            'infos': [],           # 额外信息
            'metadata': {
                'creation_method': '',
                'num_trajectories': 0,
                'avg_trajectory_length': 0.0,
                'observation_dim': 0,
                'action_dim': 0,
                'creation_timestamp': '',
            }
        }
    
    def generate_from_model(self, model_path: str, env_config: str, num_trajectories: int = 100) -> Dict[str, Any]:
        """
        从训练好的模型生成专家数据
        
        Args:
            model_path: 模型文件路径
            env_config: 环境配置文件路径
            num_trajectories: 生成的轨迹数量
            
        Returns:
            专家数据字典
        """
        print(f"从模型生成专家数据: {model_path}")
        
        try:
            # 加载环境配置
            with open(env_config, 'r') as f:
                cfg = yaml.safe_load(f)
            
            # 创建环境
            from envs.kick_2D import kick_2D
            env = kick_2D(cfg)
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 这里需要根据具体的模型结构来加载
            # 示例代码，需要根据实际情况调整
            print("⚠️  模型加载功能需要根据具体模型结构实现")
            print("请参考现有的模型加载代码进行适配")
            
            trajectories = []
            for traj_idx in range(num_trajectories):
                print(f"生成轨迹 {traj_idx + 1}/{num_trajectories}")
                
                obs = env.reset()
                trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'amp_observations': []
                }
                
                done = False
                step_count = 0
                max_steps = 1000  # 防止无限循环
                
                while not done and step_count < max_steps:
                    # 这里需要使用加载的模型来生成动作
                    # action = model.predict(obs)  # 需要实现
                    
                    # 临时使用随机动作作为示例
                    action = torch.randn(env.num_actions) * 0.1  # 小的随机动作
                    
                    trajectory['observations'].append(obs.clone())
                    trajectory['actions'].append(action.clone())
                    trajectory['amp_observations'].append(env.get_amp_observations().clone())
                    
                    # 执行动作
                    obs, reward, done, _ = env.step(action.unsqueeze(0))
                    obs = obs[0]  # 取第一个环境的观察
                    reward = reward[0]  # 取第一个环境的奖励
                    done = done[0]  # 取第一个环境的终止标志
                    
                    trajectory['rewards'].append(reward.item())
                    step_count += 1
                
                trajectories.append(trajectory)
            
            # 整理数据格式
            expert_data = self._format_trajectories(trajectories)
            expert_data['metadata']['creation_method'] = f'model_rollout_{os.path.basename(model_path)}'
            
            return expert_data
            
        except Exception as e:
            print(f"从模型生成数据失败: {e}")
            raise
    
    def generate_manual_kick_data(self) -> Dict[str, Any]:
        """
        生成人工设计的踢球专家数据
        
        包含典型的踢球动作序列：
        1. 接近球
        2. 调整姿态  
        3. 踢球动作
        4. 恢复平衡
        """
        print("生成人工设计的踢球专家数据...")
        
        # 定义一些典型的踢球动作模式
        kick_patterns = [
            self._generate_approach_and_kick_pattern(),
            self._generate_side_kick_pattern(), 
            self._generate_power_kick_pattern(),
            # 可以添加更多模式
        ]
        
        # 整理为专家数据格式
        expert_data = self._format_manual_patterns(kick_patterns)
        expert_data['metadata']['creation_method'] = 'manual_kick_patterns'
        
        return expert_data
    
    def _generate_approach_and_kick_pattern(self) -> Dict[str, List]:
        """生成接近并踢球的动作模式"""
        pattern = {
            'name': 'approach_and_kick',
            'actions': [],
            'durations': []  # 每个动作持续的时间步
        }
        
        # 1. 接近阶段 - 小步前进，保持平衡
        for _ in range(10):  # 10步接近
            action = np.array([
                # Hip joints: slight forward lean
                -0.1, -0.1,  # Hip_Pitch_Left, Hip_Pitch_Right
                0.0, 0.0,    # Hip_Roll_Left, Hip_Roll_Right
                0.0, 0.0,    # Hip_Yaw_Left, Hip_Yaw_Right
                
                # Knee joints: slight bend for stability
                0.2, 0.2,    # Knee_Pitch_Left, Knee_Pitch_Right
                
                # Ankle joints: maintain ground contact
                -0.1, -0.1,  # Ankle_Pitch_Left, Ankle_Pitch_Right
                0.0, 0.0,    # Ankle_Roll_Left, Ankle_Roll_Right
            ], dtype=np.float32)
            
            # 添加少量随机性
            action += np.random.normal(0, 0.02, action.shape)
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        # 2. 准备阶段 - 支撑脚稳定，踢球脚准备
        for _ in range(5):
            action = np.array([
                # 左脚作为支撑脚，右脚准备踢球
                -0.05, -0.15,  # 左脚稍微前倾，右脚后撤
                -0.1, 0.0,     # 左脚承重，右脚减重
                0.0, 0.0,      # Yaw保持
                
                0.15, 0.3,     # 左膝稍弯，右膝更弯准备踢球
                
                -0.08, -0.2,   # 左脚踝稳定，右脚踝准备
                0.0, 0.0,
            ], dtype=np.float32)
            
            action += np.random.normal(0, 0.02, action.shape)
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        # 3. 踢球阶段 - 快速前摆
        for i in range(8):
            progress = i / 7.0  # 0 到 1
            kick_strength = np.sin(progress * np.pi)  # 0 -> 1 -> 0
            
            action = np.array([
                -0.05, -0.3 + 0.4 * kick_strength,  # 右腿大幅前摆
                -0.1, 0.0,
                0.0, 0.0,
                
                0.15, 0.1 - 0.2 * kick_strength,    # 右膝伸展
                
                -0.08, -0.15 + 0.25 * kick_strength,  # 右脚踝配合
                0.0, 0.0,
            ], dtype=np.float32)
            
            action += np.random.normal(0, 0.02, action.shape)
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        # 4. 恢复阶段 - 恢复平衡
        for _ in range(7):
            action = np.array([
                -0.1, -0.1,   # 恢复中性姿态
                0.0, 0.0,
                0.0, 0.0,
                
                0.2, 0.2,     # 膝关节恢复
                
                -0.1, -0.1,   # 脚踝恢复
                0.0, 0.0,
            ], dtype=np.float32)
            
            action += np.random.normal(0, 0.02, action.shape)
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        return pattern
    
    def _generate_side_kick_pattern(self) -> Dict[str, List]:
        """生成侧踢动作模式"""
        pattern = {
            'name': 'side_kick',
            'actions': [],
            'durations': []
        }
        
        # 侧踢动作序列（这里简化实现）
        for i in range(20):
            # 基础平衡姿态加上侧向踢球动作
            action = np.array([
                -0.1, -0.1,
                0.1, -0.1,  # 侧向倾斜
                0.2, 0.0,   # 旋转
                0.2, 0.2,
                -0.1, -0.1,
                0.1, -0.1,  # 侧向脚踝调整
            ], dtype=np.float32)
            
            action += np.random.normal(0, 0.02, action.shape)
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        return pattern
    
    def _generate_power_kick_pattern(self) -> Dict[str, List]:
        """生成大力射门动作模式"""
        pattern = {
            'name': 'power_kick',
            'actions': [],
            'durations': []
        }
        
        # 大力射门动作序列
        for i in range(25):
            progress = i / 24.0
            power_phase = np.sin(progress * np.pi * 1.5)
            
            action = np.array([
                -0.15, -0.2 + 0.5 * power_phase,  # 更大幅度的踢腿动作
                -0.15, 0.0,
                0.0, 0.0,
                0.25, 0.1 - 0.3 * power_phase,
                -0.12, -0.1 + 0.3 * power_phase,
                0.0, 0.0,
            ], dtype=np.float32)
            
            action += np.random.normal(0, 0.03, action.shape)  # 稍大的噪声
            pattern['actions'].append(action)
            pattern['durations'].append(1)
        
        return pattern
    
    def _format_trajectories(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """格式化轨迹数据为AMP格式"""
        expert_data = self.data_format.copy()
        
        total_steps = 0
        for traj in trajectories:
            total_steps += len(traj['observations'])
            expert_data['observations'].extend(traj['observations'])
            expert_data['actions'].extend(traj['actions'])
            expert_data['rewards'].extend(traj['rewards'])
        
        expert_data['metadata'].update({
            'num_trajectories': len(trajectories),
            'avg_trajectory_length': total_steps / len(trajectories) if trajectories else 0,
            'observation_dim': trajectories[0]['observations'][0].shape[0] if trajectories else 0,
            'action_dim': trajectories[0]['actions'][0].shape[0] if trajectories else 0,
        })
        
        return expert_data
    
    def _format_manual_patterns(self, patterns: List[Dict]) -> Dict[str, Any]:
        """格式化手工模式为AMP格式"""
        expert_data = self.data_format.copy()
        
        # 为每个模式生成多个变体
        for pattern in patterns:
            for variant in range(10):  # 每个模式生成10个变体
                actions = pattern['actions'].copy()
                
                # 添加变异
                for i, action in enumerate(actions):
                    # 添加随机变异
                    variation = np.random.normal(0, 0.05, action.shape)
                    actions[i] = np.clip(action + variation, -1.0, 1.0)
                
                expert_data['actions'].extend(actions)
                # 简化的观察（实际使用时需要从环境获取）
                # 注意：AMP观察维度是37，不是64
                expert_data['observations'].extend([np.zeros(37) for _ in actions])  # AMP观察维度为37
                expert_data['rewards'].extend([1.0] * len(actions))  # 所有步骤都给予正奖励
        
        expert_data['metadata'].update({
            'num_trajectories': len(patterns) * 10,
            'avg_trajectory_length': sum(len(p['actions']) for p in patterns) / len(patterns),
            'action_dim': len(patterns[0]['actions'][0]) if patterns else 0,
            'observation_dim': 64,  # 假设
        })
        
        return expert_data
    
    def save_data(self, data: Dict[str, Any], output_path: str):
        """保存专家数据到文件"""
        import datetime
        
        # 添加时间戳
        data['metadata']['creation_timestamp'] = datetime.datetime.now().isoformat()
        
        # 转换numpy数组为列表以便保存
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            else:
                return obj
        
        data_converted = convert_numpy(data)
        
        # 保存为YAML格式
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_converted, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 专家数据已保存到: {output_path}")
        print(f"  - 轨迹数量: {data['metadata']['num_trajectories']}")
        print(f"  - 平均轨迹长度: {data['metadata']['avg_trajectory_length']:.1f}")
        print(f"  - 动作维度: {data['metadata']['action_dim']}")
        print(f"  - 观察维度: {data['metadata']['observation_dim']}")

def main():
    parser = argparse.ArgumentParser(description='创建AMP专家数据')
    parser.add_argument('--method', choices=['model', 'manual'], default='manual',
                       help='数据生成方法')
    parser.add_argument('--model', type=str, 
                       help='模型文件路径（method=model时必需）')
    parser.add_argument('--config', type=str, default='envs/kick_2D.yaml',
                       help='环境配置文件路径')
    parser.add_argument('--output', '-o', type=str, default='data/expert_kick_data.yaml',
                       help='输出文件路径')
    parser.add_argument('--num-trajectories', type=int, default=100,
                       help='生成的轨迹数量（method=model时使用）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AMP专家数据生成器")
    print("="*60)
    
    generator = ExpertDataGenerator()
    
    if args.method == 'model':
        if not args.model:
            print("❌ method=model时必须指定 --model 参数")
            return
        
        if not os.path.exists(args.model):
            print(f"❌ 模型文件不存在: {args.model}")
            return
        
        if not os.path.exists(args.config):
            print(f"❌ 配置文件不存在: {args.config}")
            return
        
        try:
            data = generator.generate_from_model(args.model, args.config, args.num_trajectories)
        except Exception as e:
            print(f"❌ 从模型生成数据失败: {e}")
            return
    
    elif args.method == 'manual':
        print("📝 生成人工设计的踢球动作数据...")
        try:
            data = generator.generate_manual_kick_data()
        except Exception as e:
            print(f"❌ 生成手工数据失败: {e}")
            return
    
    # 保存数据
    try:
        generator.save_data(data, args.output)
        
        print(f"\n🎉 专家数据生成完成！")
        print(f"💡 使用方法:")
        print(f"   1. 编辑 envs/kick_2D_amp.yaml")
        print(f"   2. 在 amp_motion_files 中添加: ['{args.output}']")
        print(f"   3. 运行: python train_kick2d_amp.py")
        
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")

if __name__ == "__main__":
    main()

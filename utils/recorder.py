import torch
from torch.utils.tensorboard.writer import SummaryWriter
import os
import time
import yaml
import numpy as np


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.dir = os.path.join("logs", name)
        os.makedirs(self.dir)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        
        # 只使用tensorboard
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        print(f"📊 TensorBoard日志保存到: {os.path.join(self.dir, 'summaries')}")
        print(f"💾 模型文件保存到: {self.model_dir}")

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=torch.int32)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)

    def record_gradients(self, model, it, prefix=""):
        """记录模型梯度信息到TensorBoard
        
        Args:
            model: 要记录梯度的模型
            it: 迭代步数
            prefix: 前缀，用于区分不同的模型
        """
        if prefix:
            prefix = prefix + "/"
        
        # 收集所有梯度
        gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                gradients.append(grad.flatten())
                
                # 检查是否有NaN或inf
                has_nan = torch.isnan(grad).any()
                has_inf = torch.isinf(grad).any()
                
                # 记录到tensorboard
                self.writer.add_scalar(f"{prefix}gradients/has_nan/{name}", float(has_nan), it)
                self.writer.add_scalar(f"{prefix}gradients/has_inf/{name}", float(has_inf), it)
                
                # 计算梯度统计信息
                grad_norm = torch.norm(grad).item()
                grad_mean = torch.mean(grad).item()
                grad_std = torch.std(grad).item()
                grad_min = torch.min(grad).item()
                grad_max = torch.max(grad).item()
                
                # 记录每层的梯度信息
                layer_name = name.replace('.', '/')
                self.writer.add_scalar(f"{prefix}gradients/norm/{layer_name}", grad_norm, it)
                self.writer.add_scalar(f"{prefix}gradients/mean/{layer_name}", grad_mean, it)
                self.writer.add_scalar(f"{prefix}gradients/std/{layer_name}", grad_std, it)
                self.writer.add_scalar(f"{prefix}gradients/min/{layer_name}", grad_min, it)
                self.writer.add_scalar(f"{prefix}gradients/max/{layer_name}", grad_max, it)
                
                # 记录梯度分布直方图
                self.writer.add_histogram(f"{prefix}gradients/histogram/{layer_name}", grad, it)
        
        # 计算全局梯度统计信息
        if gradients:
            all_gradients = torch.cat(gradients)
            
            # 全局梯度范数
            global_grad_norm = torch.norm(all_gradients).item()
            self.writer.add_scalar(f"{prefix}gradients/global_norm", global_grad_norm, it)
            
            # 全局梯度统计
            self.writer.add_scalar(f"{prefix}gradients/global_mean", torch.mean(all_gradients).item(), it)
            self.writer.add_scalar(f"{prefix}gradients/global_std", torch.std(all_gradients).item(), it)
            self.writer.add_scalar(f"{prefix}gradients/global_min", torch.min(all_gradients).item(), it)
            self.writer.add_scalar(f"{prefix}gradients/global_max", torch.max(all_gradients).item(), it)
            
            # 检查全局NaN/Inf
            global_has_nan = torch.isnan(all_gradients).any()
            global_has_inf = torch.isinf(all_gradients).any()
            
            self.writer.add_scalar(f"{prefix}gradients/global_has_nan", float(global_has_nan), it)
            self.writer.add_scalar(f"{prefix}gradients/global_has_inf", float(global_has_inf), it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("💾 保存模型到: {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def close(self):
        """关闭tensorboard writer"""
        self.writer.close()
        print("📊 TensorBoard日志已关闭")

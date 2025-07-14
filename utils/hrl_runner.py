import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio

# 先导入envs（包含isaacgym）
from envs import *

# 然后导入torch相关模块
import torch
import torch.nn.functional as F
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder



class HRLRunner:
    """HRL专用训练器"""

    def __init__(self, test=False):
        self.test = test
        # 准备环境
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        
        # 创建HRL环境
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        
        # 创建高层策略的ActorCritic网络 (需要训练)
        from utils.model import ActorCritic
        self.model = ActorCritic(
            self.env.num_actions,  # 3维：[ΔVx, ΔVy, ΔWz]
            self.env.num_obs,      # 14维：高层观察
            self.env.num_privileged_obs  # 20维：特权观察
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()

        # 创建经验缓冲区
        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        """从命令行参数更新配置"""
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        """设置随机种子"""
        seed = self.cfg["basic"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _load(self):
        """加载检查点"""
        if self.cfg["basic"]["checkpoint"] and self.cfg["basic"]["checkpoint"] != "-1":
            checkpoint_path = self.cfg["basic"]["checkpoint"]
            if not os.path.exists(checkpoint_path):
                checkpoint_path = sorted(
                    glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), 
                    key=os.path.getmtime
                )[-1]
            print(f"📂 加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self):
        """HRL训练主循环"""
        self.recorder = Recorder(self.cfg)
        print("\n🚀 开始HRL训练...")
        print(f"📊 高层策略频率: 5Hz (每10帧更新)")
        print(f"📊 低层控制频率: 50Hz (每帧执行)")
        print(f"📈 要查看训练进度，请打开新终端并运行:")
        print(f"   tensorboard --logdir=logs")
        print(f"   然后在浏览器中打开: http://localhost:6006")
        print("-" * 60)
        
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        
        try:
            for it in range(self.cfg["basic"]["max_iterations"]):
                # 收集经验
                for n in range(self.cfg["runner"]["horizon_length"]):
                    self.buffer.update_data("obses", n, obs)
                    self.buffer.update_data("privileged_obses", n, privileged_obs)
                    
                    with torch.no_grad():
                        dist = self.model.act(obs)
                        # 采样动作（3维速度增量）
                        act = dist.sample()    
                    # 执行动作
                    obs, rew, done, infos = self.env.step(act)
                    obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                    privileged_obs = infos["privileged_obs"].to(self.device)
                    
                    # 存储经验
                    self.buffer.update_data("actions", n, act)
                    self.buffer.update_data("rewards", n, rew)
                    self.buffer.update_data("dones", n, done)
                    self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                    
                    # 记录episode信息
                    ep_info = {"reward": rew}
                    ep_info.update(infos["rew_terms"])
                    self.recorder.record_episode_statistics(
                        done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1)
                    )

                # 计算旧动作的对数概率
                with torch.no_grad():
                    old_dist = self.model.act(self.buffer["obses"])
                    old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

                # PPO更新
                mean_value_loss = 0
                mean_actor_loss = 0
                mean_bound_loss = 0
                mean_entropy = 0
                
                for n in range(self.cfg["runner"]["mini_epochs"]):
                    # 计算价值和优势
                    values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                    last_values = self.model.est_value(obs, privileged_obs)
                    
                    with torch.no_grad():
                        self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                        advantages = discount_values(
                            self.buffer["rewards"],
                            self.buffer["dones"] | self.buffer["time_outs"],
                            values,
                            last_values,
                            self.cfg["algorithm"]["gamma"],
                            self.cfg["algorithm"]["lam"],
                        )
                        returns = values + advantages
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # 价值函数损失
                    value_loss = F.mse_loss(values, returns)

                    # 策略损失
                    dist = self.model.act(self.buffer["obses"])
                    actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                    actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

                    # 边界损失（保持动作在合理范围内）
                    bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + \
                                torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                    # 熵损失
                    entropy = dist.entropy().sum(dim=-1)

                    # 总损失
                    loss = (
                        value_loss
                        + actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    )
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度监控 - 每10个训练迭代记录一次
                    if it % 10 == 0:  # 只在第一个mini_epoch记录以减少开销
                        self.recorder.record_gradients(self.model, it, prefix="model")
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # KL散度自适应学习率
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(dist.scale / old_dist.scale)
                            + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                            - 0.5,
                            dim=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                    mean_value_loss += value_loss.item()
                    mean_actor_loss += actor_loss.item()
                    mean_bound_loss += bound_loss.item()
                    mean_entropy += entropy.mean()
                
                # 平均损失
                mean_value_loss /= self.cfg["runner"]["mini_epochs"]
                mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
                mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
                mean_entropy /= self.cfg["runner"]["mini_epochs"]
                
                # 记录统计信息
                stats = {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                }
                
                # HRL特定统计信息
                if hasattr(self.env, 'curriculum_level'):
                    curriculum_distribution = torch.bincount(self.env.curriculum_level)
                    for i, count in enumerate(curriculum_distribution):
                        stats[f"curriculum/level_{i}_count"] = count.item()
                
                self.recorder.record_statistics(stats, it)

                # 保存检查点
                if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                    save_dict = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    if hasattr(self.env, 'curriculum_level'):
                        save_dict["curriculum_level"] = self.env.curriculum_level
                    
                    self.recorder.save(save_dict, it + 1)
                
                print("📊 HRL训练进度: {}/{} | TensorBoard: http://localhost:6006".format(
                    it + 1, self.cfg["basic"]["max_iterations"]
                ))
                
        except KeyboardInterrupt:
            print("\n⏹️  HRL训练被用户中断")
        except Exception as e:
            print(f"\n❌ HRL训练出错: {e}")
            raise
        finally:
            self.recorder.close()
            print("✅ HRL训练结束")

    def play(self):
        """HRL测试模式"""
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("hrl_%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc  # 使用均值而不是采样
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True 
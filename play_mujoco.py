import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.model import *


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu")
    model.load_state_dict(model_dict["model"])

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            ball_pos = mj_data.sensor("ball_position").data.astype(np.float32)
            ball_vel = mj_data.sensor("ball_velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                
                # 原有的47维观察空间
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[35:47] = actions
                
                # 新增的17维观察空间 (47:64) - 与kick.py保持一致
                # 获取机器人位置和四元数
                robot_pos = mj_data.qpos[:3].astype(np.float32)  # 机器人位置
                robot_quat = mj_data.qpos[3:7].astype(np.float32)  # 机器人四元数 [x,y,z,w]
                
                # 1. 球相对于机器人的位置 (局部坐标系) - ball_local_position (3维)
                ball_rel_pos = ball_pos - robot_pos
                ball_rel_pos_local = quat_rotate_inverse(robot_quat, ball_rel_pos)
                obs[47:50] = ball_rel_pos_local
                
                # 2. 球相对于机器人的速度 (局部坐标系) - ball_local_velocity (3维)
                # 这里需要计算球相对于机器人的速度，而不是球的绝对速度
                robot_vel = mj_data.qvel[:3].astype(np.float32)  # 机器人线速度
                ball_rel_vel = ball_vel - robot_vel
                ball_rel_vel_local = quat_rotate_inverse(robot_quat, ball_rel_vel)
                obs[50:53] = ball_rel_vel_local * cfg["normalization"]["lin_vel"]
                
                # 3. 球门方向 (局部坐标系) - goal_dir_relative (3维)
                goal_pos = np.array([0.0, 4.5, 0.4], dtype=np.float32)
                goal_rel_pos = goal_pos - robot_pos
                goal_distance = np.linalg.norm(goal_rel_pos[:2]) + 1e-8
                goal_direction_world = goal_rel_pos / goal_distance  # 单位向量
                goal_dir_local = quat_rotate_inverse(robot_quat, goal_direction_world)
                obs[53:56] = goal_dir_local
                
                # 4. 机器人朝向角度 - heading_angle (1维)
                # 机器人前进方向（局部坐标系x轴）与球门方向的夹角
                robot_forward_local = np.array([1.0, 0.0, 0.0])
                cos_angle = np.dot(robot_forward_local, goal_dir_local)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                heading_angle = np.arccos(cos_angle)
                obs[56] = heading_angle
                
                # 5. 球到球门向量 (世界坐标系) - ball_to_goal_vec (3维)
                ball_to_goal_vec = goal_pos - ball_pos
                obs[57:60] = ball_to_goal_vec
                
                # 6. 射门指令 - shooting_command (1维)
                # 简化处理，设为0（左上角）
                obs[60] = 0
                
                # 7. 当前目标点（局部坐标系） - active_target_points (3维)
                # 使用球门中心作为目标点
                target_rel_pos = goal_pos - robot_pos
                target_rel_pos_local = quat_rotate_inverse(robot_quat, target_rel_pos)
                obs[61:64] = target_rel_pos_local
                
                dist = model.act(torch.tensor(obs).unsqueeze(0))
                actions[:] = dist.loc.detach().numpy()
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)

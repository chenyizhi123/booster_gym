import numpy as np
import mujoco
import mujoco.viewer
import yaml

def main():
    # 从配置文件读取初始状态
    with open("envs/kick_2D.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 加载MuJoCo模型
    mj_model = mujoco.MjModel.from_xml_path("resources/T1/T1.xml")
    mj_data = mujoco.MjData(mj_model)
    
    # 重置数据
    mujoco.mj_resetData(mj_model, mj_data)
    
    # 设置默认关节角度
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        actuator_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in actuator_name:
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
                print(f"设置关节 {actuator_name}: {default_dof_pos[i]} rad")
                break
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]
            print(f"使用默认值设置关节 {actuator_name}: {default_dof_pos[i]} rad")
    
    # 设置机器人初始位置和姿态
    mj_data.qpos = np.concatenate([
        np.array(cfg["init_state"]["pos"], dtype=np.float32),  # 位置
        np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),  # 四元数 [w,x,y,z]
        default_dof_pos,  # 关节角度
    ])
    
    # 前向计算更新机器人状态
    mujoco.mj_forward(mj_model, mj_data)
    
    print(f"\n机器人初始化完成:")
    print(f"位置: {cfg['init_state']['pos']}")
    print(f"旋转: {cfg['init_state']['rot']}")
    print(f"总关节数: {mj_model.nu}")
    
    # 启动可视化器
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("\n可视化器已启动，按任意键退出...")
        viewer.cam.distance = 3.0  # 设置相机距离
        viewer.cam.elevation = -20  # 设置相机仰角
        viewer.cam.azimuth = 45     # 设置相机方位角
        
        # 保持可视化器运行
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
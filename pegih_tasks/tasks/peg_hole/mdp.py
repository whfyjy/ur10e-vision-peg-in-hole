"""UR10e 视觉插销（peg-in-hole）任务的 MDP 函数。

坐标系（与 genpih 完全一致，我们原样复用它的网格）：
  - peg 连杆坐标系 == peg 尖端坐标系（从 tool0 经固定关节连接，沿 tool0 的 z 轴 +0.120 m）
  - 孔坐标系原点 == 孔板**底面**上的孔心，孔轴沿 +z

与 genpih 的场景差异（这正是本项目存在的意义）：
  孔板放在一块实心木板上，木板顶面位于 BOARD_TOP_Z，于是孔坐标系原点恰好落在木板表面上。
  peg 穿过孔板 43 mm 的通孔下压，尖端被木板挡住——这使尖端恰好停在孔坐标系原点，
  即 dp -> 0。因此 genpih 的成功判据（dp < 3 mm 且 dq < 0.05 rad）**原样成立**；
  木板把"完全插入 = 尖端到达孔底"这个抽象换成了真实的物理止挡。
"""

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

# 成功判据阈值（genpih / RAAD 2025 论文 Eq. 7）
ALIGN_ROT_TOL = 0.05    # rad，姿态对准容差
INSERT_POS_TOL = 0.003  # m，插入位置容差
DH_POS_UNIT = 0.043     # m，孔径 = 2 * HOLE_R（见 genpih assets/make_meshes.py）


def peg_tip_pose_w(env):
    """peg 尖端在世界坐标系下的位姿（peg 是机器人 articulation 的一个刚体）。"""
    robot = env.scene["robot"]
    idx = robot.find_bodies("peg_link")[0][0]
    return robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]


def hole_pose_w(env):
    """孔坐标系在世界坐标系下的位姿。"""
    hole = env.scene["hole"]
    return hole.data.root_pos_w, hole.data.root_quat_w


def peg_pose_in_hole(env):
    """返回 (dp, dq)：peg 尖端位姿在孔坐标系下的表达。

    dp：peg 尖端到孔坐标系原点的笛卡尔距离。
    dq：peg 几何轴线与孔 +z 轴之间的夹角。

    关于方向约定的说明：peg 挂在 tool0 下，其连杆坐标系 +z 指向**下**（朝尖端），
    而孔坐标系 +z 指向**上**。因此我们量的是 peg 的"尖端->法兰"轴（连杆坐标系的 -z）
    与孔 +z 轴之间的**有向**夹角。正确的插入姿态下 dq -> 0；而 peg 朝上（那种能白拿
    对准奖励的退化姿态）会得到 dq -> pi，一分不得。**不要**用不带方向的 acos(|z·z|)：
    它会把倒过来的 peg 也算成 0。
    """
    p_peg, q_peg = peg_tip_pose_w(env)
    p_hole, q_hole = hole_pose_w(env)
    q_inv = quat_conjugate(q_hole)
    p_rel = quat_apply(q_inv, p_peg - p_hole)
    dp = torch.norm(p_rel, dim=-1)

    z_axis = torch.zeros_like(p_peg)
    z_axis[..., 2] = 1.0
    z_peg = -quat_apply(q_peg, z_axis)  # 尖端 -> 法兰
    z_hole = quat_apply(q_hole, z_axis)
    cos_a = (z_peg * z_hole).sum(dim=-1).clamp(min=-1.0, max=1.0)
    dq = torch.acos(cos_a)
    return dp, dq


# --------------------------------------------------------------------------
# 奖励
# --------------------------------------------------------------------------
#
# 原样沿用 genpih——那边的形式是对着实测数据选出来的。简版：论文 Eq. 5-6 的密集项
# 1 - tanh(sqrt(dq^2 + dp^2)) 把弧度加到了米上，且 u 超过约 3 之后就被压平到机器精度，
# 于是工作空间的大部分区域里奖励和它的梯度都消失了——在其下跑 3000 次迭代收敛到
# "对准了，但悬在孔上方 47 mm"，256 个环境 0 次插入。替代方案保留了原意（靠近目标位姿），
# 但让每个误差用它自己的任务单位来度量，并且用对准来把关位置得分：
#
#     r_dense = 1/(1 + dp/DH_POS_UNIT) * 1/(1 + dq/ALIGN_ROT_TOL)
#
# 用乘积而不是 L2 求和，因为歪掉的 peg 无论多用力都推进不了——这才是这个任务的真实结构。


def dense_reward(env):
    """r_dense = 1/(1 + dp/DH_POS_UNIT) * 1/(1 + dq/ALIGN_ROT_TOL)。"""
    dp, dq = peg_pose_in_hole(env)
    return 1.0 / (1.0 + dp / DH_POS_UNIT) / (1.0 + dq / ALIGN_ROT_TOL)


def parse_reward(env):
    """稀疏奖励：对准时 2.6，插入时 10（genpih 论文 Eq. 7）。"""
    dp, dq = peg_pose_in_hole(env)
    aligned = dq < ALIGN_ROT_TOL
    inserted = aligned & (dp < INSERT_POS_TOL)
    r = torch.zeros_like(dp)
    r = torch.where(aligned, 2.6, r)
    r = torch.where(inserted, 10.0, r)
    return r


def success_term(env):
    """成功插入：dq < 0.05 rad 且 dp < 0.003 m。"""
    dp, dq = peg_pose_in_hole(env)
    return (dq < ALIGN_ROT_TOL) & (dp < INSERT_POS_TOL)


# --------------------------------------------------------------------------
# 观测
# --------------------------------------------------------------------------


def hole_pose_in_peg(env):
    """孔坐标系在 peg 尖端坐标系下的表达：(位置, 四元数)。"""
    p_peg, q_peg = peg_tip_pose_w(env)
    p_hole, q_hole = hole_pose_w(env)
    q_inv = quat_conjugate(q_peg)
    return quat_apply(q_inv, p_hole - p_peg), quat_mul(q_inv, q_hole)


def hole_pos_in_peg(env):
    """孔原点在 peg 尖端坐标系下的位置——策略要伺服消除的位置误差。"""
    return hole_pose_in_peg(env)[0]


def hole_quat_in_peg(env):
    """孔姿态在 peg 尖端坐标系下的表达（wxyz）——姿态对准误差。"""
    return hole_pose_in_peg(env)[1]


def peg_tip_height(env):
    """peg 尖端高出木板顶面的高度，单位米，形状 (N, 1)。

    给 critic 的一个廉价特权信号：与 dp 不同，它不会把横向误差和垂直误差压缩成
    一个数，所以价值函数能区分"悬在高处且居中"和"已经坐在木板上"。
    """
    p_peg, _ = peg_tip_pose_w(env)
    board_top = torch.tensor(env.cfg.board_top_z, device=env.device)
    return (p_peg[:, 2] - board_top).unsqueeze(-1)


def camera_image(env, sensor_cfg=None, data_type: str = "rgb"):
    """眼在手上相机图像，缩放到 [0, 1] 的浮点数，形状 (N, H, W, C)。

    **刻意不用** `isaaclab.envs.mdp.observations.image` 的默认 `normalize=True`：
    对 rgb 而言它会先除以 255，然后**减去当前 batch 的均值**。这会让某个环境的观测
    取决于恰好与它同批的其它环境，于是同一个场景在 batch=1024 和 batch=1 下产生不同的
    观测——这正是"训练"与"单环境确定性评估"之间的差异来源。固定的逐像素缩放让观测
    只依赖于智能体自己的视角。
    """
    from isaaclab.envs.mdp import observations as obs_mdp
    from isaaclab.managers import SceneEntityCfg

    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("tiled_camera")
    img = obs_mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
    img = img.float() / 255.0

    # 显式展平成 (N, H*W*C)。Isaac Lab 的 ObservationManager **不会**对只含单个项的组
    # 做拼接，所以图像本来会以 (N, H, W, C) 的形状到达 wrapper——而 RslRlVecEnvWrapper
    # 用 `group_obs_dim["policy"][0]` 给策略定维度，那会把它读成 H（=100）而不是 H*W*C。
    # 按行主序展平，网络那边用 view(N, H, W, C).permute(0, 3, 1, 2) 还原。
    return img.reshape(img.shape[0], -1)


# --------------------------------------------------------------------------
# 事件
# --------------------------------------------------------------------------


def reset_robot_to_home(env, env_ids: torch.Tensor):
    """恢复固定的 home 关节配置（不加抖动：本任务没有域随机化）。

    按**名字**查关节而不是按列索引：USD 导入器吐出的顺序和这些常量的书写顺序不一样，
    而写错关节是一个悄无声息、且看起来很合理的 bug（机械臂只是从别的地方起步，
    所有可达性计算就全错了）。
    """
    robot = env.scene["robot"]
    names = list(env.cfg.home_joint_names)
    values = torch.tensor(env.cfg.home_joint_values, device=env.device)

    joint_ids, found = robot.find_joints(names, preserve_order=True)
    # 报错信息保留英文：Isaac Lab 自己抛的错全是英文，混着中文更难查，而且英文报错可以搜
    if list(found) != names:
        raise RuntimeError(f"home joints not found on the robot: wanted {names}, found {list(found)}")
    if robot.num_joints != len(names):
        raise RuntimeError(
            f"robot has {robot.num_joints} joints ({robot.joint_names}) but home defines {len(names)}; "
            "the reset would leave the others at whatever they were"
        )

    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_pos[:, joint_ids] = values.unsqueeze(0)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def reset_hole_fixed(env, env_ids: torch.Tensor):
    """把孔放到夹具位姿。每次 reset 都写同一个值——没有域随机化。

    用世界坐标写（环境原点 + 夹具偏移），与 genpih 的做法一致；无论 cloner 如何把
    作者位姿与每环境的偏移组合起来，这个写法都是对的。
    """
    hole = env.scene["hole"]
    state = hole.data.root_state_w[env_ids].clone()
    fixture = torch.tensor(env.cfg.fixture_pos, device=env.device)
    state[:, 0:3] = env.scene.env_origins[env_ids] + fixture
    state[:, 3] = 1.0
    state[:, 4:7] = 0.0
    state[:, 7:13] = 0.0
    hole.write_root_state_to_sim(state, env_ids=env_ids)

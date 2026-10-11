"""UR10e 视觉插销任务的环境配置。

场景（全部固定——本任务**没有**域随机化）：
  - UR10e + Cranfield 轴销（genpih 的资产，原样复用）
  - 孔板（genpih 的资产）放在一块实心木板上
  - 木板顶面位于 BOARD_TOP_Z = 0.1807 m = UR10e shoulder_pan_joint 的高度，
    也就是基座铸件的顶面
  - 木板是**实心**的：peg 穿过孔板 43 mm 的通孔下压，尖端被木板挡住。这使尖端恰好
    落在孔坐标系原点（dp -> 0），因此 genpih 的成功判据原样成立。
  - 眼在手上相机挂在 wrist_3_link 上（视觉观测）

动作/奖励/终止都来自 genpih，未改动：
  - 动作：6 个绝对关节位置目标
  - 奖励：密集 1/(1+dp/0.043) * 1/(1+dq/0.05) [权重 10] + 稀疏 2.6 / 10
  - 回合仅在超时时结束（900 步 @ 60 Hz），不在成功时结束
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import observations as obs_mdp
from isaaclab.envs.mdp import terminations as term_mdp
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

import pegih_tasks.tasks.peg_hole.mdp as mdp

# 项目根目录（存放本包的目录）；路径里带空格，所以这里任何东西都不能不加引号就交给 shell
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROBOT_USD = os.path.join(PROJECT_ROOT, "assets", "usd", "robot", "ur10e_peg.usd")
HOLE_USD = os.path.join(PROJECT_ROOT, "assets", "usd", "hole", "hole_plate.usd")

# ---------------------------------------------------------------------------
# 几何。下面每个数字都是用 genpih 的 URDF 做正运动学 + 阻尼最小二乘 IK 算出来并核对过的，
# 不是估算：
#
#   HOME_JOINTS -> peg 尖端在 (0.1037, 0.7054, 0.5568)，peg 轴线恰好是
#                  (0, 0, -1)，即垂直度误差 0.0000 度
#   木板顶面     -> z = 0.1807（shoulder_pan_joint 高度，即"基座顶面"）
#   下压行程     -> 0.3761 m
#
# 对 0.04 到 0.15 m 的每一个候选孔位偏移，IK 都能以 0.000 mm 的位置误差和 0.0000 度的
# 倾斜到达木板，关节偏移仅 0.60-0.71 rad——远在 ±1.5 rad 的动作裁剪之内——并且最低的
# 连杆（wrist_3，z = 0.301）始终比孔板顶面（0.2107）高 0.09 m，所以下压途中没有任何
# 东西会撞到孔板。
# ---------------------------------------------------------------------------

BOARD_TOP_Z = 0.1807
PLATE_THICKNESS = 0.03               # genpih 的 hole_plate.obj，PLATE = (0.20, 0.20, 0.03)
PLATE_TOP_Z = BOARD_TOP_Z + PLATE_THICKNESS
BOARD_SIZE = (0.60, 0.60, 0.04)      # x, y, z
BOARD_THICKNESS = BOARD_SIZE[2]
BOARD_CENTER_Z = BOARD_TOP_Z - BOARD_THICKNESS / 2
PEDESTAL_SIZE = (0.40, 0.40, BOARD_TOP_Z - BOARD_THICKNESS)
PEDESTAL_CENTER_Z = PEDESTAL_SIZE[2] / 2

# HOME 时的 peg 尖端 xy，实测值（见上）。孔相对于它做横向偏移，好让 peg 起始时**不**与孔
# 对齐——这是用户的要求。偏移量的下界是"必须真的需要修正"，上界是眼在手上相机的视场
# （起始时孔必须留在画面内）；0.06 m 舒适地落在这个区间里，并且只在一个地方可调。
HOME_TIP_XY = (0.1037, 0.7054)
START_OFFSET_XY = (0.06, 0.0)
FIXTURE_POS = (HOME_TIP_XY[0] + START_OFFSET_XY[0], HOME_TIP_XY[1] + START_OFFSET_XY[1], BOARD_TOP_Z)

# genpih 的初始关节配置。在这个位姿下 peg 恰好垂直向下悬挂（UR 惯例：
# shoulder_lift=-90、elbow=90、wrist_1=-90、wrist_2=-90、wrist_3=0）。
HOME_JOINTS = {
    "shoulder_pan_joint": 1.178097,    # 67.5 度
    "shoulder_lift_joint": -1.570796,  # -90 度
    "elbow_joint": 1.570796,           # 90 度
    "wrist_1_joint": -1.570796,        # -90 度
    "wrist_2_joint": -1.570796,        # -90 度
    "wrist_3_joint": 0.0,              # 0 度
}
# 这个顺序是为了能作为张量交给 write_joint_state_to_sim
HOME_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# 导出成 torch 友好的列表；env cfg 的字段在 __post_init__ 里填
# （关节顺序由 URDF 决定，并由 scripts/smoke_test.py 断言）

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

# 眼在手上相机的 offset，在 wrist_3_link 坐标系下，convention="ros"。
#
# **不是**从 ur10_peg_hole 抄来的：那个项目的四元数是个魔数，文档只写了
# "computed to look at the peg tip"，而且它针对的是另一个 peg（r=0.018 x 0.08 的圆柱，
# 挂在腕下 0.20 m，而不是这个 0.12 m 的 Cranfield 轴销）。几何不同，取景就不可能一样。
# 这里改用 scripts/cam_offset.py 反解：给定世界机位和世界看向点，解出对应的旋转，
# 再做 opengl -> ros 的转换。
#
# 选定的机位：相机沿 wrist 的 -x 轴方向 0.09 m（在 HOME 时这个方向是水平的，且背向孔），
# 与腕同高，瞄向 peg 尖端与孔的中点。用 24 cm 焦距 / 20.955 cm 光圈的默认值实测
# （HFOV 47.2 度）：peg 尖端落在离轴 12.6 度、距离 0.150 m 处，孔落在离轴 8.2 度、
# 距离 0.518 m 处，所以每个回合开始时两者都稳稳处在 23.6 度的半视场内。
#
# 为什么不放更远：实测渲染显示 0.12 m 和 0.20 m 会把 peg 顶出画面上方（它是离相机
# 最近的物体），而孔几乎没有改善——到 0.20 m 时 peg 基本看不见了。
#
# 一个已知限制，而且它是几何造成的、不是取景问题：peg 直径 42.4 mm、孔直径 43 mm，
# 所以尖端离孔板 30 mm 以内时，**任何**相机位置都拍不到完整的孔——把两者分开的视差
# 随尖端到孔板的距离缩小而消失。仍然可见的是孔从 peg 侧面露出来的月牙形，它的宽度
# 直接编码横向误差，所以信息没丢，只是换了形式。（参考项目的装置有同样的性质，但
# 宽容得多：它的 peg 是 36 mm 对 48 mm 的孔。）
CAMERA_OFFSET_POS = (-0.09, 0.0, 0.0)
CAMERA_OFFSET_ROT = (0.72358, 0.15792, 0.14327, 0.65648)


def make_camera_cfg() -> TiledCameraCfg:
    """Tiled 眼在手上相机。用 Tiled（而不是每环境一个 Camera）是因为一次渲染出几千张
    100x100 的图像，比分成几千次单独渲染便宜得多。"""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/Camera",
        # 每个**策略**步出一张图（控制频率 60 Hz）；保持默认的 0 会让它在每个物理步都
        # 重渲染，即白白多花一倍
        update_period=1.0 / 60.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS,
            rot=CAMERA_OFFSET_ROT,
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            # 近裁剪面 0.05 m：HOME 时 peg 尖端在相机前方约 0.13 m，所以默认的 0.01 m
            # 没必要，而更紧的近裁剪面能改善深度精度
            clipping_range=(0.05, 20.0),
        ),
    )


##
# 资产
##

UR10E_PEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        # 那些不用力/力矩传感器的探针脚本要读 peg 的接触，而且后续版本可能想要基于接触的
        # 终止条件；打开它的开销测不出来
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=HOME_JOINTS),
    actuators={
        # 从 genpih 沿用；那边的参数就是对着这条机械臂量出来的：PD 伺服跟踪一个运动目标
        # 会滞后 e = d*v/k，所以刚度既决定跟踪速度也决定下垂量。在本场景、
        # 保持零动作在 home 时实测：
        #     k=800   d=40   -> 下垂 106.5 mm，振铃 0.48 mm   (zeta ~ 0.21)
        #     k=50000 d=1500 -> 下垂   1.77 mm，振铃 0.00 mm   (zeta ~ 1.0)
        # 在 50000/1500 下，下垂量低于本任务自己的 3 mm 插入容差，且机械臂在
        # dt = 1/120 s 下是临界阻尼的。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=50000.0,
            damping=1500.0,
            effort_limit_sim=87.0,
        ),
    },
)

# kinematic（这个标志被 genpih 的 convert_assets.py 烘进了 USD）并由 reset 事件按世界
# 坐标放置——见 reset_hole_fixed。
HOLE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HOLE_USD,
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=FIXTURE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
)

# 木板。kinematic，所以它永远不会掉落或被推动——这是用户"不可穿透、不移动"的要求——
# 但它仍然是一个可碰撞的刚体，而这正是重点：它就是那个在孔底挡住 peg 的东西。
BOARD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Board",
    init_state=AssetBaseCfg.InitialStateCfg(pos=(FIXTURE_POS[0], FIXTURE_POS[1], BOARD_CENTER_Z)),
    spawn=sim_utils.CuboidCfg(
        size=BOARD_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.52, 0.33, 0.16), roughness=0.85, metallic=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.0
        ),
    ),
)

# 支撑座，免得木板悬在 180 mm 高的空中。纯视觉和结构用途；机器人是靠 fix_base 焊死在
# 世界上的，不是靠它。
PEDESTAL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Pedestal",
    init_state=AssetBaseCfg.InitialStateCfg(pos=(FIXTURE_POS[0], FIXTURE_POS[1], PEDESTAL_CENTER_Z)),
    spawn=sim_utils.CuboidCfg(
        size=PEDESTAL_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.30, 0.30, 0.32), roughness=0.9, metallic=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.0
        ),
    ),
)


@configclass
class PegHoleSceneCfg(InteractiveSceneCfg):
    """场景：UR10e + 轴销、木板上的孔板、地面、灯光、peg 接触传感器。"""

    robot: ArticulationCfg = UR10E_PEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    hole: RigidObjectCfg = HOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Hole")
    board: AssetBaseCfg = BOARD_CFG
    pedestal: AssetBaseCfg = PEDESTAL_CFG

    # 只针对 peg 的接触传感器，scripts/probe_collision.py 用它回答那个决定碰撞设计的问题：
    # PhysX 会不会在 kinematic 的凹形孔板网格和 peg 的凸包之间生成接触？
    # 注意 filter 每个环境必须恰好命名**一个** prim——一个会展开成孔板子碰撞体的正则
    # 会把张量 API 搞坏。
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/peg_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Hole", "{ENV_REGEX_NS}/Board"],
        track_air_time=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=256,
        update_period=0.0,
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # 眼在手上相机。调 PegHoleEnvCfg.enable_vision(False) 可以去掉这个传感器
    # （AppLauncher 不带 enable_cameras 跑时需要，无相机的冒烟测试也需要）；
    # InteractiveScene 会跳过值为 None 的条目。
    tiled_camera: TiledCameraCfg | None = make_camera_cfg()


##
# MDP 设置
##

ACTION_SCALE = 1.5

# JointPositionAction 只在设置了 cfg.clip 时才裁剪，而且它裁剪的是**处理之后**的目标
# （raw*scale + default），不是原始动作。genpih 当初没带这个 clip，实测
# |关节目标 - home| 最大到 21.58 rad，而 scale 暗示的只有 1.5——每个关节都被钉在限位上。
# 这些边界正是 scale 所声明的范围（home ± 1.5 rad），由 HOME_JOINTS 推导而来，
# 所以不可能和偏移量失步。
ACT_CLIP = {name: (pos - ACTION_SCALE, pos + ACTION_SCALE) for name, pos in HOME_JOINTS.items()}


@configclass
class ActionsCfg:
    """动作：6 个绝对关节位置目标。"""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=ACTION_SCALE,       # 动作在 [-1, 1] -> 相对默认位姿 ±1.5 rad
        use_default_offset=True,  # 动作为 0 == 默认（home）关节位置
        clip=ACT_CLIP,            # 强制兑现 scale 所声明的 [-1, 1]
    )


@configclass
class ObservationsCfg:
    """观测组。

    policy：部署时策略看到的东西。开启视觉时它**只有**相机图像（用户的"纯视觉"要求）；
            不开启时它是 13 维状态向量，这样不必付渲染的代价就能冒烟测试
            几何/物理/奖励。
    critic：给非对称 actor-critic 的特权状态。actor 永远看不到它，它也不参与部署；
            它只是让价值函数更准。设 privileged_critic=False 就得到完全对称的、
            端到端只有视觉的网络。
    """

    @configclass
    class StatePolicyCfg(ObsGroup):
        hole_pos = ObsTerm(func=mdp.hole_pos_in_peg)
        hole_quat = ObsTerm(func=mdp.hole_quat_in_peg)
        last_action = ObsTerm(func=obs_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImagePolicyCfg(ObsGroup):
        """部署时策略看到的东西：相机图像，别的什么都没有。"""

        image = ObsTerm(func=mdp.camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        hole_pos = ObsTerm(func=mdp.hole_pos_in_peg)
        hole_quat = ObsTerm(func=mdp.hole_quat_in_peg)
        tip_height = ObsTerm(func=mdp.peg_tip_height)
        last_action = ObsTerm(func=obs_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: StatePolicyCfg = StatePolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """genpih 的奖励项，未改动（推导见 mdp.dense_reward）。"""

    dense = RewTerm(func=mdp.dense_reward, weight=10.0)
    parse = RewTerm(func=mdp.parse_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """回合仅在超时时结束——**刻意不**在插入时结束。

    稀疏奖励是**每步**支付的（对准 2.6，插入 10）。若在插入时终止，激励顺序就被颠倒了：
    全程悬停对准 900 步反而胜过早于约第 897 步插入。genpih 在另一套奖励下实测到这个
    失败并记录了下来；这里的推理没有变。
    """

    time_out = TermTerm(func=term_mdp.time_out)


@configclass
class EventCfg:
    """Reset 事件。没有任何随机化：两者每次都写入同一个值。"""

    # 孔是一个 kinematic 的 RigidObject，它的默认状态是从 spawn 出来的 prim 上捕获的。
    # genpih 每次 reset 都显式地按**世界**坐标写它（env_origins + fixture），这个写法
    # 无论 cloner 如何组合作者位姿都成立；我们沿用这个做法。
    reset_hole_fixed = EventTerm(func=mdp.reset_hole_fixed, mode="reset")
    reset_robot_joints = EventTerm(func=mdp.reset_robot_to_home, mode="reset")


@configclass
class PegHoleEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e 视觉插销环境。"""

    # 任务专用：由 mdp 函数通过 env.cfg 读取
    fixture_pos: tuple = FIXTURE_POS
    board_top_z: float = BOARD_TOP_Z
    home_joint_names: tuple = tuple(HOME_JOINT_ORDER)
    home_joint_values: tuple = tuple(HOME_JOINTS[n] for n in HOME_JOINT_ORDER)

    # 512，而不是 genpih 的 4096/8192：配上 tiled 相机和 CNN 之后，这是这张卡真正装得下的
    # 数量。在一次实跑中于 RTX 4090（共 24.5 GB）上实测：512 环境 -> 训练进程常驻
    # 21.7 GB，只剩约 1.3 GB 空闲。1024 环境会 OOM。状态观测的训练便宜得多，可以上到
    # 高得多（1024 环境约 1.0e5 steps/s）。
    scene: PegHoleSceneCfg = PegHoleSceneCfg(num_envs=512, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1 / 120
        self.decimation = 2                    # 策略控制频率 60 Hz
        self.episode_length_s = 15.0           # 900 步

        # 1024 个环境 x 100x100x3 的 tiled 渲染再加一次 CNN 反向传播，内存画像和 genpih
        # 那 8192 个"盲"环境完全不同；这里是个起点，要对着实测显存再调。
        self.sim.physx.gpu_collision_stack_size = 2**28

    def enable_vision(self, enabled: bool = True, privileged_critic: bool = True):
        """在相机图像和状态之间切换 policy 观测。

        刻意做成可重复调用：@configclass 在构造时就跑 __post_init__，早于任何调用方设置
        字段，所以脚本必须能在改动 cfg 之后再调一次（genpih 重跑 _apply_experiment
        也是同样的原因）。

        参数：
            enabled: True -> policy 组只有相机图像。
            privileged_critic: 仅在 enabled 时有意义。True 会为 critic 保留一个独立的
                14 维状态组（非对称 actor-critic：actor 仍然只看像素，critic 拿到从不
                部署的特权状态）。False 会去掉该组，于是 critic 也吃图像，网络变成对称的
                ——端到端完全"纯视觉"。
        """
        if enabled:
            self.scene.tiled_camera = make_camera_cfg()
            self.observations.policy = ObservationsCfg.ImagePolicyCfg()
            self.observations.critic = ObservationsCfg.CriticCfg() if privileged_critic else None
        else:
            self.scene.tiled_camera = None
            self.observations.policy = ObservationsCfg.StatePolicyCfg()
            self.observations.critic = ObservationsCfg.CriticCfg()

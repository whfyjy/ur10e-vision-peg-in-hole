"""Environment configuration for the UR10e vision peg-in-hole task.

Scene (all fixed -- this task has NO domain randomization):
  - UR10e + Cranfield peg (genpih's asset, reused verbatim)
  - hole plate (genpih's asset) sitting ON a solid wooden board
  - board top face at BOARD_TOP_Z = 0.1807 m = the height of the UR10e
    shoulder_pan_joint, i.e. the top of the base casting
  - the board is SOLID: the peg descends through the plate's 43 mm bore and its
    tip is stopped by the board. That places the tip exactly at the hole frame
    origin (dp -> 0), so genpih's success criterion carries over unchanged.
  - eye-in-hand camera on wrist_3_link (vision observation)

Action/reward/termination are genpih's, unchanged:
  - action: 6 absolute joint position targets
  - reward: dense 1/(1+dp/0.043) * 1/(1+dq/0.05) [w=10] + sparse 2.6 / 10
  - episode ends on timeout only (900 steps @ 60 Hz), not on success
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

# project root (the directory holding this package); the path contains a space,
# so nothing here may be shell-interpolated unquoted
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROBOT_USD = os.path.join(PROJECT_ROOT, "assets", "usd", "robot", "ur10e_peg.usd")
HOLE_USD = os.path.join(PROJECT_ROOT, "assets", "usd", "hole", "hole_plate.usd")

# ---------------------------------------------------------------------------
# Geometry. Every number below was derived and checked with forward kinematics +
# damped-least-squares IK against genpih's URDF, not estimated:
#
#   HOME_JOINTS -> peg tip at (0.1037, 0.7054, 0.5568), peg axis exactly
#                  (0, 0, -1), i.e. vertical to 0.0000 deg
#   board top   -> z = 0.1807 (shoulder_pan_joint height, "top of the base")
#   descent     -> 0.3761 m
#
# For every candidate hole offset from 0.04 to 0.15 m the IK reaches the board
# with 0.000 mm position error and 0.0000 deg tilt using joint offsets of only
# 0.60-0.71 rad -- well inside the +-1.5 rad action clip -- and the lowest link
# (wrist_3 at z = 0.301) stays 0.09 m above the plate top (0.2107), so nothing
# clips the plate on the way down.
# ---------------------------------------------------------------------------

BOARD_TOP_Z = 0.1807
PLATE_THICKNESS = 0.03               # genpih's hole_plate.obj, PLATE = (0.20, 0.20, 0.03)
PLATE_TOP_Z = BOARD_TOP_Z + PLATE_THICKNESS
BOARD_SIZE = (0.60, 0.60, 0.04)      # x, y, z
BOARD_THICKNESS = BOARD_SIZE[2]
BOARD_CENTER_Z = BOARD_TOP_Z - BOARD_THICKNESS / 2
PEDESTAL_SIZE = (0.40, 0.40, BOARD_TOP_Z - BOARD_THICKNESS)
PEDESTAL_CENTER_Z = PEDESTAL_SIZE[2] / 2

# Peg tip xy at HOME, measured (see above). The hole is offset laterally from it
# so the peg starts NOT aligned with the hole -- the user's requirement. The
# magnitude is bounded below by "must actually need a correction" and above by
# the eye-in-hand camera's field of view (the hole has to stay in frame at the
# start); 0.06 m sits comfortably in that window and is tunable in one place.
HOME_TIP_XY = (0.1037, 0.7054)
START_OFFSET_XY = (0.06, 0.0)
FIXTURE_POS = (HOME_TIP_XY[0] + START_OFFSET_XY[0], HOME_TIP_XY[1] + START_OFFSET_XY[1], BOARD_TOP_Z)

# genpih's initial joint configuration. At this pose the peg hangs exactly
# vertically (UR convention: shoulder_lift=-90, elbow=90, wrist_1=-90,
# wrist_2=-90, wrist_3=0).
HOME_JOINTS = {
    "shoulder_pan_joint": 1.178097,    # 67.5 deg
    "shoulder_lift_joint": -1.570796,  # -90 deg
    "elbow_joint": 1.570796,           # 90 deg
    "wrist_1_joint": -1.570796,        # -90 deg
    "wrist_2_joint": -1.570796,        # -90 deg
    "wrist_3_joint": 0.0,              # 0 deg
}
# ordered so it can be handed to write_joint_state_to_sim as a tensor
HOME_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# exported as a torch-friendly list; the env cfg field is filled in __post_init__
# (the joint order is fixed by the URDF and asserted by scripts/smoke_test.py)

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

# Eye-in-hand camera offset in the wrist_3_link frame, convention="ros".
#
# NOT copied from ur10_peg_hole: that project's quaternion is a magic constant
# documented only as "computed to look at the peg tip", and it targets a
# different peg (a r=0.018 x 0.08 cylinder hanging 0.20 m below the wrist, not
# this 0.12 m Cranfield peg). Given a different geometry it would not frame the
# same way. Derived instead by scripts/cam_offset.py, which solves for the
# rotation that looks from a chosen world point at a chosen world target, then
# converts opengl -> ros.
#
# Chosen view: camera 0.09 m along the wrist's -x axis (which, at HOME, is
# horizontal and points away from the hole), level with the wrist, aimed at the
# midpoint between the peg tip and the hole. Measured with the 24 cm focal
# length / 20.955 cm aperture default (HFOV 47.2 deg): the peg tip lands 12.6 deg
# off-axis at 0.150 m and the hole 8.2 deg off-axis at 0.518 m, so both sit well
# inside the 23.6 deg half-FOV at the start of every episode.
#
# Why not further out: measured renders at 0.12 m and 0.20 m push the peg out of
# the top of the frame (it is the closest thing to the camera) while the hole
# barely improves -- at 0.20 m the peg is nearly gone.
#
# Known limitation, and it is geometry rather than framing: the peg is 42.4 mm
# and the bore is 43 mm, so once the tip is within ~30 mm of the plate the peg
# occludes the hole from ANY camera position -- the parallax that separates them
# shrinks with the tip-to-plate distance. What stays visible is the crescent of
# bore showing past the peg's flank, whose width directly encodes the lateral
# misalignment, so the signal is not lost, only changed in form. (The reference
# project's rig has the same property and is far more forgiving: its peg is
# 36 mm against a 48 mm hole.)
CAMERA_OFFSET_POS = (-0.09, 0.0, 0.0)
CAMERA_OFFSET_ROT = (0.72358, 0.15792, 0.14327, 0.65648)


def make_camera_cfg() -> TiledCameraCfg:
    """Tiled eye-in-hand camera. Tiled (not per-env Camera) because a few
    thousand 100x100 render products in one pass is far cheaper than a few
    thousand separate ones."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/Camera",
        # one image per POLICY step (control runs at 60 Hz); leaving this at 0
        # would re-render on every physics step, i.e. twice as often for nothing
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
            # near plane 0.05 m: the peg tip sits ~0.13 m in front of the
            # camera at HOME, so the 0.01 m default is not needed and a tighter
            # near plane improves depth precision
            clipping_range=(0.05, 20.0),
        ),
    )


##
# Assets
##

UR10E_PEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        # the F/T-free probe scripts read peg contacts, and a later revision may
        # want contact-based termination; enabling it costs nothing measurable
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=HOME_JOINTS),
    actuators={
        # Carried over from genpih, where these were measured against this exact
        # arm: a PD servo lags a moving target by e = d*v/k, so stiffness sets
        # tracking speed as well as droop. Measured on this scene with zero
        # action held at home:
        #     k=800   d=40   -> droop 106.5 mm, ringing 0.48 mm   (zeta ~ 0.21)
        #     k=50000 d=1500 -> droop   1.77 mm, ringing 0.00 mm   (zeta ~ 1.0)
        # At 50000/1500 the droop is below the task's own 3 mm insertion
        # tolerance and the arm is critically damped at dt = 1/120 s.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=50000.0,
            damping=1500.0,
            effort_limit_sim=87.0,
        ),
    },
)

# Kinematic (the flag is baked into the USD by genpih's convert_assets.py) and
# placed by a reset event in world coordinates -- see reset_hole_fixed.
HOLE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HOLE_USD,
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=FIXTURE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
)

# Wooden board. Kinematic so it can never fall or be pushed -- the user's
# "impenetrable, does not move" requirement -- but still a collidable body, which
# is the whole point: it is what stops the peg at the bore bottom.
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

# Support block so the board is not floating 180 mm in the air. Purely visual
# and structural; the robot is welded to the world by fix_base, not by this.
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
    """Scene: UR10e + peg, hole plate on a wooden board, ground, light, peg contact sensor."""

    robot: ArticulationCfg = UR10E_PEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    hole: RigidObjectCfg = HOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Hole")
    board: AssetBaseCfg = BOARD_CFG
    pedestal: AssetBaseCfg = PEDESTAL_CFG

    # peg-only contact sensor, used by scripts/probe_collision.py to answer the
    # one question that decides the collision design: does PhysX generate
    # contacts between the kinematic concave plate mesh and the peg's convex
    # hull? NOTE the filter must name exactly ONE prim per env -- a regex that
    # expands to the plate's child colliders breaks the tensor API.
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

    # Eye-in-hand camera. Call PegHoleEnvCfg.enable_vision(False) to drop the
    # sensor (needed when AppLauncher runs without enable_cameras, and for the
    # camera-free smoke tests); InteractiveScene skips None entries.
    tiled_camera: TiledCameraCfg | None = make_camera_cfg()


##
# MDP settings
##

ACTION_SCALE = 1.5

# JointPositionAction only clamps when cfg.clip is set, and it clamps the
# PROCESSED target (raw*scale + default), not the raw action. genpih shipped
# without the clip and measured max |joint_target - home| reaching 21.58 rad
# against the 1.5 the scale implies -- every joint pinned at its limit. These
# bounds are exactly what the scale documents (home +- 1.5 rad), derived from
# HOME_JOINTS so they cannot drift out of sync with the offset.
ACT_CLIP = {name: (pos - ACTION_SCALE, pos + ACTION_SCALE) for name, pos in HOME_JOINTS.items()}


@configclass
class ActionsCfg:
    """Action: six absolute joint position targets."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=ACTION_SCALE,       # action in [-1, 1] -> +-1.5 rad around default pose
        use_default_offset=True,  # action 0 == default (home) joint positions
        clip=ACT_CLIP,            # enforce the [-1, 1] the scale documents
    )


@configclass
class ObservationsCfg:
    """Observation groups.

    policy: what the deployed policy sees. With vision enabled this is the
            camera image ONLY (the user's "pure vision" requirement); without it
            it is the 13-dim state vector, so geometry/physics/reward can be
            smoke-tested without paying for rendering.
    critic: privileged state for an asymmetric actor-critic. The actor never
            sees this and it is not deployed; it only makes the value function
            accurate. Set privileged_critic=False for a fully symmetric
            vision-only network.
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
        """What the deployed policy sees: the camera image, nothing else."""

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
    """genpih's reward terms, unchanged (see mdp.dense_reward for the derivation)."""

    dense = RewTerm(func=mdp.dense_reward, weight=10.0)
    parse = RewTerm(func=mdp.parse_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Episode ends on timeout only -- deliberately NOT on insertion.

    The sparse reward is paid PER STEP (2.6 aligned, 10 inserted). Terminating
    on insertion inverts the incentive order: hovering aligned for the full
    900 steps then beats inserting at any step before ~897. genpih measured
    exactly that failure under a different reward and documented it; the
    reasoning is unchanged here.
    """

    time_out = TermTerm(func=term_mdp.time_out)


@configclass
class EventCfg:
    """Reset events. There is no randomization: both write the same value every time."""

    # The hole is a kinematic RigidObject whose default state is captured from
    # the spawned prim. genpih writes it explicitly in WORLD coordinates on
    # every reset (env_origins + fixture), which is provably correct regardless
    # of how the cloner composes the authored pose; we keep that approach.
    reset_hole_fixed = EventTerm(func=mdp.reset_hole_fixed, mode="reset")
    reset_robot_joints = EventTerm(func=mdp.reset_robot_to_home, mode="reset")


@configclass
class PegHoleEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e vision peg-in-hole environment."""

    # task-specific: consumed by mdp functions through env.cfg
    fixture_pos: tuple = FIXTURE_POS
    board_top_z: float = BOARD_TOP_Z
    home_joint_names: tuple = tuple(HOME_JOINT_ORDER)
    home_joint_values: tuple = tuple(HOME_JOINTS[n] for n in HOME_JOINT_ORDER)

    # 512, not genpih's 4096/8192: with the tiled camera and the CNN this is what
    # the card actually holds. Measured on the RTX 4090 (24.5 GB total) during a
    # live run: 512 envs -> 21.7 GB resident for the training process, leaving
    # ~1.3 GB free. 1024 envs would OOM. State-observation runs are far cheaper
    # and can go much higher (1024 envs trains at ~1.0e5 steps/s).
    scene: PegHoleSceneCfg = PegHoleSceneCfg(num_envs=512, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1 / 120
        self.decimation = 2                    # policy control at 60 Hz
        self.episode_length_s = 15.0           # 900 steps

        # 1024 environments x 100x100x3 tiled rendering plus a CNN backward pass
        # is a very different memory profile from genpih's 8192 blind envs; this
        # is a starting point to be re-tuned against measured VRAM.
        self.sim.physx.gpu_collision_stack_size = 2**28

    def enable_vision(self, enabled: bool = True, privileged_critic: bool = True):
        """Switch the policy observation between the camera image and state.

        Re-runnable on purpose: @configclass runs __post_init__ at construction
        time, before a caller can set any field, so scripts must be able to call
        this again after mutating the cfg (the same reason genpih re-runs
        _apply_experiment).

        Args:
            enabled: True -> the policy group is the camera image only.
            privileged_critic: only meaningful when enabled. True keeps a
                separate 14-dim state group for the critic (asymmetric
                actor-critic: the actor still sees only pixels, the critic gets
                privileged state that is never deployed). False drops the group,
                so the critic also consumes the image and the network is
                symmetric -- fully "pure vision" end to end.
        """
        if enabled:
            self.scene.tiled_camera = make_camera_cfg()
            self.observations.policy = ObservationsCfg.ImagePolicyCfg()
            self.observations.critic = ObservationsCfg.CriticCfg() if privileged_critic else None
        else:
            self.scene.tiled_camera = None
            self.observations.policy = ObservationsCfg.StatePolicyCfg()
            self.observations.critic = ObservationsCfg.CriticCfg()

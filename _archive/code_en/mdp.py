"""MDP functions for the UR10e vision peg-in-hole task.

Frames (identical to genpih's, whose meshes we reuse verbatim):
  - peg link frame == peg TIP frame (fixed joint from tool0, +0.120 m along tool0 z)
  - hole frame == bore center on the plate BOTTOM plane, bore axis along +z

Scene difference from genpih (this is the whole point of this project):
  the plate sits ON a solid wooden board whose top face is at BOARD_TOP_Z, so the
  hole frame origin lies exactly on the board surface. The peg descends through
  the plate's 43 mm bore and its tip is stopped by the board, which puts the tip
  exactly at the hole frame origin -> dp -> 0. genpih's success criterion
  (dp < 3 mm and dq < 0.05 rad) therefore carries over UNCHANGED; the board
  replaces the "fully inserted = tip at the bore bottom" abstraction with an
  actual physical stop.
"""

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

# success thresholds (genpih / RAAD 2025 paper Eq. 7)
ALIGN_ROT_TOL = 0.05    # rad, orientation alignment
INSERT_POS_TOL = 0.003  # m, insertion
DH_POS_UNIT = 0.043     # m, bore diameter = 2 * HOLE_R (see genpih assets/make_meshes.py)


def peg_tip_pose_w(env):
    """Peg tip pose in world frame (the peg is a body of the robot articulation)."""
    robot = env.scene["robot"]
    idx = robot.find_bodies("peg_link")[0][0]
    return robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]


def hole_pose_w(env):
    """Hole frame pose in world frame."""
    hole = env.scene["hole"]
    return hole.data.root_pos_w, hole.data.root_quat_w


def peg_pose_in_hole(env):
    """Return (dp, dq): peg tip pose expressed in the hole frame.

    dp: Cartesian distance between the peg tip and the hole frame origin.
    dq: angle between the peg's geometric axis and the hole's +z axis.

    NOTE on the direction convention: the peg hangs from tool0 with its link
    frame +z pointing DOWN (toward the tip) while the hole frame +z points UP.
    We therefore measure the DIRECTED angle between the peg's tip->flange axis
    (-z of the link frame) and the hole's +z axis. At the correct insertion
    posture dq -> 0; a peg pointing UP (the degenerate posture that would farm
    the alignment bonus) gives dq -> pi and earns nothing. Do NOT use the
    undirected acos(|z . z|): it also assigns 0 to the upside-down peg.
    """
    p_peg, q_peg = peg_tip_pose_w(env)
    p_hole, q_hole = hole_pose_w(env)
    q_inv = quat_conjugate(q_hole)
    p_rel = quat_apply(q_inv, p_peg - p_hole)
    dp = torch.norm(p_rel, dim=-1)

    z_axis = torch.zeros_like(p_peg)
    z_axis[..., 2] = 1.0
    z_peg = -quat_apply(q_peg, z_axis)  # tip -> flange
    z_hole = quat_apply(q_hole, z_axis)
    cos_a = (z_peg * z_hole).sum(dim=-1).clamp(min=-1.0, max=1.0)
    dq = torch.acos(cos_a)
    return dp, dq


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
#
# Carried over verbatim from genpih, where the form was chosen against
# measurements. The short version: the paper's Eq. 5-6 dense term
# 1 - tanh(sqrt(dq^2 + dp^2)) adds radians to metres and flattens to machine
# precision past u ~ 3, so over most of the workspace both the reward and its
# gradient vanish -- a 3000-iteration run under it converged to "aligned but
# 47 mm above the hole" with 0/256 insertions. The replacement keeps the intent
# (closeness to the goal pose) but measures each error in its own task unit and
# gates position credit on alignment:
#
#     r_dense = 1/(1 + dp/DH_POS_UNIT) * 1/(1 + dq/ALIGN_ROT_TOL)
#
# a product rather than an L2 sum, because a misaligned peg cannot advance
# however hard it pushes -- that is the real structure of the task.


def dense_reward(env):
    """r_dense = 1/(1 + dp/DH_POS_UNIT) * 1/(1 + dq/ALIGN_ROT_TOL)."""
    dp, dq = peg_pose_in_hole(env)
    return 1.0 / (1.0 + dp / DH_POS_UNIT) / (1.0 + dq / ALIGN_ROT_TOL)


def parse_reward(env):
    """Sparse reward: 2.6 when aligned, 10 when inserted (genpih paper Eq. 7)."""
    dp, dq = peg_pose_in_hole(env)
    aligned = dq < ALIGN_ROT_TOL
    inserted = aligned & (dp < INSERT_POS_TOL)
    r = torch.zeros_like(dp)
    r = torch.where(aligned, 2.6, r)
    r = torch.where(inserted, 10.0, r)
    return r


def success_term(env):
    """Successful insertion: dq < 0.05 rad and dp < 0.003 m."""
    dp, dq = peg_pose_in_hole(env)
    return (dq < ALIGN_ROT_TOL) & (dp < INSERT_POS_TOL)


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


def hole_pose_in_peg(env):
    """Hole frame expressed in the peg tip frame: (position, quaternion)."""
    p_peg, q_peg = peg_tip_pose_w(env)
    p_hole, q_hole = hole_pose_w(env)
    q_inv = quat_conjugate(q_peg)
    return quat_apply(q_inv, p_hole - p_peg), quat_mul(q_inv, q_hole)


def hole_pos_in_peg(env):
    """Hole origin in the peg tip frame -- the position error the policy servos on."""
    return hole_pose_in_peg(env)[0]


def hole_quat_in_peg(env):
    """Hole orientation in the peg tip frame (wxyz) -- the alignment error."""
    return hole_pose_in_peg(env)[1]


def peg_tip_height(env):
    """Peg tip height above the board top, in metres, shape (N, 1).

    Cheap privileged signal for the critic: unlike dp it does not collapse the
    lateral and vertical errors into one number, so the value function can tell
    "hovering high and centred" from "seated on the board".
    """
    p_peg, _ = peg_tip_pose_w(env)
    board_top = torch.tensor(env.cfg.board_top_z, device=env.device)
    return (p_peg[:, 2] - board_top).unsqueeze(-1)


def camera_image(env, sensor_cfg=None, data_type: str = "rgb"):
    """Eye-in-hand image, scaled to [0, 1] float, shape (N, H, W, C).

    Deliberately does NOT use `isaaclab.envs.mdp.observations.image` with its
    default `normalize=True`, which for rgb divides by 255 and then SUBTRACTS
    THE MEAN OF THE CURRENT BATCH. That makes one environment's observation
    depend on whichever other environments happen to share the batch, so the
    same scene produces a different observation at batch size 1024 than at
    batch size 1 -- exactly the difference between training and a deterministic
    single-environment evaluation. A fixed per-pixel scale keeps the observation
    a pure function of the agent's own view.
    """
    from isaaclab.envs.mdp import observations as obs_mdp
    from isaaclab.managers import SceneEntityCfg

    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("tiled_camera")
    img = obs_mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
    img = img.float() / 255.0

    # Flatten to (N, H*W*C) explicitly. Isaac Lab's ObservationManager does NOT
    # concatenate a group that holds a single term, so the image would otherwise
    # reach the wrapper as (N, H, W, C) -- and RslRlVecEnvWrapper sizes the
    # policy with `group_obs_dim["policy"][0]`, which would read that as H (=100)
    # instead of H*W*C. Row-major, so the network undoes it with
    # view(N, H, W, C).permute(0, 3, 1, 2).
    return img.reshape(img.shape[0], -1)


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def reset_robot_to_home(env, env_ids: torch.Tensor):
    """Restore the fixed home joint configuration (no jitter: this task has no DR).

    Looks joints up BY NAME rather than by column index: the order the USD
    importer emits is not the order these constants are written in, and writing
    the wrong joint is a silent, plausible-looking bug (the arm just starts
    somewhere else and every reachability calculation is off).
    """
    robot = env.scene["robot"]
    names = list(env.cfg.home_joint_names)
    values = torch.tensor(env.cfg.home_joint_values, device=env.device)

    joint_ids, found = robot.find_joints(names, preserve_order=True)
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
    """Place the hole at the fixture pose. Same value every reset -- no DR.

    Written in WORLD coordinates (env origin + fixture), matching genpih's
    approach, which is correct regardless of how the cloner composes the
    authored pose with the per-environment offset.
    """
    hole = env.scene["hole"]
    state = hole.data.root_state_w[env_ids].clone()
    fixture = torch.tensor(env.cfg.fixture_pos, device=env.device)
    state[:, 0:3] = env.scene.env_origins[env_ids] + fixture
    state[:, 3] = 1.0
    state[:, 4:7] = 0.0
    state[:, 7:13] = 0.0
    hole.write_root_state_to_sim(state, env_ids=env_ids)

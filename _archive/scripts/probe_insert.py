"""Focused diagnostic: can a scripted approach actually seat the peg (dp < 3 mm)?

probe_collision.py established two things: the plate's mesh collider DOES stop
the peg (so genpih's USD is usable as-is), and a coarse two-waypoint descent
jams with the tip ~7.6 mm into the bore, dp = 22 mm, 188 N on the plate.

Whether that is a property of the task or of the actuator settings decides
whether the env is well-posed at all: if no scripted descent can reach
dp < 3 mm, the success criterion is unreachable and every training run is
guaranteed to fail. So this script descends in small increments and reports,
at every step, the quantity that actually explains a jam:

    lateral  = distance from the peg TIP axis to the bore axis (not dp, whose
               vertical component hides it)
    tilt     = dq, degrees
    F(hole)  = contact force on the plate

Theory: for a peg of radius r in a bore of radius R (clearance c = R - r =
0.3 mm here) tilted by theta and engaged a depth L, the peg's flank reaches the
bore wall at L ~ c / theta. genpih documents the same threshold
independently ("a peg tilted by dq = 0.043 rad jams at dp ~ 9.5 mm; at 0.01 rad
the same descent reaches dp = 1.2 mm"). If the measured tilt is load-induced
rather than commanded, the actuator's effort limit is the lever to pull.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/probe_insert.py --headless
    $PY scripts/probe_insert.py --headless --effort 330 --label "effort=330"
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Fine-grained scripted insertion probe.")
parser.add_argument("--stiffness", type=float, default=None)
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--effort", type=float, default=None, help="effort_limit_sim; URDF value is 330")
parser.add_argument("--step_mm", type=float, default=1.0, help="descent increment")
parser.add_argument("--dwell", type=int, default=30, help="physics steps to dwell at each increment")
parser.add_argument("--label", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import kin  # noqa: E402
import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
import pegih_tasks.tasks.peg_hole.mdp as mdp  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import (  # noqa: E402
    BOARD_TOP_Z,
    FIXTURE_POS,
    HOME_JOINT_ORDER,
    PLATE_TOP_Z,
    PegHoleEnvCfg,
)

HOLE_R = 0.0215


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    # these probes are about physics, not pixels: drop the camera so they do
    # not require --enable_cameras
    env_cfg.enable_vision(False)
    act = env_cfg.scene.robot.actuators["arm"]
    if args_cli.stiffness is not None:
        act.stiffness = args_cli.stiffness
    if args_cli.damping is not None:
        act.damping = args_cli.damping
    if args_cli.effort is not None:
        act.effort_limit_sim = args_cli.effort

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    contact = env.scene["contact"]
    dt = env.sim.get_physics_dt()
    joint_ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    home = np.array(kin.HOME_JOINTS)

    label = args_cli.label or f"k={act.stiffness:.0f} d={act.damping:.0f} effort={act.effort_limit_sim:.0f}"
    print(f"\n### {label}")
    print(f"    actuator: stiffness={act.stiffness} damping={act.damping} effort_limit_sim={act.effort_limit_sim}")

    def command(q_urdf):
        tgt = robot.data.joint_pos.clone()
        tgt[0, joint_ids] = torch.tensor(q_urdf, device=env.device, dtype=tgt.dtype)
        robot.set_joint_position_target(tgt)

    def spin(n):
        for _ in range(n):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)

    fx, fy = FIXTURE_POS[0], FIXTURE_POS[1]

    def probe():
        tip = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]]
        dp, dq = mdp.peg_pose_in_hole(env)
        fm = contact.data.force_matrix_w[0, 0]
        # body_pos_w is world; FIXTURE_POS is in the robot base frame. Isaac Lab
        # lays envs out on a grid, so drop the env origin before comparing xy.
        tip_local = tip[:2] - env.scene.env_origins[0, :2]
        lateral = float(torch.linalg.norm(tip_local - torch.tensor([fx, fy], device=env.device)))
        return float(tip[2]), lateral, float(dp[0]), float(dq[0]), \
            float(torch.norm(fm[0])), float(torch.norm(fm[1]))

    spin(200)
    # go to a safe hover above the bore first, then descend in small increments
    start_z = PLATE_TOP_Z + 0.02
    q, err, tilt = kin.ik_tip((fx, fy, start_z), q0=home)
    assert err < 1e-4, f"IK failed: {err}"
    command(q)
    spin(120)

    print(f"    bore axis at ({fx:.4f}, {fy:.4f});  plate_top={PLATE_TOP_Z:.4f}  board_top={BOARD_TOP_Z:.4f}")
    print(f"    clearance = {(HOLE_R - 0.0212)*1000:.1f} mm radial;  jam depth ~ clearance / tilt")
    print()
    print(f"    {'cmd_z':>8s} {'tip_z':>8s} {'z_err':>8s} {'lateral':>9s} {'tilt':>7s} "
          f"{'dp':>8s} {'F(hole)':>9s} {'F(board)':>9s}")

    zs = np.arange(start_z, BOARD_TOP_Z - 0.003, -args_cli.step_mm / 1000.0)
    best_dp = 1e9
    for i, z in enumerate(zs):
        q, err, _ = kin.ik_tip((fx, fy, z), q0=q)
        command(q)
        spin(args_cli.dwell)
        tip_z, lat, dp, dq, fh, fb = probe()
        best_dp = min(best_dp, dp)
        flag = ""
        if fh > 1.0 and dp < best_dp + 1e-9 and i > 0:
            flag = "  <- first real contact"
        print(f"    {z:8.4f} {tip_z:8.4f} {(tip_z-z)*1000:+7.2f}m {lat*1000:8.3f}m "
              f"{np.degrees(dq):6.3f}d {dp*1000:7.3f}m {fh:8.3f}N {fb:8.3f}N{flag}")

    tip_z, lat, dp, dq, fh, fb = probe()
    print()
    print(f"    RESULT  best dp = {best_dp*1000:.3f} mm   final tip_z = {tip_z:.4f} "
          f"(board top {BOARD_TOP_Z:.4f})  lateral = {lat*1000:.3f} mm  tilt = {np.degrees(dq):.3f} deg")
    print(f"    SEATED (dp < 3 mm): {best_dp < 0.003}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

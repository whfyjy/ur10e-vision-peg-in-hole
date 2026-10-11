"""Scripted approach-and-insert demo, for watching the scene work.

No policy: it drives the peg with IK through the same motions the task expects
-- hover, lateral move over the bore, gentle descent until the board stops it --
and prints dp/dq and the contact force at each stage so the physics is visible
alongside the picture.

The descent is deliberately in 1 mm increments with a dwell. A coarse two-waypoint
descent jams: the peg is 42.4 mm in a 43 mm bore (0.3 mm radial clearance), so a
large single command overshoots into the chamfer, the servo loads up, the peg
tilts, and the flank bites the bore wall. Incremental descent seats it.

    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    DISPLAY=:1 $PY scripts/gui_demo.py --gui
    $PY scripts/gui_demo.py --headless          # same run, no viewport
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Scripted peg-in-hole demo.")
parser.add_argument("--hold_s", type=float, default=8.0, help="seconds to hold at each stage")
parser.add_argument("--dwell", type=int, default=30, help="physics steps per descent increment")
parser.add_argument("--increment_mm", type=float, default=1.0)
# AppLauncher supplies --headless (viewport is shown when it is absent). This is
# only an explicit alias for that default, so the invocation reads the way the
# docs write it.
parser.add_argument("--gui", action="store_true", help="show the viewport (same as omitting --headless)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False

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


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    # no camera: this demo is about the motion, and rendering only slows it down
    env_cfg.enable_vision(False)

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    contact = env.scene["contact"]
    dt = env.sim.get_physics_dt()
    joint_ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    home = np.array(kin.HOME_JOINTS)

    if not args_cli.headless:
        try:
            import isaacsim.core.utils.viewports as vp_util

            vp_util.frame_viewport_prims(prims=["/World/envs/env_0/Robot", "/World/envs/env_0/Board"])
        except Exception as e:  # framing is cosmetic; never fail the demo for it
            print(f"  (viewport framing skipped: {e})")

    def command(q_urdf):
        tgt = robot.data.joint_pos.clone()
        tgt[0, joint_ids] = torch.tensor(q_urdf, device=env.device, dtype=tgt.dtype)
        robot.set_joint_position_target(tgt)

    def spin(steps):
        for _ in range(steps):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)

    def report(tag):
        tip = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]]
        dp, dq = mdp.peg_pose_in_hole(env)
        fm = contact.data.force_matrix_w[0, 0]
        fh, fb = float(torch.norm(fm[0])), float(torch.norm(fm[1]))
        print(f"  {tag:26s} tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})  "
              f"dp={float(dp[0])*1000:7.3f} mm  dq={np.degrees(float(dq[0])):5.2f} deg  "
              f"F(hole)={fh:7.2f} N  F(board)={fb:7.2f} N")
        return float(dp[0])

    fx, fy = FIXTURE_POS[0], FIXTURE_POS[1]
    hold = max(1, int(args_cli.hold_s / dt))

    print(f"\nfixture at ({fx:.4f}, {fy:.4f}); plate top {PLATE_TOP_Z:.4f}; board top {BOARD_TOP_Z:.4f}")
    print("peg tip xy at home is (0.1037, 0.7054), so the hole starts "
          f"{np.hypot(fx-0.1037, fy-0.7054)*1000:.0f} mm away from the peg\n")

    spin(240)
    report("at home")

    q = home.copy()
    stages = [
        ("hover above the bore", (fx, fy, PLATE_TOP_Z + 0.05)),
        ("descend to the chamfer", (fx, fy, PLATE_TOP_Z + 0.005)),
    ]
    for tag, target in stages:
        q, err, _ = kin.ik_tip(target, q0=q)
        assert err < 1e-4, f"IK failed for {tag}: {err}"
        command(q)
        spin(hold)
        report(tag)

    print("\n  descending into the bore in "
          f"{args_cli.increment_mm:.1f} mm increments ...")
    best = 1e9
    for z in np.arange(PLATE_TOP_Z, BOARD_TOP_Z - 0.003, -args_cli.increment_mm / 1000.0):
        q, _, _ = kin.ik_tip((fx, fy, float(z)), q0=q)
        command(q)
        spin(args_cli.dwell)
        best = min(best, report(f"  z_target={z:.4f}"))

    report("seated")
    from pegih_tasks.tasks.peg_hole.mdp import INSERT_POS_TOL

    print(f"\n  best dp = {best*1000:.3f} mm  ->  success criterion (dp < {INSERT_POS_TOL*1000:.0f} mm): "
          f"{'REACHED' if best < INSERT_POS_TOL else 'NOT REACHED'}")
    print(f"  holding for {args_cli.hold_s:.0f} s for inspection ...\n")
    spin(hold)

    safe_close(env, label="env")


if __name__ == "__main__":
    main()
    simulation_app.close()

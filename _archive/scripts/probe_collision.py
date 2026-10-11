"""Resolve the one unknown that decides the whole collision design.

Two projects in this workspace disagree about the same PhysX setup:

  - ur10_peg_hole/README.md: "PhysX generates no contacts between the kinematic
    body's concave triangle mesh and the peg's convex hull (measured: the peg
    fell straight through the plate)" -- which is why that project rebuilt its
    hole plate out of 20 overlapping convex boxes.
  - genpih uses exactly that setup (kinematic rigid body, mesh collider with
    collision_approximation="none", concave plate mesh) and its docs report the
    peg resting ON the plate's top face.

Both cannot be true. This script drives the peg to two places and reports the
measured contact forces:

  A. press down on the plate's TOP FACE, offset from the bore
     -> if the plate collides, the peg stops at plate_top with peg<->hole force
  B. descend through the BORE to the wooden board
     -> the peg must pass the plate with ~zero side force and stop on the board

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/probe_collision.py --headless
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Peg/plate/board collision probe.")
parser.add_argument("--settle", type=int, default=200, help="physics steps to settle before probing.")
parser.add_argument("--steps_per_waypoint", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import kin  # noqa: E402
import pegih_tasks.tasks.peg_hole  # noqa: F401, E402  (registers the gym id)
import pegih_tasks.tasks.peg_hole.mdp as mdp  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import (  # noqa: E402
    BOARD_TOP_Z,
    FIXTURE_POS,
    HOME_JOINT_ORDER,
    PLATE_TOP_Z,
    PegHoleEnvCfg,
)

PLATE_TOP = PLATE_TOP_Z
HOLE_R = 0.0215
PEG_R = 0.0212


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    # these probes are about physics, not pixels: drop the camera so they do
    # not require --enable_cameras
    env_cfg.enable_vision(False)
    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    contact = env.scene["contact"]
    dt = env.sim.get_physics_dt()

    # map URDF-ordered joint vectors into the robot's own column order
    joint_ids, found = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    assert list(found) == list(HOME_JOINT_ORDER), f"joint order mismatch: {found}"
    home = np.array(kin.HOME_JOINTS)

    def command(q_urdf):
        tgt = robot.data.joint_pos.clone()
        tgt[0, joint_ids] = torch.tensor(q_urdf, device=env.device, dtype=tgt.dtype)
        robot.set_joint_position_target(tgt)

    def spin(n):
        for _ in range(n):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)

    def probe():
        tip = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]]
        dp, dq = mdp.peg_pose_in_hole(env)
        fm = contact.data.force_matrix_w[0, 0]           # (num_filters, 3)
        return (
            float(tip[2]),
            float(dp[0]),
            float(dq[0]),
            float(torch.norm(fm[0])),                     # -> Hole
            float(torch.norm(fm[1])),                     # -> Board
        )

    print(f"\n[geom] board_top={BOARD_TOP_Z:.4f}  plate_top={PLATE_TOP:.4f}  fixture={FIXTURE_POS}")
    print(f"       bore r={HOLE_R}  peg r={PEG_R}  diametral clearance={2*(HOLE_R-PEG_R)*1000:.1f} mm")

    spin(args_cli.settle)
    z, dp, dq, fh, fb = probe()
    print(f"\n[home] tip_z={z:.4f}  dp={dp*1000:8.3f} mm  dq={np.degrees(dq):7.3f} deg  "
          f"F(hole)={fh:7.3f} N  F(board)={fb:7.3f} N")

    results = {}

    def run_segment(label, waypoints):
        q = home.copy()
        for i, wp in enumerate(waypoints):
            q, perr, tilt = kin.ik_tip(wp, q0=q)
            if perr > 1e-4:
                print(f"  !! IK failed for waypoint {i}: err={perr*1000:.3f} mm")
            command(q)
            spin(args_cli.steps_per_waypoint)
            z, dp, dq, fh, fb = probe()
            print(f"  {label} wp{i} target_z={wp[2]:.4f}  tip_z={z:8.4f}  tip_err={(z-wp[2])*1000:+8.3f} mm  "
                  f"dp={dp*1000:8.3f} mm  dq={np.degrees(dq):6.3f} deg  "
                  f"F(hole)={fh:7.3f}  F(board)={fb:7.3f}")
        return q

    # ---- A: press onto the plate's top face, clear of the bore -------------
    ax, ay = FIXTURE_POS[0] + 0.06, FIXTURE_POS[1]
    print(f"\n[A] press on the plate top face at ({ax:.4f}, {ay:.4f}); "
          f"peg edge sits {0.06-PEG_R:.4f} m from the bore axis (bore r={HOLE_R})")
    run_segment("A", [(ax, ay, PLATE_TOP + 0.03), (ax, ay, PLATE_TOP - 0.005)])
    z, dp, dq, fh, fb = probe()
    results["plate_stops_peg"] = fh > 0.5

    # retract before moving sideways
    command(kin.ik_tip((ax, ay, PLATE_TOP + 0.05))[0])
    spin(args_cli.steps_per_waypoint)

    # ---- B: descend through the bore onto the board ------------------------
    print(f"\n[B] descend through the bore at ({FIXTURE_POS[0]:.4f}, {FIXTURE_POS[1]:.4f}) "
          f"to the board at z={BOARD_TOP_Z:.4f}")
    run_segment("B", [(FIXTURE_POS[0], FIXTURE_POS[1], PLATE_TOP + 0.03),
                      (FIXTURE_POS[0], FIXTURE_POS[1], BOARD_TOP_Z + 0.003),
                      (FIXTURE_POS[0], FIXTURE_POS[1], BOARD_TOP_Z - 0.002)])
    z, dp, dq, fh, fb = probe()
    results["board_stops_peg"] = fb > 0.5
    results["dp_at_bottom_ok"] = dp < 0.003
    results["no_side_force"] = fh < 0.5
    bottom = (z, dp, dq, fh, fb)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  plate stops the peg (A, F_hole > 0.5 N)      : {results['plate_stops_peg']}   "
          f"(plate mesh collider {'WORKS -- keep genpih USD as-is' if results['plate_stops_peg'] else 'DOES NOT COLLIDE -- need convex-box ring'})")
    print(f"  board stops the peg (B, F_board > 0.5 N)     : {results['board_stops_peg']}")
    print(f"  peg passes the bore cleanly (B, F_hole small): {results['no_side_force']}")
    print(f"  dp at the bottom (B) < 3 mm                  : {results['dp_at_bottom_ok']}   "
          f"(dp={bottom[1]*1000:.3f} mm, tip_z={bottom[0]:.4f}, plate_top={PLATE_TOP:.4f})")
    print("=" * 72)
    print(f"  bottom state: tip_z={bottom[0]:.4f} dp={bottom[1]*1000:.3f} mm "
          f"dq={np.degrees(bottom[2]):.3f} deg F(hole)={bottom[3]:.3f} F(board)={bottom[4]:.3f} N")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Render the eye-in-hand view at the poses that matter, and save PNGs.

The camera offset is derived analytically by scripts/cam_offset.py, but the only
way to know a view is right is to look at it: this renders HOME (episode start),
mid-descent, and seated on the board, and reports the policy observation shape
so the vision path is checked at the same time.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/cam_calib.py --headless --out /tmp/cam
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import numpy as np  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Eye-in-hand camera calibration renders.")
parser.add_argument("--out", type=str, default="/tmp/cam")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--lateral", type=float, default=None,
                    help="override: camera offset from the tool axis along the wrist -x, metres")
parser.add_argument("--along", type=float, default=None,
                    help="override: camera offset down the tool axis, metres")
parser.add_argument("--look", type=float, default=0.5,
                    help="0 = aim at the peg tip, 1 = aim at the hole (default 0.5, midway)")
AppLauncher.add_app_launcher_args(parser)
# a camera task cannot run without this
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import kin  # noqa: E402
import cam_offset as co  # noqa: E402
import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import (  # noqa: E402
    BOARD_TOP_Z,
    FIXTURE_POS,
    HOME_JOINT_ORDER,
    PLATE_TOP_Z,
    PegHoleEnvCfg,
)


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(True)

    # Optional view override, computed BEFORE the env is built because the camera
    # prim is spawned with whatever offset the cfg carries. The wrist frame comes
    # from the URDF chain at HOME (verified against genpih's measured home pose),
    # not from the simulation, which does not exist yet.
    if args_cli.lateral is not None or args_cli.along is not None:
        lateral = 0.07 if args_cli.lateral is None else args_cli.lateral
        along = 0.0 if args_cli.along is None else args_cli.along
        w_pos, w_R = co.wrist_frame(kin.HOME_JOINTS)
        tip = kin.tip_pos(kin.HOME_JOINTS)
        hole = np.array(FIXTURE_POS)
        cam = w_pos + (-lateral) * w_R[:, 0] + along * w_R[:, 2]
        tgt = tip + args_cli.look * (hole - tip)
        pos_off, quat = co.offset_from_world(cam, tgt)
        env_cfg.scene.tiled_camera.offset.pos = tuple(float(v) for v in pos_off)
        env_cfg.scene.tiled_camera.offset.rot = tuple(float(v) for v in quat)
        ang = co.frame_angles(cam, tgt, {"tip": tip, "hole": hole})
        print(f"\n[override] lateral={lateral:.3f} along={along:.3f} look={args_cli.look:.2f}")
        print(f"           cam_world={np.round(cam, 4)}  offset_pos={np.round(pos_off, 5)}")
        print(f"           tip {ang['tip'][0]:6.2f} deg off-axis (d={ang['tip'][1]:.3f} m)   "
              f"hole {ang['hole'][0]:6.2f} deg (d={ang['hole'][1]:.3f} m)   HFOV/2={co.HFOV_DEG/2:.1f}")

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.scene["robot"]
    cam = env.scene["tiled_camera"]
    dt = env.sim.get_physics_dt()
    joint_ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    home = np.array(kin.HOME_JOINTS)

    print(f"\n[camera] prim={cam.cfg.prim_path}")
    print(f"         offset pos={cam.cfg.offset.pos} rot={cam.cfg.offset.rot} convention={cam.cfg.offset.convention}")
    print(f"         resolution={cam.cfg.width}x{cam.cfg.height}  focal={cam.cfg.spawn.focal_length} cm")
    print(f"[obs]    policy group shape = {tuple(obs['policy'].shape)}  "
          f"(expected ({args_cli.num_envs}, {cam.cfg.height*cam.cfg.width*3}))")
    if "critic" in env.observation_manager.group_obs_dim:
        print(f"         critic group shape = {tuple(obs['critic'].shape)}")
    else:
        print("         critic group: absent (symmetric vision network)")

    os.makedirs(args_cli.out, exist_ok=True)

    def command(q_urdf):
        tgt = robot.data.joint_pos.clone()
        tgt[:, joint_ids] = torch.tensor(q_urdf, device=env.device, dtype=tgt.dtype)
        robot.set_joint_position_target(tgt)

    def spin(n):
        for _ in range(n):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)
        cam.update(dt)

    def save(tag):
        rgb = cam.data.output["rgb"]
        tip = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]]
        for i in range(min(args_cli.num_envs, 4)):
            img = rgb[i].cpu().numpy()
            path = os.path.join(args_cli.out, f"{tag}_env{i}.png")
            Image.fromarray(img).resize((400, 400), Image.NEAREST).save(path)
            print(f"  wrote {path}   mean={img.mean():6.1f} std={img.std():5.1f} "
                  f"min={img.min():3d} max={img.max():3d}")
        print(f"  [{tag}] env0 tip_z={float(tip[2]):.4f}")

    spin(200)
    save("1_home")

    fx, fy = FIXTURE_POS[0], FIXTURE_POS[1]
    q = home.copy()
    for tag, z in [("2_hover", PLATE_TOP_Z + 0.03), ("3_seated", BOARD_TOP_Z)]:
        q, err, _ = kin.ik_tip((fx, fy, z), q0=q)
        assert err < 1e-4, f"IK failed for {tag}: {err}"
        command(q)
        spin(150)
        save(tag)

    safe_close(env, label="env")
    print("\nDONE. Inspect the PNGs: the peg should be in frame at home, and the "
          "hole should be visible throughout and centred as the peg seats.\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

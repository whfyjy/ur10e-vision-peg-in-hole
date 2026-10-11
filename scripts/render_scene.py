"""Render the scene from fixed external viewpoints, plus the policy's own view.

For looking at the environment rather than training it. Produces PNGs of:
  1_overview      - the whole setup in third person
  2_board_closeup - the plate, the bore, and the board it sits on
  3_elevation     - side view, which is the one that shows the board top sitting
                    level with the top of the robot's base casting
  4_eyetohand_*   - what the policy actually sees, at home / hover / seated
  5_insert_*      - the same four moments from the overview camera, in sequence

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/render_scene.py --headless --out /tmp/scene
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Render the peg-in-hole scene.")
parser.add_argument("--out", type=str, default="/tmp/scene")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import cam_offset as co  # noqa: E402
import kin  # noqa: E402
import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import (  # noqa: E402
    BOARD_TOP_Z,
    FIXTURE_POS,
    HOME_JOINT_ORDER,
    PLATE_TOP_Z,
    PegHoleEnvCfg,
)

from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(True)

    # External cameras, purely so a human can look at the scene; they are not part
    # of the task. Placed with an explicit OffsetCfg rather than by calling
    # set_world_poses() at run time -- that call did not take effect before the
    # render (three shots came out byte-identical, all from the spawn pose).
    # convention="opengl" means the offset rotation IS the camera orientation in
    # the parent frame (-Z forward, +Y up), and the parent here is /World, so the
    # numbers are plain world poses.
    fx, fy = FIXTURE_POS[0], FIXTURE_POS[1]
    VIEWS = {
        "overview": ((1.45, -0.35, 1.15), (0.28, 0.55, 0.35)),
        "board_closeup": ((0.95, 0.42, 0.72), (fx, fy, PLATE_TOP_Z - 0.02)),
        # looking horizontally along -x: the view that shows the board top sitting
        # level with the top of the robot's base casting
        "elevation": ((1.30, fy + 0.02, BOARD_TOP_Z + 0.10), (0.02, fy, BOARD_TOP_Z + 0.09)),
    }
    for name, (pos, tgt) in VIEWS.items():
        q = co.quat_from_matrix(co.look_at_rotation(pos, tgt))
        env_cfg.scene.__dict__[f"cam_{name}"] = CameraCfg(
            prim_path=f"/World/cam_{name}",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(pos=pos, rot=tuple(float(v) for v in q), convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 100.0)),
        )

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    cam_eye = env.scene["tiled_camera"]
    dt = env.sim.get_physics_dt()
    joint_ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    home = np.array(kin.HOME_JOINTS)

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
        cam_eye.update(dt)
        for name in VIEWS:
            env.scene[f"cam_{name}"].update(dt)

    def shoot(sensor, tag, note=""):
        rgb = sensor.data.output["rgb"][0].cpu().numpy()
        path = os.path.join(args_cli.out, f"{tag}.png")
        Image.fromarray(rgb).save(path)
        print(f"  {path}   ({rgb.shape[1]}x{rgb.shape[0]})  {note}")

    spin(200)

    print("\n--- 外部机位 ---")
    shoot(env.scene["cam_overview"], "1_overview", "全景：机械臂 + 木板 + 孔板")
    shoot(env.scene["cam_board_closeup"], "2_board_closeup", "孔板与孔")
    shoot(env.scene["cam_elevation"], "3_elevation", "侧视：木板顶面与基座顶面等高")

    print("\n--- 策略视角（眼在手上）---")
    shoot(cam_eye, "4_eyetohand_home", "episode 起点")

    seq = [("hover", (fx, fy, PLATE_TOP_Z + 0.04)),
           ("chamfer", (fx, fy, PLATE_TOP_Z + 0.005)),
           ("seated", (fx, fy, BOARD_TOP_Z))]
    q = home.copy()
    for tag, target in seq:
        q, err, _ = kin.ik_tip(target, q0=q)
        assert err < 1e-4, f"IK failed for {tag}"
        command(q)
        spin(160)
        shoot(cam_eye, f"4_eyetohand_{tag}")
        shoot(env.scene["cam_overview"], f"5_insert_{tag}", "全景机位的同一时刻")

    safe_close(env, label="env")
    print(f"\n全部写入 {args_cli.out}\n")


if __name__ == "__main__":
    import threading

    main()
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=20)
    os._exit(0)

"""Open the scene and just stand there, so it can be inspected.

No policy, no scripted motion -- this resets once and then holds, printing the
numbers that define the starting state. Use the mouse in the viewport to orbit,
pan and zoom while it holds.

    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    DISPLAY=:1 $PY scripts/show_scene.py --gui
    $PY scripts/show_scene.py --headless --hold_s 0        # just print the numbers

Press Ctrl+C in the terminal to quit.
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Show the initial state of the scene.")
parser.add_argument("--hold_s", type=float, default=1e9, help="seconds to hold (default: forever)")
parser.add_argument("--settle", type=int, default=200, help="physics steps before reporting")
parser.add_argument("--gui", action="store_true", help="show the viewport (same as omitting --headless)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import kin  # noqa: E402
import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
import pegih_tasks.tasks.peg_hole.mdp as mdp  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import (  # noqa: E402
    BOARD_SIZE,
    BOARD_TOP_Z,
    FIXTURE_POS,
    HOME_JOINTS,
    HOME_TIP_XY,
    PLATE_THICKNESS,
    PLATE_TOP_Z,
    START_OFFSET_XY,
    PegHoleEnvCfg,
)


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    # no camera: this is a scene inspection, and it keeps startup light
    env_cfg.enable_vision(False)

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped

    if not args_cli.headless:
        try:
            import isaacsim.core.utils.viewports as vp_util

            vp_util.frame_viewport_prims(prims=["/World/envs/env_0/Robot", "/World/envs/env_0/Board",
                                                "/World/envs/env_0/Pedestal"])
        except Exception as e:
            print(f"  (viewport framing skipped: {e})")

    env.reset()
    dt = env.sim.get_physics_dt()
    for _ in range(args_cli.settle):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt)

    robot = env.scene["robot"]
    origin = env.scene.env_origins[0].cpu().numpy()
    tip_world = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]].cpu().numpy()
    tip = tip_world - origin                      # base_link frame
    dp, dq = mdp.peg_pose_in_hole(env)

    print("\n" + "=" * 68)
    print("初始状态（回合开始那一刻，未训练、未动作）")
    print("=" * 68)
    print(f"  机器人       UR10e，基座固定在原点 (base_link 系)")
    print(f"  初始关节角   {[round(np.degrees(v), 1) for v in HOME_JOINTS.values()]} deg")
    print(f"               (肩 67.5°, 其余 -90/+90, 腕3 = 0 —— 轴销严格垂直)")
    print(f"  peg 尖端     [{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]  (base_link 系)")
    print(f"               垂直度误差 {kin.tilt_deg(np.array(list(HOME_JOINTS.values()))):.4f}°")
    print(f"  木板顶面     z = {BOARD_TOP_Z:.4f}   ({BOARD_SIZE[0]}x{BOARD_SIZE[1]}x{BOARD_SIZE[2]} m，实心、固定)")
    print(f"  孔板顶面     z = {PLATE_TOP_Z:.4f}   (厚 {PLATE_THICKNESS} m，Ø43 通孔)")
    print(f"  孔坐标系原点 [{FIXTURE_POS[0]:.4f}, {FIXTURE_POS[1]:.4f}, {FIXTURE_POS[2]:.4f}]")
    print(f"               (孔底圆心，就在木板顶面上)")
    print(f"  横向偏移     {START_OFFSET_XY[0]*1000:.0f} mm  ->  peg 尖端 xy = "
          f"({HOME_TIP_XY[0]:.4f}, {HOME_TIP_XY[1]:.4f})，孔 xy = "
          f"({FIXTURE_POS[0]:.4f}, {FIXTURE_POS[1]:.4f})")
    print(f"  起始误差     dp = {float(dp[0])*1000:.1f} mm  (几乎全是竖直高度)")
    print(f"               dq = {np.degrees(float(dq[0])):.2f}°  (已经对准，只需横移+下压)")
    print(f"  成功判据     dp < 3 mm 且 dq < 2.86°")
    print("=" * 68)

    if args_cli.hold_s > 0 and args_cli.hold_s < 1e8:
        print(f"\n  保持 {args_cli.hold_s:.0f} 秒 ...")
        time.sleep(args_cli.hold_s)
    elif not args_cli.headless:
        print("\n  视口已打开，可以鼠标拖动旋转/缩放。终端按 Ctrl+C 退出。\n")
        try:
            while True:
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(dt)
        except KeyboardInterrupt:
            print("\n  退出。")

    safe_close(env, label="env")


if __name__ == "__main__":
    import threading

    main()
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=20)
    os._exit(0)

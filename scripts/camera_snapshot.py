"""How to get the camera data, demonstrated end to end.

There are two levels, and they are the same pixels:

  1. the sensor buffer   -- env.scene["tiled_camera"].data.output["rgb"]
                            (N, H, W, 3) uint8 on the GPU
  2. the policy          -- mdp.camera_image(env)
     observation             (N, H*W*C) float32 in [0, 1], row-major over (H, W, C)
                            so it undoes exactly with .view(H, W, C)

This script prints shape/dtype/range at each level, checks that level 2 really is
level 1 scaled, and (if a checkpoint is given) drives the arm with the trained
policy while dumping frames to PNG.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/camera_snapshot.py --headless --out /tmp/frames --frames 8
    $PY scripts/camera_snapshot.py --headless --out /tmp/frames \
        --checkpoint runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_550.pt
"""

import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Read the eye-in-hand camera and dump frames.")
parser.add_argument("--out", type=str, default="/tmp/frames")
parser.add_argument("--frames", type=int, default=8, help="how many frames to save")
parser.add_argument("--every", type=int, default=20, help="policy steps between saved frames")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="optional: drive with a trained policy instead of random actions")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
import pegih_tasks.tasks.peg_hole.mdp as mdp  # noqa: E402
from pegih_tasks.tasks.peg_hole.cnn_policy import install as install_cnn_policy, load_checkpoint  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import IMAGE_HEIGHT, IMAGE_WIDTH, PegHoleEnvCfg  # noqa: E402


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(True)

    # gym.make() creates a SimulationContext, and there can only be one per
    # process -- calling it twice dies with "Simulation context already exists".
    # So the env is created exactly once here, and the rsl_rl wrapper (which the
    # runner insists on) wraps THIS instance rather than a second one.
    gym_env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg)
    env = gym_env.unwrapped
    obs, _ = env.reset()

    cam = env.scene["tiled_camera"]
    print(f"\n[camera prim] {cam.cfg.prim_path}")
    print(f"              mounted on wrist_3_link, offset {cam.cfg.offset.pos} "
          f"convention={cam.cfg.offset.convention}")
    print(f"              {cam.cfg.width}x{cam.cfg.height}  data_types={cam.cfg.data_types}  "
          f"update_period={cam.cfg.update_period:.5f} s (one image per policy step)")

    # ---- level 1: the raw sensor buffer -------------------------------------
    rgb = cam.data.output["rgb"]
    print(f"\n[level 1] cam.data.output['rgb']")
    print(f"          shape={tuple(rgb.shape)}  dtype={rgb.dtype}  device={rgb.device}  "
          f"range=[{int(rgb.min())}, {int(rgb.max())}]")

    # ---- level 2: what the policy is actually fed ---------------------------
    pol = mdp.camera_image(env)
    print(f"\n[level 2] mdp.camera_image(env)   <- the policy observation")
    print(f"          shape={tuple(pol.shape)}  dtype={pol.dtype}  device={pol.device}  "
          f"range=[{float(pol.min()):.3f}, {float(pol.max()):.3f}]")

    # ---- they are the same pixels ------------------------------------------
    back = pol.view(args_cli.num_envs, IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255.0
    err = (back - rgb.float()).abs().max().item()
    print(f"\n[check]   policy observation rebuilt as an image differs from the raw buffer "
          f"by at most {err:.4f} / 255  -> {'OK (same pixels)' if err < 1.0 else 'MISMATCH'}")

    # ---- optional: drive with a trained policy ------------------------------
    policy = None
    if args_cli.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from pegih_tasks.config.rsl_rl_ppo_cfg import PegIhPPORunnerCfg

        install_cnn_policy()
        wrapped = RslRlVecEnvWrapper(gym_env)          # wraps the SAME env
        agent_cfg = PegIhPPORunnerCfg()
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_checkpoint(runner, args_cli.checkpoint)
        policy = runner.alg.policy
        obs, _ = env.reset()
        print(f"\n[policy]  driving with {os.path.basename(args_cli.checkpoint)}")

    if os.path.isdir(args_cli.out):
        shutil.rmtree(args_cli.out)
    os.makedirs(args_cli.out, exist_ok=True)

    for k in range(args_cli.frames):
        for _ in range(args_cli.every):
            if policy is not None:
                with torch.no_grad():
                    a = policy.act_inference(obs["policy"])
            else:
                a = torch.clamp(torch.randn(args_cli.num_envs, 6, device=env.device) * 0.3, -1, 1)
            # raw ManagerBasedRLEnv.step: gymnasium 5-tuple, obs is a dict.
            # (RslRlVecEnvWrapper.step returns 4 -- see eval.py.)
            obs, rew, term, trunc, extras = env.step(a)

        rgb = cam.data.output["rgb"]
        dp, dq = mdp.peg_pose_in_hole(env)
        for i in range(min(args_cli.num_envs, 2)):
            arr = rgb[i].cpu().numpy()
            path = os.path.join(args_cli.out, f"frame_{k:03d}_env{i}.png")
            Image.fromarray(arr).resize((300, 300), Image.NEAREST).save(path)
        print(f"  frame {k:02d}  step {(k+1)*args_cli.every:4d}  "
              f"env0 dp={float(dp[0])*1000:8.3f} mm  dq={np.degrees(float(dq[0])):6.2f} deg  "
              f"pixel mean={rgb[0].float().mean():6.1f}")

    print(f"\nwrote {args_cli.frames} frames x {min(args_cli.num_envs,2)} envs to {args_cli.out}")
    print("each PNG is the raw 100x100 buffer upscaled 3x for viewing\n")
    safe_close(env, label="env")


if __name__ == "__main__":
    import threading

    main()
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=20)
    os._exit(0)

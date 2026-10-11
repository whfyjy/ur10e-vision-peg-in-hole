"""Watch the TRAINED policy do the insertion, in the Isaac Sim viewport.

gui_demo.py drives the arm with scripted IK; this drives it with the network --
image in, joint targets out, with nothing telling it where the hole is. It is
the same code path as evaluation, just rendered.

    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    DISPLAY=:1 $PY scripts/gui_policy_demo.py --gui \
        --checkpoint runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_550.pt

    # state-observation policy (no camera, runs lighter) -- add --no-vision
    DISPLAY=:1 $PY scripts/gui_policy_demo.py --gui --no-vision \
        --checkpoint runs/rsl_rl/pegih_vision/2026-10-09_16-00-10_state_long_resume/model_3875.pt

Add --loops N to replay several episodes (the scene is deterministic, so unless
you perturb it they will look identical).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Viewport demo driven by a trained policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--loops", type=int, default=1, help="episodes to replay")
parser.add_argument("--report_every", type=int, default=30, help="policy steps between status lines")
parser.add_argument("--hold_s", type=float, default=4.0, help="seconds to hold after the last episode")
parser.add_argument("--no-vision", action="store_true")
parser.add_argument("--symmetric-critic", action="store_true")
parser.add_argument("--gui", action="store_true", help="show the viewport (same as omitting --headless)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False
if not args_cli.no_vision:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import pegih_tasks.tasks.peg_hole  # noqa: F401, E402
import pegih_tasks.tasks.peg_hole.mdp as mdp  # noqa: E402
from pegih_tasks.tasks.peg_hole.cnn_policy import install as install_cnn_policy, load_checkpoint  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import PegHoleEnvCfg  # noqa: E402
from pegih_tasks.config.rsl_rl_ppo_cfg import PegIhPPORunnerCfg  # noqa: E402


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(not args_cli.no_vision, privileged_critic=not args_cli.symmetric_critic)

    install_cnn_policy()

    gym_env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env)
    raw = env.unwrapped

    agent_cfg = PegIhPPORunnerCfg()
    agent_cfg.device = args_cli.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, args_cli.checkpoint)
    policy = runner.alg.policy

    if not args_cli.headless:
        try:
            import isaacsim.core.utils.viewports as vp_util

            vp_util.frame_viewport_prims(prims=["/World/envs/env_0/Robot", "/World/envs/env_0/Board"])
        except Exception as e:  # framing is cosmetic
            print(f"  (viewport framing skipped: {e})")

    print(f"\n>>> policy   : {os.path.basename(args_cli.checkpoint)}")
    print(f">>> input    : {'camera image 100x100x3' if not args_cli.no_vision else '13-dim state'}")
    print(f">>> viewport : {'on' if not args_cli.headless else 'off (headless)'}")
    print(f">>> success criterion: dq < 0.05 rad and dp < 0.003 m\n")
    print(f"    {'step':>5s} {'dp(mm)':>9s} {'dq(deg)':>9s}   status")

    for loop in range(args_cli.loops):
        obs, _ = env.reset()
        policy.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
        if args_cli.loops > 1:
            print(f"\n  --- episode {loop + 1}/{args_cli.loops} ---")

        seated_at = None
        # RslRlVecEnvWrapper auto-resets an environment on done, so any state read
        # AFTER the step that ends the episode is the post-reset state -- reading
        # dp there reports the home distance (~381 mm), not where the peg ended
        # up. Track the closest approach instead, and only sample on steps that
        # did not reset.
        min_dp = float("inf")
        dp_now, dq_now = 0.0, 0.0        # defined even if step 0 ends the episode
        for t in range(raw.max_episode_length):
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, rew, done, _ = env.step(actions)

            if not bool(done[0]):
                dp_now, dq_now = mdp.peg_pose_in_hole(raw)
                dp_now, dq_now = float(dp_now[0]), float(dq_now[0])
                min_dp = min(min_dp, dp_now)

            ins = mdp.success_term(raw)
            if bool(ins[0]) and seated_at is None:
                seated_at = t
                print(f"    {t:5d} {dp_now*1000:9.3f} {np.degrees(dq_now):9.2f}   INSERTED (first time)")
            if t % args_cli.report_every == 0:
                print(f"    {t:5d} {dp_now*1000:9.3f} {np.degrees(dq_now):9.2f}   "
                      f"{'seated' if bool(ins[0]) else 'working'}")
            if done.any():
                break

        verdict = "SUCCESS" if seated_at is not None else "FAILED"
        print(f"  episode {loop + 1}: {verdict}, first seated at step {seated_at}, "
              f"closest approach dp = {min_dp*1000:.3f} mm\n")

    if args_cli.hold_s > 0:
        print(f"  holding {args_cli.hold_s:.0f} s for inspection ...")
        import time

        time.sleep(args_cli.hold_s)
    safe_close(env, label="env")


if __name__ == "__main__":
    import threading

    main()
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=20)
    os._exit(0)

"""How much does the trained policy actually react to what it sees?

The environment has no domain randomization, so the hole is always in the same
place and the robot always starts from the same pose. That makes two very
different things look identical:

  * a closed-loop policy that servos on the image and corrects errors, and
  * a memorised fixed trajectory that happens to hit the one hole it was trained on.

The way to tell them apart is to move the hole and see what happens. This
perturbs the fixture in x and y (leaving z alone, so the plate still sits flush
on the board, and keeping the offset small enough that the 200 mm plate stays on
the 600 mm board) and re-evaluates the same frozen checkpoint at each offset.

A policy that still succeeds at +-10 mm is genuinely using the image. One that
drops to 0 the moment the hole moves had encoded the scene, not the task.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/probe_robustness.py --headless \
        --checkpoint runs/rsl_rl/pegih_vision/2026-10-08_22-56-56_vision/model_550.pt
    $PY scripts/probe_robustness.py --headless --no-vision \
        --checkpoint runs/rsl_rl/pegih_vision/2026-10-09_16-00-10_state_long_resume/model_3875.pt
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Perturb the hole and see if the policy copes.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--offsets_mm", type=str, default="0,2,5,10,20",
                    help="comma-separated offsets applied in both +x and +y")
parser.add_argument("--episodes", type=int, default=1, help="episodes per offset")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--no-vision", action="store_true")
parser.add_argument("--symmetric-critic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.no_vision:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(not args_cli.no_vision, privileged_critic=not args_cli.symmetric_critic)

    install_cnn_policy()

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = PegIhPPORunnerCfg()
    agent_cfg.device = args_cli.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint(runner, args_cli.checkpoint)
    policy = runner.alg.policy

    raw = env.unwrapped
    n = raw.num_envs
    device = raw.device
    base = tuple(env_cfg.fixture_pos)
    print(f"\n>>> checkpoint : {os.path.basename(args_cli.checkpoint)}")
    print(f">>> fixture    : {base}  (nominal)")
    print(f">>> num_envs   : {n}   episodes/offset: {args_cli.episodes}\n")
    print(f"    {'offset':>8s} {'success %':>10s} {'insert/step':>12s} {'return':>9s}   note")

    offsets = [float(v) for v in args_cli.offsets_mm.split(",")]
    for off_mm in offsets:
        d = off_mm / 1000.0
        # Move the hole. reset_hole_fixed() reads env.cfg.fixture_pos on every
        # reset, so changing it here takes effect at the next env.reset() -- no
        # need to rebuild the env (and a second gym.make would fail anyway:
        # "Simulation context already exists").
        raw.cfg.fixture_pos = (base[0] + d, base[1] + d, base[2])

        ever = torch.zeros(n, dtype=torch.bool, device=device)
        ep_return = torch.zeros(n, device=device)
        succ, rets, ins_sum, steps = [], [], 0.0, 0
        obs, _ = env.reset()
        policy.reset(torch.ones(n, dtype=torch.bool, device=device))

        for _ in range(args_cli.episodes * raw.max_episode_length + 1):
            with torch.no_grad():
                a = policy.act_inference(obs)
            obs, rew, done, _ = env.step(a)
            ep_return += rew
            ins = mdp.success_term(raw)
            ever |= ins
            ins_sum += float(ins.float().sum())
            steps += n
            if done.any():
                idx = torch.nonzero(done).flatten()
                for i in idx.tolist():
                    succ.append(bool(ever[i]))
                    rets.append(float(ep_return[i]))
                ever[idx] = False
                ep_return[idx] = 0.0

        rate = 100.0 * sum(succ) / max(len(succ), 1)
        note = "nominal (what was trained)" if off_mm == 0 else ""
        print(f"    {off_mm:6.0f}mm {rate:9.1f}% {ins_sum/max(steps,1):12.4f} "
              f"{sum(rets)/max(len(rets),1):9.2f}   {note}")

    raw.cfg.fixture_pos = base
    print("\n  A policy that keeps succeeding as the hole moves is using the image.")
    print("  One that collapses to 0 once the hole moves had memorised the scene.\n")
    env.close()


if __name__ == "__main__":
    import threading

    main()
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=20)
    os._exit(0)

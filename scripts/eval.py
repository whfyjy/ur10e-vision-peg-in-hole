"""Evaluate a trained checkpoint on the UR10e vision peg-in-hole task.

Reports three numbers per action source:

  success %   -- fraction of episodes that satisfied the task criterion
                 (dq < 0.05 rad AND dp < 0.003 m) at least once.
  insert/step -- mean fraction of ENVIRONMENTS satisfying the criterion at each
                 step. Not the same quantity as success %: that one saturates
                 (once an episode has inserted it counts for the whole episode),
                 so insert/step is what distinguishes "seated at step 100" from
                 "scratched in at step 850", and it is the more sensitive signal
                 when comparing checkpoints.
  return      -- mean episode reward.

Both action sources are reported by default because they differ on this task:
PPO trains on sampled actions, but a deployment takes the policy mean. A
checkpoint can look fine sampled and be degenerate deterministically -- genpih
measured a checkpoint whose sampled return was 4.43/step while its deterministic
return was 2.31/step with insert/step = 0.00.

--which last | best. LAST IS OFTEN NOT THE ONE YOU WANT: this task family shows
a reproducible rise-then-collapse, so the final checkpoint can be deep inside a
collapse. `best` scans checkpoints and picks by insert/step. genpih's own runs
needed this (its yaw run logged 9.44/step at the end while its model_2999 was
dead). This is a real effect here too, not a precaution: the vision run in this
repo peaked at parse 6.94 (iteration ~385) and was at 4.09 by iteration 487.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/eval.py --headless --run_dir runs/rsl_rl/pegih_vision/<run> --which best
    $PY scripts/eval.py --headless --checkpoint <path/to/model_N.pt>
    $PY scripts/eval.py --headless --no-vision --checkpoint <state-policy ckpt>
"""

import argparse
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate a peg-in-hole checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="path to one model_*.pt")
parser.add_argument("--run_dir", type=str, default=None, help="run directory to take checkpoints from")
parser.add_argument("--which", type=str, default="last", choices=["last", "best"])
parser.add_argument("--scan_stride", type=int, default=0,
                    help="for --which best: consider every N-th checkpoint (0 = aim for ~12 candidates)")
parser.add_argument("--scan_episodes", type=int, default=1, help="episodes per candidate during the scan")
parser.add_argument("--episodes", type=int, default=2, help="episodes per environment for the final numbers")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--action", type=str, default="both", choices=["sampled", "deterministic", "both"])
parser.add_argument("--no-vision", action="store_true")
parser.add_argument("--symmetric-critic", action="store_true")
parser.add_argument("--seed", type=int, default=42)
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


def iteration_of(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def candidate_checkpoints():
    if args_cli.checkpoint:
        return [args_cli.checkpoint]
    if not args_cli.run_dir:
        raise SystemExit("give --checkpoint or --run_dir")
    ckpts = sorted(glob.glob(os.path.join(args_cli.run_dir, "model_*.pt")), key=iteration_of)
    if not ckpts:
        raise SystemExit(f"no model_*.pt in {args_cli.run_dir}")
    if args_cli.which == "last":
        return [ckpts[-1]]
    stride = args_cli.scan_stride or max(1, len(ckpts) // 12)
    picked = ckpts[::stride]
    if ckpts[-1] not in picked:
        picked.append(ckpts[-1])
    return picked


def evaluate(env, policy, deterministic, episodes):
    """Run `episodes` full episodes per environment; return a dict of metrics.

    `policy` is the raw module (runner.alg.policy), NOT the closure returned by
    runner.get_inference_policy() -- that one is a bare function with no
    .reset(), and calling .reset on it fails with
    "'function' object has no attribute 'reset'".
    """
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    horizon = env.unwrapped.max_episode_length

    ep_ever = torch.zeros(n, dtype=torch.bool, device=device)
    ep_return = torch.zeros(n, device=device)
    successes, returns = [], []
    insert_sum, step_count = 0.0, 0

    obs, _ = env.reset()
    policy.reset(torch.ones(n, dtype=torch.bool, device=device))

    # every environment starts at step 0 after reset, so they all time out
    # together: this many steps yields exactly `episodes` completed episodes each
    for _ in range(episodes * horizon + 1):
        with torch.no_grad():
            actions = policy.act_inference(obs) if deterministic else policy.act(obs)
        # RslRlVecEnvWrapper.step returns 4 values, not gymnasium's 5: the
        # wrapper has already OR'd terminated|truncated into `dones`.
        obs, rew, done, _ = env.step(actions)
        ep_return += rew

        ins = mdp.success_term(env.unwrapped)
        ep_ever |= ins
        insert_sum += float(ins.float().sum())
        step_count += n

        if done.any():
            idx = torch.nonzero(done).flatten()
            for i in idx.tolist():
                successes.append(bool(ep_ever[i]))
                returns.append(float(ep_return[i]))
            ep_ever[idx] = False
            ep_return[idx] = 0.0

    return {
        "n": len(successes),
        "success": 100.0 * sum(successes) / max(len(successes), 1),
        "insert_step": insert_sum / max(step_count, 1),
        "return": sum(returns) / max(len(returns), 1),
    }


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.enable_vision(not args_cli.no_vision, privileged_critic=not args_cli.symmetric_critic)

    install_cnn_policy()

    ckpts = candidate_checkpoints()
    print(f"\n>>> observation : {'state' if args_cli.no_vision else 'image'}"
          f"{' + privileged critic' if not args_cli.no_vision and not args_cli.symmetric_critic else ''}")
    print(f">>> num_envs    : {args_cli.num_envs}   episodes/env: {args_cli.episodes}")
    print(f">>> candidates  : {len(ckpts)}")

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = PegIhPPORunnerCfg()
    agent_cfg.device = args_cli.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    chosen = ckpts[-1]
    if len(ckpts) == 1:
        print(f">>> checkpoint  : {chosen}\n")
    else:
        print(f"\n>>> scan (checkpoint scan; genpih showed the last one is often collapsed)")
        print(f"    {'checkpoint':>24s} {'insert/step':>12s} {'return':>10s}")
        best_score = -1.0
        for c in ckpts:
            load_checkpoint(runner, c)
            m = evaluate(env, runner.alg.policy, deterministic=True, episodes=args_cli.scan_episodes)
            print(f"    {os.path.basename(c):>24s} {m['insert_step']:12.4f} {m['return']:10.2f}")
            if m["insert_step"] > best_score:
                best_score, chosen = m["insert_step"], c
        print(f">>> selected    : {chosen}  (insert/step = {best_score:.4f})\n")

    load_checkpoint(runner, chosen)
    sources = ["deterministic", "sampled"] if args_cli.action == "both" else [args_cli.action]
    print(f">>> final evaluation of {os.path.basename(chosen)}")
    for src in sources:
        m = evaluate(env, runner.alg.policy, deterministic=(src == "deterministic"), episodes=args_cli.episodes)
        print(f"    [{src:13s}] success = {m['success']:6.1f} %  ({int(m['success']*m['n']/100)}/{m['n']} episodes)   "
              f"insert/step = {m['insert_step']:.4f}   return = {m['return']:8.2f}")

    print()
    safe_close(env, label="env")


if __name__ == "__main__":
    # simulation_app.close() is known to hang after a run like this one (it hung
    # for over two hours here, long after the script had already raised, which
    # made a 76-second crash look like a 2.4-hour stall). Close on a daemon
    # thread with a deadline and hard-exit.
    import threading
    import traceback

    _code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        _code = 1
    _t = threading.Thread(target=simulation_app.close, daemon=True)
    _t.start()
    _t.join(timeout=15)
    os._exit(_code)

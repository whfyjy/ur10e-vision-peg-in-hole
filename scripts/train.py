"""Train PPO on the UR10e vision peg-in-hole task.

    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/train.py --headless --num_envs 1024
    $PY scripts/train.py --headless --num_envs 256 --max_iterations 20 --no-vision   # fast sanity run

--no-vision swaps the policy observation to the 13-dim state vector and drops
the camera entirely. That path exists so a geometry/physics problem can be told
apart from a vision-learning problem: if the state policy learns and the vision
policy does not, the scene is fine and the camera or the encoder is the issue.
"""

import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train PPO on UR10e vision peg-in-hole.")
parser.add_argument("--num_envs", type=int, default=None, help="default: the cfg's 1024")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-vision", action="store_true",
                    help="use the 13-dim state observation instead of the camera (camera disabled)")
parser.add_argument("--symmetric-critic", action="store_true",
                    help="drop the privileged critic group so the critic also consumes the image")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--no-guard", action="store_true",
                    help="disable the critic-divergence guard (it saves and stops early by default)")
parser.add_argument("--guard-factor", type=float, default=1e3,
                    help="trip if the value-function loss grows by this factor in one iteration")
parser.add_argument("--guard-patience", type=int, default=2,
                    help="consecutive such jumps required to trip")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--load_checkpoint", type=str, default="model_.*.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# a camera task cannot run without this; the env cfg skips the sensor instead if
# vision is off, so the flag is harmless there
if not args_cli.no_vision:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import pegih_tasks.tasks.peg_hole  # noqa: F401, E402  (registers the gym id)
from pegih_tasks.tasks.peg_hole.cnn_policy import install as install_cnn_policy, load_checkpoint  # noqa: E402
from pegih_tasks.tasks.peg_hole.peg_hole_env_cfg import PegHoleEnvCfg  # noqa: E402
from pegih_tasks.config.rsl_rl_ppo_cfg import PegIhPPORunnerCfg  # noqa: E402
from pegih_tasks.train_guard import CriticDiverged, GuardedRunner  # noqa: E402


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg = PegHoleEnvCfg()
    env_cfg.enable_vision(
        enabled=not args_cli.no_vision,
        privileged_critic=not args_cli.symmetric_critic,
    )
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    agent_cfg = PegIhPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    mode = "state" if args_cli.no_vision else ("vision-sym" if args_cli.symmetric_critic else "vision")
    agent_cfg.run_name = args_cli.run_name or mode
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.resume:
        agent_cfg.resume = True
        agent_cfg.load_run = args_cli.load_run
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    log_root = os.path.join(PROJECT_ROOT, "runs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}")

    print(f"\n>>> task        : UR10e-VisionPegInHole-v0")
    print(f">>> observation : {'camera image only' if not args_cli.no_vision else 'state (13)'}"
          f"{' + privileged critic (14)' if not args_cli.no_vision and not args_cli.symmetric_critic else ''}")
    print(f">>> num_envs    : {env_cfg.scene.num_envs}")
    print(f">>> log_dir     : {log_dir}\n")

    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Must happen before the runner resolves class_name="ActorCriticCNN".
    # Installed unconditionally: with --no-vision the observation is a 13-dim
    # state vector, ActorCriticCNN sees an input that is not the image and falls
    # back to a plain MLP, so the same class_name works on both paths.
    install_cnn_policy()

    # GuardedRunner saves a checkpoint and stops the moment the critic starts to
    # diverge, instead of letting it run until float32 overflows and takes the
    # policy down with it. Pass --no-guard for the stock behaviour.
    runner_cls = OnPolicyRunner if args_cli.no_guard else GuardedRunner
    guard_kwargs = {} if args_cli.no_guard else dict(
        guard_factor=args_cli.guard_factor, guard_patience=args_cli.guard_patience
    )
    if not args_cli.no_guard:
        print(f">>> guard       : stop if Loss/value_function grows x{args_cli.guard_factor:g} "
              f"for {args_cli.guard_patience} consecutive iterations")
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir,
                        device=agent_cfg.device, **guard_kwargs)
    if agent_cfg.resume:
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f">>> resuming from {resume_path}")
        load_checkpoint(runner, resume_path)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except CriticDiverged as e:
        # The guard already saved a checkpoint and printed rollback instructions.
        # Exiting 0: this is a handled stop, not a crash.
        print(e)
        safe_close(env, label="env")
        return
    safe_close(env, label="env")


if __name__ == "__main__":
    main()
    simulation_app.close()

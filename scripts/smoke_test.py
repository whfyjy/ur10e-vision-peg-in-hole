"""Smoke test for the UR10e vision peg-in-hole environment.

Checks, in order, each printing PASS/FAIL:

  1. env builds; observation and action shapes match the configured mode
  2. HOME pose: peg tip where the URDF says it is, peg axis vertical, and the
     peg is NOT aligned with the hole (the task must require a correction)
  3. reward sanity: dense in (0, 1], sparse only ever 0 / 2.6 / 10
  4. reset determinism: this task has no domain randomization, so two resets
     must produce bit-identical state -- if they do not, something is still
     random and every "fixed" number in the config is a lie
  5. scripted insertion reaches dp < 3 mm (the success criterion is REACHABLE)
  6. camera: image is not constant, and changes between HOME and hover
  7. no NaN in observations or rewards over a run of random actions

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/smoke_test.py --headless
    $PY scripts/smoke_test.py --headless --no-vision
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pegih_tasks.utils import safe_close  # noqa: E402
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Smoke test for the vision peg-in-hole env.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--no-vision", action="store_true")
parser.add_argument("--image_dir", type=str, default=None, help="write camera PNGs here")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.no_vision:
    args_cli.enable_cameras = True

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

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def main():
    env_cfg = PegHoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.sim.device = "cuda:0"
    env_cfg.enable_vision(not args_cli.no_vision)

    print(f"\n=== UR10e vision peg-in-hole smoke test  (vision={not args_cli.no_vision}) ===\n")
    print("[1] environment construction")
    env = gym.make("UR10e-VisionPegInHole-v0", cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    pol = obs["policy"]
    if args_cli.no_vision:
        check("policy obs is the 13-dim state", tuple(pol.shape) == (args_cli.num_envs, 13), str(tuple(pol.shape)))
    else:
        check("policy obs is the flattened image", tuple(pol.shape) == (args_cli.num_envs, 100 * 100 * 3),
              str(tuple(pol.shape)))
        check("policy pixels in [0, 1]", float(pol.min()) >= 0.0 and float(pol.max()) <= 1.0,
              f"min={float(pol.min()):.3f} max={float(pol.max()):.3f}")
    check("critic group present", "critic" in env.observation_manager.group_obs_dim,
          str(dict(env.observation_manager.group_obs_dim)))
    check("action dim is 6", env.action_manager.total_action_dim == 6)
    check("episode is 900 steps (15 s at 60 Hz)", env.max_episode_length == 900, str(env.max_episode_length))

    robot = env.scene["robot"]
    dt = env.sim.get_physics_dt()

    print("\n[2] home pose and starting misalignment")
    tip = robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]]
    # body_pos_w is WORLD; the URDF chain is in the robot base frame, and Isaac
    # Lab lays the envs out on a grid around the world origin (env 0 sits at
    # +spacing/2, -spacing/2, not at the origin), so subtract the env origin
    tip_np = tip.cpu().numpy() - env.scene.env_origins[0].cpu().numpy()
    check("peg tip matches the URDF-derived pose", np.allclose(tip_np, kin.tip_pos(kin.HOME_JOINTS), atol=3e-3),
          f"sim {np.round(tip_np, 4)} vs URDF {np.round(kin.tip_pos(kin.HOME_JOINTS), 4)}")

    q = robot.data.joint_pos[0].cpu().numpy()
    ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    check("peg axis is vertical at home", kin.tilt_deg(q[ids]) < 0.5,
          f"tilt {kin.tilt_deg(q[ids]):.3f} deg")

    dp, dq = mdp.peg_pose_in_hole(env)
    d0, q0 = float(dp[0]), float(dq[0])
    check("peg does NOT start aligned with the hole", d0 > 0.02,
          f"dp = {d0*1000:.1f} mm (must be well above the 3 mm success tolerance)")
    check("peg does start near-vertical w.r.t. the hole", np.degrees(q0) < 5.0, f"dq = {np.degrees(q0):.2f} deg")

    print("\n[3] reward sanity")
    dense = mdp.dense_reward(env)
    parse = mdp.parse_reward(env)
    check("dense reward in (0, 1]", bool((dense > 0).all() and (dense <= 1 + 1e-6).all()),
          f"range [{float(dense.min()):.4f}, {float(dense.max()):.4f}]")
    check("sparse reward only 0 / 2.6 / 10",
          bool(torch.isin(parse, torch.tensor([0.0, 2.6, 10.0], device=parse.device)).all()),
          f"values {sorted(set(parse.tolist()))}")

    print("\n[4] reset determinism (no domain randomization)")
    def snapshot():
        return (robot.data.joint_pos.clone(), robot.data.body_pos_w.clone(),
                env.scene["hole"].data.root_pos_w.clone(), env.scene["hole"].data.root_quat_w.clone())
    env.reset()
    a = snapshot()
    env.reset()
    b = snapshot()
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    check("two resets give identical state", same,
          "" if same else "TASK IS STILL RANDOMIZING SOMEWHERE -- the config claims it is fixed")

    print("\n[5] scripted insertion reaches the success criterion")
    joint_ids, _ = robot.find_joints(list(HOME_JOINT_ORDER), preserve_order=True)
    home = np.array(kin.HOME_JOINTS)

    def command(q_urdf):
        tgt = robot.data.joint_pos.clone()
        tgt[:, joint_ids] = torch.tensor(q_urdf, device=env.device, dtype=tgt.dtype)
        robot.set_joint_position_target(tgt)

    def spin(n):
        for _ in range(n):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt)

    # The scripted descent below is ~55 waypoints x 30 physics steps. With the
    # camera live that is a tiled render every other step, which dominates the
    # runtime and tests nothing about insertion -- so suspend sensor updates for
    # its duration and restore them for the camera check.
    cam_cfg = None
    if not args_cli.no_vision:
        cam_cfg = env.scene["tiled_camera"].cfg
        cam_cfg._saved_update_period = cam_cfg.update_period
        cam_cfg.update_period = 1.0e9

    fx, fy = FIXTURE_POS[0], FIXTURE_POS[1]
    qq = home.copy()
    best = 1e9
    for z in np.arange(PLATE_TOP_Z + 0.02, BOARD_TOP_Z - 0.003, -0.001):
        qq, err, _ = kin.ik_tip((fx, fy, float(z)), q0=qq)
        command(qq)
        spin(30)
        best = min(best, float(mdp.peg_pose_in_hole(env)[0][0]))
    check("dp < 3 mm is reachable by a scripted descent", best < 0.003, f"best dp = {best*1000:.3f} mm")
    tip_z = float(robot.data.body_pos_w[0, robot.find_bodies("peg_link")[0][0]][2])
    check("peg tip is stopped by the board, not passing through it", abs(tip_z - BOARD_TOP_Z) < 0.004,
          f"tip_z = {tip_z:.4f}, board top = {BOARD_TOP_Z:.4f}")

    print("\n[6] camera")
    if args_cli.no_vision:
        check("camera disabled with --no-vision", "tiled_camera" not in env.scene.sensors)
    else:
        cam_cfg.update_period = cam_cfg._saved_update_period
        cam = env.scene["tiled_camera"]
        img_seated = cam.data.output["rgb"].float()
        env.reset()
        spin(60)
        img_home = cam.data.output["rgb"].float()
        check("image is not constant", float(img_home.std()) > 1.0, f"std = {float(img_home.std()):.2f}")
        mse = float(((img_home - img_seated) ** 2).mean())
        check("image changes between seated and home", mse > 5.0, f"MSE = {mse:.1f}")
        if args_cli.image_dir:
            from PIL import Image
            os.makedirs(args_cli.image_dir, exist_ok=True)
            for tag, im in (("home", img_home), ("seated", img_seated)):
                p = os.path.join(args_cli.image_dir, f"smoke_{tag}.png")
                Image.fromarray(im[0].cpu().numpy().astype(np.uint8)).resize((400, 400), Image.NEAREST).save(p)
                print(f"       wrote {p}")

    print(f"\n[7] {args_cli.steps} steps of random actions")
    env.reset()
    rew_sum = 0.0
    bad = False
    for _ in range(args_cli.steps):
        act = torch.clamp(torch.randn(args_cli.num_envs, 6, device=env.device) * 0.2, -1.0, 1.0)
        o, r, term, trunc, _ = env.step(act)
        rew_sum += float(r.mean())
        if not torch.isfinite(r).all() or not torch.isfinite(o["policy"]).all():
            bad = True
            break
    check("no NaN/Inf in observations or rewards", not bad)
    check("rewards are non-zero under random actions", rew_sum != 0.0, f"sum of means = {rew_sum:.3f}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  peak VRAM: {vram:.2f} GB at {args_cli.num_envs} envs")

    safe_close(env, label="env")

    n_pass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n=== {n_pass}/{len(RESULTS)} checks passed ===")
    for name, ok in RESULTS:
        if not ok:
            print(f"    FAILED: {name}")
    print("SMOKE OK" if n_pass == len(RESULTS) else "SMOKE FAIL")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)

"""PPO runner configuration for the UR10e vision peg-in-hole task.

The algorithm block is genpih's, unchanged, including the fixed learning-rate
schedule -- see the comment on `schedule` below for why rsl_rl's adaptive
schedule is unusable here.

The policy block is the one thing that differs: rsl_rl 2.3.3 ships no CNN, so
`class_name` names a class that does not exist in the library.
`pegih_tasks.tasks.peg_hole.cnn_policy.install()` binds it into the namespace
`OnPolicyRunner` resolves the name from; scripts must call that before building
the runner, or the runner dies with `NameError: ActorCriticCNN`.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PegIhPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Vision rollouts are far more expensive per step than genpih's blind ones
    # (a render plus a conv forward and backward over 100x100x3 per environment),
    # so the horizon is longer and the batch is smaller than genpih's 24 x 8192.
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 25
    experiment_name = "pegih_vision"
    run_name = ""
    resume = False
    # images arrive already scaled to [0,1] by mdp.camera_image; a running
    # normalizer on top of that would be fitting a moving distribution to pixels
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        # resolved through eval() inside rsl_rl.runners.on_policy_runner --
        # see cnn_policy.install(). The hidden dims below are consumed by
        # ActorCriticCNN as the MLP head after the conv stem:
        #   Conv(3->32,k5,s2) -> Conv(32->64,k3,s2) -> Conv(64->64,k3,s2)
        #   -> Conv(64->64,k3,s2) -> flatten(3136) -> Linear(256) -> Linear(128)
        class_name="ActorCriticCNN",
        init_noise_std=1.0,
        # rsl_rl's default, and what the one run that actually learned this task
        # used. It was briefly switched to "log" and then switched back.
        #
        # The switch was made to dodge
        #     Assertion `normal expects all elements of std >= 0.0` failed
        # but that was a misdiagnosis: the assertion is a *symptom* of the critic
        # diverging (value loss reaching ~1e25 and overflowing float32), which
        # drives the std parameter negative. "log" hides the symptom by
        # construction -- std = exp(log_std) cannot go negative -- while removing
        # the signal. The real defence is pegih_tasks/train_guard.py; keep it
        # enabled (do not pass --no-guard).
        #
        # Measured after switching back: "scalar" does NOT rescue a failing run.
        # Two otherwise identical 512-env vision runs differing only in this knob
        #
        #             noise_std peak   entropy peak   parse peak
        #   "scalar"      1.15             8.97         2.36   never inserted
        #   "log"         2.03            11.98         2.38   never inserted
        #
        # so "log" does make the noise runaway worse, but it is not what decides
        # the outcome -- both plateau at the same parse. Do not read the smaller
        # "scalar" number as a fix.
        #
        # What does decide it is run-to-run stochasticity. Three vision runs of
        # identical code, identical hyperparameters and identical seed 42 already
        # disagree at iteration 0 (Episode_Reward/dense 0.006018 / 0.005987 /
        # 0.005994): Isaac's scene and renderer setup consumes RNG, so the seed
        # does not pin the run. One of the three learned to insert and the other
        # two never left the "aligned but not approaching" plateau at parse ~2.0.
        # The learned one was NOT a stable success either -- see the note on
        # learning_rate below.
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        # This rate reaches a good policy but does not hold one, and that is the
        # most concrete defect measured in this project. In the one vision run
        # that learned (Oct-8, 512 envs), a checkpoint scan with deterministic
        # actions, 1 episode per env, gives:
        #
        #   iteration  200   250   300   350   400   450   500   550   600+
        #   insert/st  .42   .60   .66   .63   .02   .72   .84   .84   0.00
        #
        # Two separate failures, not one. model_400 is a transient collapse
        # (.63 -> .02 -> .72) the run recovers from. The one at 600 is terminal:
        # Loss/value_function goes 4.3 (it 550) -> 91 (it 600) -> 1.3e7 (it 650)
        # and insert/step is 0.0000 for every checkpoint from 600 on, because the
        # critic diverged and took the actor with it. Both survivors are kept in
        # best_policy/: model_550.pt is the one to use (100% of 96 episodes with
        # deterministic actions, 96.9% sampled), model_500.pt is the runner-up
        # (100% of 64, 90.6% sampled).
        #
        # So: lower this first (3e-4) when retraining, and keep the guard in
        # pegih_tasks/train_guard.py enabled so a blow-up stops the run instead
        # of destroying the newest checkpoint.
        learning_rate=1.0e-3,
        # "fixed", NOT rsl_rl's default "adaptive". The adaptive KL schedule runs
        # inside the mini-batch loop and multiplies or divides the learning rate
        # by 1.5 per mini-batch -- 5 epochs x 4 minibatches = 20 adjustments, i.e.
        # up to 1.5**20 ~ 3300x in one iteration, which slams the rate into a
        # bound and pins it there. genpih measured 94% of one run sitting at
        # 1e-5, which is a no-op. desired_kl is kept but is inert under "fixed".
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

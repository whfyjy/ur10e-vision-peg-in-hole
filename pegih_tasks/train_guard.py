"""Stop training when the critic starts to diverge, instead of when it explodes.

The failure mode, measured twice in this project (README "已知问题"): the mean
value function loss grows by many orders of magnitude per iteration until
float32 overflows, and the run dies with

    Assertion `normal expects all elements of std >= 0.0` failed
    torch.AcceleratorError: CUDA error: device-side assert triggered

Observed series:

    state run   2.7e6 -> 2.25e12 -> 1.66e18 -> 1.99e24 -> 4.6e30   (dead)
    resumed       410 ->   1.4e14 -> 2.0e25                        (dead)

Meanwhile reward, sparse reward, surrogate loss, entropy, action noise and
episode length all stay flat -- the policy is fine, the critic is not.

WHY STOP EARLY RATHER THAN LET IT CRASH. By the time the loss is visibly
exploding the policy has already been degraded:

    model_5075   insert/step 0.9436
    model_5150   insert/step 0.4521     <- 75 iterations later, already spoiled
    (next iteration: loss = 1e25, crash)

Waiting for the crash means the newest checkpoint you own is the spoiled one.
Stopping on the first sign and rolling back keeps the clean one.

HOW TO USE. Wrap the runner and catch the exception:

    runner = GuardedRunner(env, cfg.to_dict(), log_dir=..., device=...)
    try:
        runner.learn(...)
    except CriticDiverged as e:
        print(e)          # includes the checkpoint to resume from

The guard writes `model_DIVERGED_<it>.pt` before raising, because it runs inside
`log()`, which rsl_rl calls BEFORE its own periodic save -- so without that save
the iteration would be lost.
"""

from __future__ import annotations

import math
import os

from rsl_rl.runners import OnPolicyRunner


class CriticDiverged(RuntimeError):
    """Raised by GuardedRunner when the critic loss runs away."""


class GuardedRunner(OnPolicyRunner):
    """OnPolicyRunner that saves and stops when the value-function loss explodes.

    Args:
        guard_factor: trip if the loss exceeds the previous iteration's by this
            factor. 1e3 is far above normal iteration-to-iteration noise.
        guard_patience: consecutive trips required. 1 would already be
            conclusive; 2 costs one iteration and removes any doubt.
        guard_floor: ignore jumps while the loss is below this. Early in
            training the loss is tiny and relative swings mean nothing.
        guard_warmup: do not check at all before this iteration.
    """

    def __init__(self, *args, guard_factor: float = 1e3, guard_patience: int = 2,
                 guard_floor: float = 1e3, guard_warmup: int = 50, guard_window: int = 5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_factor = guard_factor
        self.guard_patience = guard_patience
        self.guard_floor = guard_floor
        self.guard_warmup = guard_warmup
        self.guard_window = guard_window
        self._history: list[float] = []
        self._strikes = 0

    # -- the check ---------------------------------------------------------
    def _trips(self, it: int, vf: float | None) -> tuple[bool, str]:
        """Compare the loss against the MEDIAN of the recent past, not the last value.

        Comparing with the previous iteration only misses a real pattern: once
        the loss has jumped to 1e7, the next iteration's 3e7 is only 3x and does
        not look like a jump any more -- that is exactly the shape the vision run
        died with. A rolling median keeps the baseline where it should be, and it
        also ignores a single spike that falls straight back, which comparing
        "2 consecutive jumps" would have flagged.
        """
        if vf is None:
            return False, "value_function not in loss_dict"
        if it < self.guard_warmup:
            self._history.append(vf)
            return False, "warmup"
        if not math.isfinite(vf):
            return True, f"loss is not finite ({vf})"

        tripped = False
        why = "ok"
        if len(self._history) >= 1:
            baseline = sorted(self._history)[len(self._history) // 2]
            if baseline > 0:  # a zero/negative baseline makes the ratio meaningless
                limit = max(baseline * self.guard_factor, self.guard_floor)
                if vf > limit:
                    self._strikes += 1
                    if self._strikes >= self.guard_patience:
                        tripped = True
                        why = (f"loss ran away: baseline {baseline:.4g} -> {vf:.4g} "
                               f"(x{vf / baseline:.3g}) for {self._strikes} consecutive iterations")
                else:
                    self._strikes = 0
        self._history.append(vf)
        if len(self._history) > self.guard_window:
            self._history.pop(0)
        return tripped, why

    def log(self, locs: dict, **kwargs):
        # rsl_rl logs and only then saves on `it % save_interval == 0`, so raising
        # from here would skip that save -- do it ourselves before raising.
        loss_dict = locs.get("loss_dict", {})
        vf = loss_dict.get("value_function")
        vf = float(vf) if vf is not None else None
        it = int(locs.get("it", self.current_learning_iteration))

        trip, why = self._trips(it, vf)
        super().log(locs, **kwargs)          # keep the tensorboard record intact

        if trip:
            name = f"model_DIVERGED_{it}.pt"
            path = os.path.join(self.log_dir, name)
            self.save(path)
            raise CriticDiverged(
                f"\n\n  critic diverged at iteration {it}: {why}\n"
                f"  saved   : {path}\n"
                f"  NOTE this checkpoint is already degraded; roll back further.\n\n"
                f"  next steps:\n"
                f"    1) scan the checkpoints before this one and pick the best:\n"
                f"       $PY scripts/eval.py --headless --which best \\\n"
                f"           --run_dir {self.log_dir} --num_envs 32 --episodes 2\n"
                f"    2) resume from it:\n"
                f"       $PY scripts/train.py --headless --resume --load_run "
                f"{os.path.basename(self.log_dir)} --load_checkpoint model_XXXX.pt ...\n"
            )

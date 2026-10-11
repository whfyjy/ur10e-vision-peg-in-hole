#!/usr/bin/env python
"""Plot the runs that matter, with a resumed run stitched onto its parent.

`plot_results.py` draws each run directory as its own column, which is right for
a survey but wrong for the state-policy story: that run is one continuous
training history split across two directories because a critic divergence killed
it at iteration 3958 and it was resumed from a checkpoint. Drawn separately it
looks like two short runs with a gap.

This stitches them by iteration number (later directory wins on overlap) and
draws the three curves worth looking at:

    vision     2026-10-08_22-56-56_vision                     (0 -> 799)
    state      2026-10-08_23-54-56_state_long             (0 -> 3950)
            +  2026-10-09_16-00-10_state_long_resume   (3875 -> 5150)

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/plot_curves.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(PROJECT_ROOT, "runs", "rsl_rl", "pegih_vision")

SERIES = [
    ("vision (512 envs)", "#1f6fb4", ["2026-10-08_22-56-56_vision"]),
    ("state (1024 envs)", "#c0392b", ["2026-10-08_23-54-56_state_long",
                                      "2026-10-09_16-00-10_state_long_resume"]),
]

PANELS = [
    ("Episode_Reward/parse", "sparse reward\n2.6 = aligned only, 10 = inserted", False, True),
    ("Loss/value_function", "value function loss\n(last one to blow up kills the run)", True, False),
    ("Train/mean_reward", "mean episode reward", False, False),
]


def read_tag(run_dirs, tag):
    """Merge the tag across run dirs, keyed by iteration (later dir wins)."""
    merged = {}
    for name in run_dirs:
        d = os.path.join(LOG_ROOT, name)
        files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
        if not files:
            continue
        ea = EventAccumulator(max(files, key=os.path.getsize))
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for s in ea.Scalars(tag):
                merged[s.step] = s.value
    if not merged:
        return None
    steps = sorted(merged)
    return steps, [merged[s] for s in steps]


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.2))
    for ax, (tag, title, logy, refs) in zip(axes, PANELS):
        for label, colour, dirs in SERIES:
            got = read_tag(dirs, tag)
            if got:
                ax.plot(got[0], got[1], lw=1.1, color=colour, label=label)
        if logy:
            ax.set_yscale("log")
        if refs:
            ax.axhline(2.6, color="#7f8c8d", ls=":", lw=1.1)
            ax.axhline(10.0, color="#27ae60", ls=":", lw=1.1)
            ax.text(ax.get_xlim()[1], 2.6, " 2.6 = aligned only", va="center", fontsize=7, color="#7f8c8d")
            ax.text(ax.get_xlim()[1], 10.0, " 10 = inserted", va="center", fontsize=7, color="#27ae60")
            ax.set_ylim(-0.4, 11.2)
        ax.set_xlabel("training iteration", fontsize=9)
        ax.set_ylabel(title.split("\n")[0], fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, loc="best")

    # the resumed run is a separate directory but one continuous history
    axes[0].axvspan(3950, 3975, color="k", alpha=0.07)
    axes[0].annotate("critic divergence\n-> resumed from ckpt 3875",
                     xy=(3960, 5.0), xytext=(2200, 3.0), fontsize=7,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))

    fig.suptitle("UR10e vision peg-in-hole — the two runs that matter "
                 "(state history stitched across the crash at iteration 3958)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(PROJECT_ROOT, "results", "training_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

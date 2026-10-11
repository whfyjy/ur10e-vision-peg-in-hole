"""Plot the training curves from the rsl_rl tensorboard event files.

rsl_rl writes scalars to <log_dir>/events.out.tfevents.*; this reads them back
and draws the four that matter, one column per run. Everything is read from the
event files, nothing is transcribed by hand, so the figures cannot drift from
the runs they describe.

The key curve is Episode_Reward/parse: 2.6 means "aligned but never inserted",
10 means "inserted every step", and values between are the fraction of steps
seated. Watching it is the only way to see the rise-then-collapse pattern that
makes picking the right checkpoint necessary.

Run:
    PY=/home/whf/whf/miniconda3/envs/isaac_env/bin/python
    $PY scripts/plot_results.py                       # every run under runs/
    $PY scripts/plot_results.py --runs 2026-10-08_22-56-56_vision 2026-10-09_16-00-10_state_long_resume
    $PY scripts/plot_results.py --out /tmp/curves.png
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(PROJECT_ROOT, "runs", "rsl_rl", "pegih_vision")

PANELS = [
    ("Episode_Reward/parse", "sparse reward (2.6 = aligned only, 10 = inserted)", False),
    ("Episode_Reward/dense", "dense reward (weighted)", False),
    ("Loss/value_function", "value function loss  -- explodes before the critic dies", True),
    ("Train/mean_reward", "mean episode reward", False),
]


def read_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        return None
    # a run that was resumed has several event files; the largest is the one
    # that actually accumulated the steps
    path = max(files, key=os.path.getsize)
    ea = EventAccumulator(path)
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for tag, _, _ in PANELS:
        if tag in have:
            s = ea.Scalars(tag)
            out[tag] = ([x.step for x in s], [x.value for x in s])
    return out or None


def main():
    ap = argparse.ArgumentParser(description="Plot training curves from tensorboard event files.")
    ap.add_argument("--runs", nargs="*", default=None, help="run directory names (default: all)")
    ap.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "results", "training_curves.png"))
    args = ap.parse_args()

    names = args.runs or sorted(
        d for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))
    )
    runs = []
    for n in names:
        d = os.path.join(LOG_ROOT, n)
        if not os.path.isdir(d):
            d = n
        data = read_run(d)
        if data:
            runs.append((os.path.basename(d.rstrip("/")), data))
        else:
            print(f"  (skip {n}: no event file with scalars)")

    if not runs:
        raise SystemExit("no runs with tensorboard scalars found")

    fig, axes = plt.subplots(len(PANELS), len(runs),
                             figsize=(4.2 * len(runs), 3.0 * len(PANELS)),
                             squeeze=False)
    for col, (name, data) in enumerate(runs):
        for row, (tag, title, logy) in enumerate(PANELS):
            ax = axes[row][col]
            if tag in data:
                steps, vals = data[tag]
                ax.plot(steps, vals, lw=1.0, color="#1f6fb4")
            else:
                ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes)
            if logy:
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title.split(" -- ")[0], fontsize=7)
            else:
                ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
            if row == len(PANELS) - 1:
                ax.set_xlabel("iteration", fontsize=8)
            if tag == "Episode_Reward/parse":
                ax.axhline(2.6, color="#c0392b", ls=":", lw=0.8)
                ax.axhline(10.0, color="#27ae60", ls=":", lw=0.8)

    fig.suptitle("UR10e vision peg-in-hole — training curves "
                 "(dotted lines on the top row: 2.6 = aligned only, 10 = inserted)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    # also print the headline numbers so they exist outside the image
    print(f"\n{'run':>34s} {'it':>6s} {'parse@last':>11s} {'parse@max':>10s} {'vf_loss@last':>13s}")
    for name, data in runs:
        p = data.get("Episode_Reward/parse", ([], []))
        v = data.get("Loss/value_function", ([], []))
        last = p[1][-1] if p[1] else float("nan")
        mx = max(p[1]) if p[1] else float("nan")
        vf = v[1][-1] if v[1] else float("nan")
        it = p[0][-1] if p[0] else -1
        print(f"{name:>34s} {it:6d} {last:11.3f} {mx:10.3f} {vf:13.3g}")


if __name__ == "__main__":
    main()

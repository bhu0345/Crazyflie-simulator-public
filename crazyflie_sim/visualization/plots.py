"""Static comparison plots for finished runs.

``overview(runs)`` draws several controllers on one 2x3 figure: position
error, altitude, horizontal path, tilt, world-z spin rate and total thrust.
Labels and colours come from the controller registry, so a newly registered
controller shows up without touching this file.

Compare archived runs from the command line::

    python -m crazyflie_sim.visualization.plots RUN_DIR/baseline.npz RUN_DIR/mellinger.npz -o cmp.png
"""

import argparse
from pathlib import Path

import numpy as np

from crazyflie_sim.utils.utils import RPYToRot


def _style(mode):
    from crazyflie_sim.controllers import registry

    try:
        spec = registry.get(mode)
        return spec.label, spec.color
    except ValueError:
        return str(mode), None


def tilt_deg(att):
    att = np.asarray(att, dtype=float)
    n3z = np.array([RPYToRot(*row)[2, 2] for row in att])
    return np.degrees(np.arccos(np.clip(n3z, -1.0, 1.0)))


def spin_dps(att, omega):
    """World-z angular rate (deg/s) from attitude and body rates."""
    att = np.asarray(att, dtype=float)
    omega = np.asarray(omega, dtype=float)
    out = np.empty(len(att))
    for k, (row, w) in enumerate(zip(att, omega)):
        out[k] = RPYToRot(*row).T.dot(w)[2]
    return np.degrees(out)


def overview(runs, event_time=None, event_label="", title=None):
    """runs: {mode_or_label: Simulation.results()}.  Returns the Figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_err, ax_z, ax_xy, ax_tilt, ax_spin, ax_thrust = axes.ravel()
    for name, run in runs.items():
        label, color = _style(name)
        kw = dict(color=color, lw=1.4, label=label)
        t = np.asarray(run["t"])
        ax_err.plot(t, run["err"], **kw)
        ax_z.plot(t, np.asarray(run["pos"])[:, 2], **kw)
        ax_xy.plot(np.asarray(run["pos"])[:, 0], np.asarray(run["pos"])[:, 1], **kw)
        ax_tilt.plot(t, tilt_deg(run["att"]), **kw)
        ax_spin.plot(t, spin_dps(run["att"], run["omega"]), **kw)
        ax_thrust.plot(t, np.sum(np.asarray(run["motor_thrusts"]), axis=1), **kw)
    for ax, ylabel in (
        (ax_err, "|p - p_d| (m)"),
        (ax_z, "z (m)"),
        (ax_tilt, "tilt (deg)"),
        (ax_spin, "world-z rate (deg/s)"),
        (ax_thrust, "total thrust (N)"),
    ):
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if event_time is not None:
            ax.axvline(event_time, color="gray", ls=":", lw=1.0)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(alpha=0.3)
    ax_err.legend(fontsize=8)
    if event_time is not None and event_label:
        ax_err.annotate(event_label, xy=(event_time, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -12), textcoords="offset points", fontsize=8, color="gray")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Overlay archived runs (timeseries .npz files).")
    parser.add_argument("runs", nargs="+", help=".npz files written by experiments/ or runsim --archive")
    parser.add_argument("-o", "--output", default="overview.png")
    parser.add_argument("--event-time", type=float, default=None)
    args = parser.parse_args(argv)
    runs = {}
    for path in args.runs:
        with np.load(path) as data:
            runs[Path(path).stem] = {key: data[key] for key in data.files}
    overview(runs, event_time=args.event_time).savefig(args.output, dpi=120)
    print("wrote", args.output)


if __name__ == "__main__":
    main()

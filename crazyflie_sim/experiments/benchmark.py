"""Score controllers on the standard benchmark suite (crazyflie_sim/configs/benchmark/).

    python -m crazyflie_sim.experiments.benchmark                               # every controller
    python -m crazyflie_sim.experiments.benchmark --controllers my_ctrl baseline mellinger
    python -m crazyflie_sim.experiments.benchmark --scenarios hover_gust partial_loss --jobs 4
    python -m crazyflie_sim.experiments.benchmark --controllers my_ctrl --seeds 20

The suite is the numbered YAML files of configs/benchmark/: hover in a gust,
trajectory tracking, a 50 % and a complete rotor loss, the gust on the
navigation EKF, on a mismatched plant and through a lossy command link.  Every
controller flies every scenario identically, and every run is scored the same
way from the disturbance onset (the wind or fault time; 1 s for a trajectory)
to the end:

    survived      no ground contact, finite state, tilt below 90 deg
    rms_err_m     RMS position error
    max_err_m     peak position error
    final_err_m   mean position error over the last 2 s
    max_tilt_deg  peak tilt
    settle_s      time until the error stays below 0.1 m for 1 s (NaN: never)
    saturation_pct  share of PWM samples at a limit
    error         None, or the exception the run raised: it then counts as not
                  survived with NaN metrics, and the rest of the suite still runs

Seeds.  A scenario whose runs depend on its seeds (estimator noise, Dryden
turbulence, a lossy or jittery command link) is flown ``--seeds`` times;
replicate k adds k to each of those seeds, so replicate 0 is the config as
written and every controller flies the same replicates.  A deterministic
scenario is flown once.  Seeds inside a controller are not varied.

Statistics.  Per scenario and controller: the median RMS error over the seeds
with a 95 % bootstrap interval, and the paired ratio to the reference
controller (``--reference``, default baseline): the geometric mean over the
seeds of rms / rms_reference with a 95 % bootstrap interval.  A controller is
better or worse than the reference when that interval excludes 1 (a
percentile bootstrap, so approximate for ten seeds; exact for a deterministic
scenario).  The score is the geometric mean of the ratios over the scenarios:
it does not depend on which other controllers flew.

A new controller registered in controllers/ is scored by naming it.  Output:
<results>/<stamp>_benchmark/ with scorecard.md, scorecard.json (every
replicate's metrics), scorecard.png and, per scenario, overview.png and one
time series per controller, of replicate 0.
"""

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import crazyflie_sim.model.params as params
from crazyflie_sim.controllers import registry
from crazyflie_sim.experiments.compare import archive_plain
from crazyflie_sim.sim import archive, metrics as sim_metrics
from crazyflie_sim.sim.config import load_config, params_overridden
from crazyflie_sim.sim.simulation import run as run_scenario

SUITE_DIR = Path(__file__).resolve().parents[1] / "configs" / "benchmark"
SETTLE_M = 0.1
SETTLE_HOLD_S = 1.0
NO_EVENT_START_S = 1.0  # scoring start of a scenario without a disturbance
FINAL_WINDOW_S = 2.0
CRASH_TILT_DEG = 90.0
METRICS = ("survived", "rms_err_m", "max_err_m", "final_err_m", "max_tilt_deg",
           "settle_s", "saturation_pct")
SEEDS = 10  # replicates of a stochastic scenario
REFERENCE = "baseline"  # the controller every ratio is taken against
BOOTSTRAP_RESAMPLES = 2000


def suite(names=None):
    """{scenario name: config path} in file order; files starting with "_" are bases."""
    out = {}
    for path in sorted(SUITE_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        out[load_config(path).scenario.name] = path
    if names:
        unknown = sorted(set(names) - set(out))
        if unknown:
            raise ValueError("unknown benchmark scenario(s) {}; available: {}".format(
                unknown, ", ".join(out)))
        out = {name: path for name, path in out.items() if name in names}
    return out


def stochastic(scenario):
    """True when the scenario's seeds change its runs (see the module docstring)."""
    wind = scenario.active_wind()
    link = scenario.control_link
    return bool(
        scenario.estimator.get("enabled", False)
        or (wind is not None and "dryden" in (wind.x.mode, wind.y.mode, wind.z.mode))
        or (link.get("enabled") and (link["dropout_probability"] > 0.0
                                     or link["jitter_std_steps"] > 0.0)))


def replicate(scenario, k):
    """The scenario with k added to each of its seeds; replicate 0 is the scenario itself."""
    if k == 0:
        return scenario
    return dataclasses.replace(
        scenario,
        seed=scenario.seed + k,
        wind=dataclasses.replace(scenario.wind, dryden_seed=scenario.wind.dryden_seed + k),
        control_link=dict(scenario.control_link, seed=scenario.control_link.get("seed", 0) + k),
    )


def _tilt_deg(att):
    att = np.asarray(att, dtype=float)
    return np.degrees(np.arccos(np.clip(np.cos(att[:, 0]) * np.cos(att[:, 1]), -1.0, 1.0)))


def _settle_time(t, err, start):
    """Seconds from ``start`` until err stays below SETTLE_M for SETTLE_HOLD_S."""
    inside = err < SETTLE_M
    hold = max(1, int(round(SETTLE_HOLD_S / max(float(t[1] - t[0]), 1e-9)))) if len(t) > 1 else 1
    first = np.flatnonzero(t >= start)
    for i in range(first[0] if len(first) else len(t), len(t) - hold + 1):
        if inside[i:i + hold].all():
            return float(t[i] - start)
    return float("nan")


def score(scenario, run):
    """The benchmark metrics of one run (see the module docstring)."""
    t = np.asarray(run["t"], dtype=float)
    pos = np.asarray(run["pos"], dtype=float)
    err = np.asarray(run["err"], dtype=float)
    event, _ = scenario.event()
    start = NO_EVENT_START_S if event is None else float(event)
    window = t >= start
    tilt = _tilt_deg(run["att"])
    finite = all(np.all(np.isfinite(run[k])) for k in ("pos", "vel", "att", "omega", "motor_pwm"))
    max_tilt = float(np.nanmax(tilt[window]))
    survived = bool(finite and np.all(pos[window, 2] > 0.0) and max_tilt < CRASH_TILT_DEG)
    e = err[window]
    final = t >= t[-1] - FINAL_WINDOW_S
    return {
        "survived": survived,
        "rms_err_m": float(np.sqrt(np.mean(e ** 2))),
        "max_err_m": float(np.max(e)),
        "final_err_m": float(np.mean(err[final])),
        "max_tilt_deg": max_tilt,
        "settle_s": _settle_time(t, err, start) if event is not None else float("nan"),
        "saturation_pct": 100.0 * sim_metrics.pwm_saturation_ratio(
            np.asarray(run["motor_pwm"])[window], params.pwm_min, params.pwm_max),
    }


def fly(path, mode, duration_s=None, k=0):
    """Fly replicate ``k`` of one benchmark scenario with one controller: (run, metrics)."""
    cfg = load_config(path)
    if duration_s is not None:
        cfg.scenario.duration_s = float(duration_s)
    scenario = replicate(cfg.scenario, k)
    with params_overridden(cfg.params_overrides):
        run = run_scenario(scenario, mode, cfg.gains(mode))
    return run, score(scenario, run)


def replicates(path, seeds):
    """How many times one scenario is flown: ``seeds`` if it is stochastic, else once."""
    return int(seeds) if stochastic(load_config(path).scenario) else 1


def _job(args):
    name, path, mode, k, duration_s = args
    start = time.perf_counter()
    try:
        run, metrics = fly(path, mode, duration_s, k)
        metrics["error"], trace = None, None
    except Exception as exc:  # one failing controller must not stop the suite
        run, trace = {}, traceback.format_exc()
        metrics = dict({key: float("nan") for key in METRICS}, survived=False,
                       error=traceback.format_exception_only(type(exc), exc)[-1].strip())
    # only replicate 0 keeps its time series (plots and archive)
    keep = {key: v for key, v in run.items() if isinstance(v, np.ndarray)} if k == 0 else None
    return name, mode, k, keep, metrics, trace, time.perf_counter() - start


def benchmark(controllers, scenarios=None, jobs=None, duration_s=None, out_dir=None,
              plots=True, progress=True, seeds=SEEDS, reference=REFERENCE):
    """Fly every (scenario, replicate, controller).

    Returns {scenario: {mode: [metrics of replicate 0, 1, ...]}}.  The
    reference controller is flown too when ``controllers`` leaves it out.
    With ``out_dir`` the time series, overview plots and scorecard are written
    there.  ``duration_s`` shortens every scenario (for smoke tests only: the
    scores are defined on the full durations).
    """
    controllers = list(controllers)
    if reference not in controllers:
        controllers.append(reference)
    for mode in controllers:
        registry.get(mode)
    if int(seeds) < 1:
        raise ValueError("seeds must be at least 1")
    cases = suite(scenarios)
    counts = {name: replicates(path, seeds) for name, path in cases.items()}
    todo = [(name, path, mode, k, duration_s) for name, path in cases.items()
            for k in range(counts[name]) for mode in controllers]
    jobs = max(1, min(int(jobs or os.cpu_count() or 1), len(todo)))
    results = {name: {mode: [None] * counts[name] for mode in controllers} for name in cases}
    runs = {name: {} for name in cases}
    if jobs == 1:
        done = map(_job, todo)
    else:
        pool = cf.ProcessPoolExecutor(max_workers=jobs)
        done = pool.map(_job, todo)
    try:
        for name, mode, k, run, metrics, trace, wall in done:
            results[name][mode][k] = metrics
            if run:
                runs[name][mode] = run
            if progress:
                status = "ERROR" if metrics["error"] else "ok" if metrics["survived"] else "CRASHED"
                print("  {:<16} {:<22} seed {:<3} {:<8} rms {:7.4f} m  ({:.1f} s)".format(
                    name, mode, k, status, metrics["rms_err_m"], wall), flush=True)
            if trace:
                print("{} / {} / seed {}:\n{}".format(name, mode, k, trace),
                      file=sys.stderr, flush=True)
    finally:
        if jobs > 1:
            pool.shutdown()
    # keep the requested controller order whatever order the workers finished in
    runs = {n: {m: runs[n][m] for m in controllers if m in runs[n]} for n in cases}
    if out_dir is not None:
        write_outputs(Path(out_dir), cases, results, runs, plots=plots, reference=reference)
    return results


# ---------------------------------------------------------------------------
# Statistics over the seeds
# ---------------------------------------------------------------------------
def _interval(values, stat):
    """95 % percentile-bootstrap interval of ``stat``, a numpy reduction taking axis=."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(stat(values)), float(stat(values))
    rng = np.random.default_rng(0)  # the same resamples every time: scorecards reproduce
    picks = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    lo, hi = np.percentile(stat(values[picks], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def summary(per_seed, reference_per_seed):
    """Seed statistics of one controller on one scenario (see the module docstring)."""
    rms = np.array([m["rms_err_m"] for m in per_seed], dtype=float)
    ref = np.array([m["rms_err_m"] for m in reference_per_seed], dtype=float)
    nan = float("nan")
    out = {"seeds": len(per_seed),
           "survived": int(sum(bool(m["survived"]) for m in per_seed)),
           "errors": int(sum(m.get("error") is not None for m in per_seed)),
           "rms_median_m": nan, "rms_ci_m": [nan, nan], "ratio": nan, "ratio_ci": [nan, nan]}
    ok = np.isfinite(rms)
    if ok.any():
        out["rms_median_m"] = float(np.median(rms[ok]))
        out["rms_ci_m"] = list(_interval(rms[ok], np.median))
    paired = ok & np.isfinite(ref)
    if paired.any():
        log_ratio = np.log(np.maximum(rms[paired], 1e-12) / np.maximum(ref[paired], 1e-12))
        lo, hi = _interval(log_ratio, np.mean)
        out["ratio"] = float(np.exp(np.mean(log_ratio)))
        out["ratio_ci"] = [float(np.exp(lo)), float(np.exp(hi))]
    return out


def verdict(s):
    """"better" / "worse" than the reference when the ratio's interval excludes 1, else "same"."""
    lo, hi = s["ratio_ci"]
    return "better" if hi < 1.0 else "worse" if lo > 1.0 else "same"


def summaries(results, reference=REFERENCE):
    """{scenario: {mode: summary}} against the reference controller."""
    return {name: {m: summary(row[m], row[reference]) for m in row}
            for name, row in results.items()}


def scores(summ):
    """{mode: geometric mean over the scenarios of the ratio}; NaN when a scenario has none."""
    out = {}
    for m in next(iter(summ.values())):
        ratios = np.array([row[m]["ratio"] for row in summ.values()], dtype=float)
        out[m] = float(np.exp(np.mean(np.log(ratios)))) if np.all(np.isfinite(ratios)) else float("nan")
    return out


def _median(per_seed, key):
    """Median over the seeds; a run that never settles counts as infinitely slow."""
    v = np.array([m[key] for m in per_seed], dtype=float)
    if key == "settle_s":
        v = np.where(np.isnan(v), np.inf, v)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _mm(value):
    if not np.isfinite(value):
        return "-"
    mm = 1000.0 * value
    return "{:.1f}".format(mm) if mm < 100.0 else "{:.0f}".format(mm)


def _num(value, fmt="{:.1f}"):
    return fmt.format(value) if np.isfinite(value) else "-"


def _rms(s):
    """Median RMS (mm) and, over several seeds, its interval."""
    text = _mm(s["rms_median_m"])
    if s["seeds"] > 1 and np.isfinite(s["rms_median_m"]):
        text += " [{}, {}]".format(_mm(s["rms_ci_m"][0]), _mm(s["rms_ci_m"][1]))
    return text


def _cell(s):
    if s["errors"] == s["seeds"]:
        return "error"
    if s["survived"] == 0:
        return "crash"
    if s["survived"] < s["seeds"]:
        return "{}, {}/{} survived".format(_rms(s), s["survived"], s["seeds"])
    return _rms(s)


def _ratio(s):
    if not np.isfinite(s["ratio"]):
        return "-"
    text = "{:.2f}".format(s["ratio"])
    if s["seeds"] > 1:
        text += " [{:.2f}, {:.2f}]".format(*s["ratio_ci"])
    return text + {"better": " better", "worse": " worse", "same": ""}[verdict(s)]


def scorecard_markdown(results, reference=REFERENCE):
    summ = summaries(results, reference)
    total = scores(summ)
    modes = list(next(iter(results.values())))
    labels = [registry.get(m).label for m in modes]
    ref_label = registry.get(reference).label
    lines = ["# Benchmark scorecard", "",
             "Median RMS position error after the disturbance onset over the seeds, mm, with its "
             "95 % bootstrap interval (**best** per scenario among the controllers that survived "
             "every seed; `crash` = ground contact, tilt over 90 deg or a non-finite state on "
             "every seed; `error` = the controller raised).", "",
             "| scenario | seeds | " + " | ".join(labels) + " |",
             "|---" * (len(modes) + 2) + "|"]
    for name, row in summ.items():
        alive = [m for m in modes if row[m]["survived"] == row[m]["seeds"]]
        best = min(alive, key=lambda m: row[m]["rms_median_m"]) if alive else None
        cells = ["**{}**".format(_cell(row[m])) if m == best else _cell(row[m]) for m in modes]
        lines.append("| {} | {} | {} |".format(name, row[modes[0]]["seeds"], " | ".join(cells)))
    survived = {m: sum(row[m]["survived"] == row[m]["seeds"] for row in summ.values())
                for m in modes}
    lines.append("| survived | | " + " | ".join(
        "{}/{}".format(survived[m], len(summ)) for m in modes) + " |")
    lines.append("| score vs {} | | ".format(ref_label) + " | ".join(
        _num(total[m], "{:.2f}") for m in modes) + " |")
    lines += ["",
              "survived: scenarios survived on every seed.  score: geometric mean over the "
              "scenarios of the paired RMS ratio to {} (below 1 is better).".format(ref_label),
              "", "## Per scenario", "",
              "`vs {}`: geometric mean over the seeds of RMS / RMS of {} with its 95 % "
              "bootstrap interval; `better` / `worse` when the interval excludes 1.  The other "
              "columns are medians over the seeds.".format(ref_label, ref_label), ""]
    for name, row in results.items():
        seeds = summ[name][modes[0]]["seeds"]
        lines += ["### {} ({} seed{})".format(name, seeds, "s" if seeds > 1 else ""), "",
                  "| controller | survived | RMS (mm) | vs {} | max (mm) | final (mm) "
                  "| max tilt (deg) | settle (s) | saturation (%) |".format(ref_label),
                  "|---|---|---|---|---|---|---|---|---|"]
        errors = []
        for m in modes:
            s, per_seed = summ[name][m], row[m]
            survived = "{}/{}".format(s["survived"], s["seeds"])
            if s["errors"]:
                survived += " ({} error)".format(s["errors"])
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                registry.get(m).label, survived, _rms(s), _ratio(s),
                _mm(_median(per_seed, "max_err_m")), _mm(_median(per_seed, "final_err_m")),
                _num(_median(per_seed, "max_tilt_deg")),
                _num(_median(per_seed, "settle_s"), "{:.2f}"),
                _num(_median(per_seed, "saturation_pct"))))
            errors += ["- {} seed {}: `{}`".format(registry.get(m).label, k, metrics["error"])
                       for k, metrics in enumerate(per_seed) if metrics.get("error")]
        lines += [""] + (errors + [""] if errors else [])
    return "\n".join(lines)


# Sequential blue ramp (light = close to the best controller, dark = worse).
_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_CRASH = "#e4e3df"


def scorecard_figure(summ):
    """Heatmap of summaries(): median RMS error relative to the best of each scenario."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    modes = list(next(iter(summ.values())))
    names = list(summ)
    ratio = np.full((len(names), len(modes)), np.nan)
    for i, name in enumerate(names):
        row = summ[name]
        alive = [row[m]["rms_median_m"] for m in modes if row[m]["survived"]]
        best = min(alive) if alive else np.nan
        for j, m in enumerate(modes):
            if row[m]["survived"]:
                ratio[i, j] = np.log2(max(row[m]["rms_median_m"] / max(best, 1e-12), 1.0))
    cmap = LinearSegmentedColormap.from_list("benchmark_blue", _RAMP).with_extremes(bad=_CRASH)
    fig, ax = plt.subplots(figsize=(1.25 * len(modes) + 3.0, 0.62 * len(names) + 2.2))
    image = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, vmin=0.0, vmax=4.0, aspect="auto")
    for i, name in enumerate(names):
        for j, m in enumerate(modes):
            s = summ[name][m]
            if not s["survived"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#a9a8a3", lw=0))
                ax.text(j, i, "error" if s["errors"] == s["seeds"] else "crash",
                        ha="center", va="center", fontsize=8, color="#3d3c39")
            else:
                text = _mm(s["rms_median_m"])
                if s["survived"] < s["seeds"]:
                    text += "\n{}/{}".format(s["survived"], s["seeds"])
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if ratio[i, j] > 2.0 else "#1f1e1c")
    ax.set_xticks(range(len(modes)), [registry.get(m).label for m in modes],
                  rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(modes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.035, pad=0.02)
    bar.ax.set_yticklabels(["best", "2x", "4x", "8x", "16x+"], fontsize=8)
    bar.outline.set_visible(False)
    ax.set_title("Median RMS position error after the disturbance (mm), relative to the best "
                 "controller per scenario", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def write_outputs(out_dir, cases, results, runs, plots=True, reference=REFERENCE):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in cases:
        case_dir = out_dir / name
        case_dir.mkdir(exist_ok=True)
        for mode, run in runs[name].items():
            archive.save_timeseries(case_dir / "{}.npz".format(mode), run)
    summ = summaries(results, reference)
    (out_dir / "scorecard.json").write_text(json.dumps(archive_plain(
        {"scenarios": {n: str(p.relative_to(SUITE_DIR.parents[1])) for n, p in cases.items()},
         "reference": reference, "score": scores(summ), "summary": summ,
         "results": results}), indent=2))
    (out_dir / "scorecard.md").write_text(scorecard_markdown(results, reference) + "\n")
    if not plots:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crazyflie_sim.visualization.plots import overview

    fig = scorecard_figure(summ)
    fig.savefig(out_dir / "scorecard.png", dpi=140)
    plt.close(fig)
    for name, path in cases.items():
        if not runs[name]:
            continue
        scenario = load_config(path).scenario
        event_time, event_label = scenario.event()
        fig = overview(runs[name], event_time, event_label, title=name)
        fig.savefig(out_dir / name / "overview.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m crazyflie_sim.experiments.benchmark",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--controllers", nargs="+", choices=registry.available(),
                        default=list(registry.available()),
                        help="default: every registered controller")
    parser.add_argument("--scenarios", nargs="+", help="default: the whole suite ({})".format(
        ", ".join(suite())))
    parser.add_argument("--seeds", type=int, default=SEEDS,
                        help="replicates of a stochastic scenario (default {})".format(SEEDS))
    parser.add_argument("--reference", choices=registry.available(), default=REFERENCE,
                        help="controller the ratios are taken against (default {})".format(
                            REFERENCE))
    parser.add_argument("--jobs", type=int, default=None, help="worker processes (default: CPUs)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for the archive directory name")
    args = parser.parse_args(argv)

    run_dir = archive.make_run_dir("benchmark" + ("_" + args.tag if args.tag else ""))
    archive.write_metadata(run_dir, {"suite": str(SUITE_DIR)},
                           extra={"controllers": args.controllers, "scenarios": args.scenarios,
                                  "seeds": args.seeds, "reference": args.reference})
    print("benchmark -> {}".format(run_dir), flush=True)
    start = time.perf_counter()
    results = benchmark(args.controllers, args.scenarios, jobs=args.jobs, out_dir=run_dir,
                        plots=not args.no_plots, seeds=args.seeds, reference=args.reference)
    print()
    print(scorecard_markdown(results, args.reference).split("\n## Per scenario")[0])
    print("({:.0f} s) archived to {}".format(time.perf_counter() - start, run_dir))


if __name__ == "__main__":
    main()

"""Fly several controllers on the same scenario and compare them.

    python -m crazyflie_sim.experiments.compare hover_wind_default.yaml
    python -m crazyflie_sim.experiments.compare hover_rotor_fault_default.yaml \\
        --controllers baseline mellinger full_state_feedback --fault-cases power_loss_25pct power_loss_50pct
    python -m crazyflie_sim.experiments.compare hover_wind_monte_carlo.yaml --monte-carlo --jobs 8

Every controller flies the identical scenario (plant, wind, fault, estimator
seed).  The metrics follow sim_mode: wind rejection after the wind onset,
fault tolerance after the fault, tracking error on a trajectory.  Output goes
to one archive directory: metrics.json, summary.md, overview.png and one
compressed time series per controller (``visualization.plots`` re-plots them).
"""

import argparse
import concurrent.futures as cf
import copy
import json
import os
import time

import numpy as np

from crazyflie_sim.controllers import registry
from crazyflie_sim.experiments import monte_carlo
from crazyflie_sim.sim import archive
from crazyflie_sim.sim.archive import archive_plain
from crazyflie_sim.sim.metrics import compute_metrics
from crazyflie_sim.sim.config import load_config, params_overridden
from crazyflie_sim.sim.scenario import FAULT_CASE_EFFECTIVENESS
from crazyflie_sim.sim.simulation import run as run_scenario



def compare(cfg, controllers, progress=False, verbose=True):
    """Run every controller on cfg.scenario; returns ({mode: run}, {mode: metrics})."""
    runs, results = {}, {}
    with params_overridden(cfg.params_overrides):
        for mode in controllers:
            start = time.perf_counter()
            run = run_scenario(cfg.scenario, mode, cfg.gains(mode), progress=progress)
            runs[mode] = run
            results[mode] = compute_metrics(cfg.scenario, run, cfg.metrics)
            if verbose:
                print("  {:<22} {:6.1f} s".format(mode, time.perf_counter() - start), flush=True)
    return runs, results


def summary_table(results):
    keys = sorted({k for m in results.values() for k, v in m.items() if isinstance(v, float)})
    keys = [k for k in keys if k not in ("wall_time_s",)][:8]
    lines = ["| controller | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for mode, values in results.items():
        cells = ["{:.4g}".format(values[k]) if k in values else "" for k in keys]
        lines.append("| {} | {} |".format(registry.get(mode).label, " | ".join(cells)))
    return "\n".join(lines)


def write_case(out_dir, cfg, runs, results, plots=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, run in runs.items():
        archive.save_timeseries(out_dir / "{}.npz".format(mode), run)
    (out_dir / "metrics.json").write_text(json.dumps(archive_plain(results), indent=2))
    (out_dir / "summary.md").write_text(summary_table(results) + "\n")
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from crazyflie_sim.visualization.plots import overview

        event_time, event_label = cfg.scenario.event()
        fig = overview(runs, event_time, event_label, title=cfg.scenario.name)
        fig.savefig(out_dir / "overview.png", dpi=120)
        plt.close(fig)


def _fly_replicate(args):
    replicate, controllers, keep_runs = args
    start = time.perf_counter()
    runs, results = compare(replicate, controllers, verbose=False)
    return (runs if keep_runs else None), results, time.perf_counter() - start


def run_monte_carlo(cfg, controllers, run_dir, figures=True, jobs=1):
    """Fly every replicate of cfg.monte_carlo.  All perturbations are drawn
    first, in replicate order, so ``jobs`` worker processes give the same
    report as one."""
    spec = monte_carlo.validate_monte_carlo_spec(cfg.monte_carlo)
    rng = np.random.default_rng(spec["seed"])
    per_replicate, replicates, todo = [], [], []
    base_seed = cfg.scenario.seed
    for i in range(spec["n_runs"]):
        overrides = monte_carlo.sample_perturbations(
            spec["perturb"], rng, cfg.scenario.actuator_profile
        )
        replicate = copy.deepcopy(cfg)
        replicate.scenario.seed = base_seed + i  # same noise for every controller
        plant, rest = monte_carlo.split_overrides(overrides)
        replicate.scenario.plant_overrides = dict(cfg.scenario.plant_overrides, **plant)
        replicate.params_overrides = dict(cfg.params_overrides, **rest)
        replicates.append({"index": i, "plant_overrides": plant, "params_overrides": rest})
        todo.append((replicate, list(controllers), i == 0 and figures))
    jobs = max(1, min(int(jobs or os.cpu_count() or 1), len(todo)))
    pool = cf.ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        done = pool.map(_fly_replicate, todo) if pool else map(_fly_replicate, todo)
        for i, (runs, results, wall) in enumerate(done):
            print("[replicate {}/{}] {:.1f} s  {}".format(
                i + 1, len(todo), wall, dict(replicates[i]["plant_overrides"],
                                             **replicates[i]["params_overrides"])), flush=True)
            per_replicate.append(results)
            replicates[i]["metrics"] = results
            if runs is not None:
                write_case(run_dir / "replicate_0", todo[i][0], runs, results)
    finally:
        if pool:
            pool.shutdown()
    aggregate = monte_carlo.aggregate_metrics(per_replicate)
    report = {"spec": spec, "replicates": replicates, "aggregate": aggregate}
    (run_dir / "monte_carlo_report.json").write_text(json.dumps(archive_plain(report), indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m crazyflie_sim.experiments.compare", description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a config path, or the name of a bundled config "
                                           "in crazyflie_sim/configs/")
    parser.add_argument("--controllers", nargs="+", choices=registry.available(),
                        default=list(registry.available()),
                        help="default: every registered controller")
    parser.add_argument("--fault-cases", nargs="+", choices=sorted(FAULT_CASE_EFFECTIVENESS),
                        help="hover_rotor_fault only: run each case (default: the config's)")
    parser.add_argument("--duration", type=float, help="override duration_s")
    parser.add_argument("--monte-carlo", action="store_true", help="use the config's monte_carlo section")
    parser.add_argument("--jobs", type=int, help="Monte Carlo worker processes (default: all cores)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for the archive directory name")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.duration:
        cfg.scenario.duration_s = args.duration
    run_dir = archive.make_run_dir("compare_" + cfg.name + ("_" + args.tag if args.tag else ""))
    archive.write_metadata(run_dir, cfg.to_dict(), extra={"controllers": args.controllers},
                           source_config_path=cfg.source)
    print("{} -> {}".format(args.config, run_dir))

    if args.monte_carlo:
        run_monte_carlo(cfg, args.controllers, run_dir, jobs=args.jobs or os.cpu_count())
        return

    cases = [None]
    if args.fault_cases and cfg.scenario.sim_mode == "hover_rotor_fault":
        cases = args.fault_cases
    for case in cases:
        case_cfg = copy.deepcopy(cfg)
        out_dir = run_dir
        if case is not None:
            case_cfg.scenario.rotor_fault.case = case
            case_cfg.scenario.name = "{} ({})".format(cfg.scenario.name, case)
            out_dir = run_dir / case
            print("fault case:", case)
        runs, results = compare(case_cfg, args.controllers)
        write_case(out_dir, case_cfg, runs, results, plots=not args.no_plots)
        print(summary_table(results))
    print("archived to", run_dir)


if __name__ == "__main__":
    main()

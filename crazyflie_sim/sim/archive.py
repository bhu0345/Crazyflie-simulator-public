"""Run archives: timestamped directories with config, environment and data.

Runs go to ``$SIM_RESULTS_DIR`` when it is set, else ``results/runs/`` under
the current directory (git-ignored in a checkout).  Keep large runs outside
the repository.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

RUNS_ROOT = Path(
    os.environ.get("SIM_RESULTS_DIR", Path.cwd() / "results" / "runs")
)


def _slug(name):
    slug = re.sub(r"[^A-Za-z0-9_]+", "-", str(name)).strip("-").lower()
    return slug or "run"


def make_run_dir(name, root=None) -> Path:
    """Create ``<root>/<YYYYmmdd_HHMMSS>_<slug(name)>/`` (UTC).

    If the directory already exists, append ``_2``, ``_3``, ... until a fresh
    path is found. Never overwrites an existing run directory.
    """
    root = Path(RUNS_ROOT if root is None else root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = "{}_{}".format(stamp, _slug(name))
    candidate = root / base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / "{}_{}".format(base, suffix)
    candidate.mkdir(parents=True)
    return candidate


def _git(args):
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git failed")
    return result.stdout.strip()


def git_info() -> dict:
    """Best-effort git provenance; "unknown" values on any failure."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    try:
        info["commit"] = _git(["rev-parse", "HEAD"])
        info["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        info["dirty"] = bool(_git(["status", "--porcelain"]))
    except Exception:
        pass
    return info


def env_info() -> dict:
    """Runtime environment snapshot."""
    import scipy

    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": sys.platform,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_metadata(run_dir, resolved_config: dict, extra: dict = None, source_config_path=None) -> None:
    """Write ``config.yaml`` (resolved config) and ``env.json`` into run_dir.

    When ``source_config_path`` is given, the original experiment config file
    is also copied verbatim to ``original_config<suffix>`` so the run can be
    reproduced with ``--config <run_dir>/original_config<suffix>``.
    """
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False)
    )
    env_payload = {"git": git_info(), "environment": env_info()}
    if extra:
        env_payload["extra"] = extra
    (run_dir / "env.json").write_text(json.dumps(env_payload, indent=2))
    if source_config_path is not None:
        src = Path(source_config_path)
        (run_dir / ("original_config" + src.suffix)).write_text(src.read_text())


def save_timeseries(path, run):
    """Write the array columns of a Simulation result (compressed .npz)."""
    arrays = {}
    for key, value in run.items():
        value = np.asarray(value)
        if value.dtype.kind in "fiub" and value.ndim >= 1:
            arrays[key] = value
    np.savez_compressed(path, **arrays)


def load_timeseries(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def archive_plain(value):
    """JSON-ready copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(value, dict):
        return {k: archive_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [archive_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return archive_plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

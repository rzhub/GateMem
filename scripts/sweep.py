#!/usr/bin/env python3
"""Launch GateMem experiment sweeps.

This script is a thin launcher around ``bench/scripts/run_eval.py``.  It supports
both:

1. Explicit run lists, useful for small custom sweeps.
2. Matrix sweeps over domains, models, and baselines, useful for reproducing
   paper-style experiment grids.

Examples:
  # Print commands for medical + GPT-4o-mini + all baselines.
  python scripts/sweep.py --config configs/sweeps/paper_matrix.yaml \
    --domains medical --models gpt4omini --dry_run

  # Run only two RAG baselines on office + Gemini.
  python scripts/sweep.py --config configs/sweeps/paper_matrix.yaml \
    --domains office --models gemini_flash_lite --baselines rag_naive rag_policy

  # Run all combinations specified by the config.
  python scripts/sweep.py --config configs/sweeps/paper_matrix.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


RunSpec = Tuple[str, Dict[str, Any]]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for sweep configs. Install with: pip install pyyaml")
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sweep config root must be a mapping/dict: {p}")
    return data


def _expand_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def _as_mapping(name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping/dict")
    return value


def _selected_keys(
    all_keys: List[str],
    requested: List[str] | None,
    field_name: str,
) -> List[str]:
    if not requested:
        return all_keys
    missing = [k for k in requested if k not in all_keys]
    if missing:
        raise ValueError(
            f"Unknown {field_name}: {missing}. Available {field_name}: {all_keys}"
        )
    return requested


def _build_matrix_runs(
    cfg: Dict[str, Any],
    *,
    requested_domains: List[str] | None,
    requested_models: List[str] | None,
    requested_baselines: List[str] | None,
) -> List[RunSpec]:
    domains = _as_mapping("domains", cfg.get("domains") or {})
    models = _as_mapping("models", cfg.get("models") or {})
    baselines = _as_mapping("baselines", cfg.get("baselines") or {})

    if not domains or not models or not baselines:
        raise ValueError(
            "Matrix sweep configs must define non-empty 'domains', 'models', and 'baselines'."
        )

    domain_keys = _selected_keys(list(domains.keys()), requested_domains, "domains")
    model_keys = _selected_keys(list(models.keys()), requested_models, "models")
    baseline_keys = _selected_keys(list(baselines.keys()), requested_baselines, "baselines")

    common_overrides = dict(cfg.get("common_overrides") or {})
    template = str(cfg.get("run_name_template") or "{domain}_{model_key}_{agent_key}")

    runs: List[RunSpec] = []
    for domain_key, model_key, baseline_key in itertools.product(domain_keys, model_keys, baseline_keys):
        domain_value = domains[domain_key]
        domain_overrides: Dict[str, Any]
        if isinstance(domain_value, str):
            domain_overrides = {"data_dir": domain_value}
        elif isinstance(domain_value, dict):
            domain_overrides = dict(domain_value)
            domain_overrides.setdefault("data_dir", domain_value.get("data_dir"))
        else:
            raise ValueError(f"Invalid domain entry for {domain_key}: {domain_value!r}")

        model_overrides = dict(models[model_key] or {})
        baseline_overrides = dict(baselines[baseline_key] or {})

        fmt = {
            "domain": domain_key,
            "domain_key": domain_key,
            "model": model_key,
            "model_key": model_key,
            "baseline": baseline_key,
            "baseline_key": baseline_key,
            "agent": baseline_overrides.get("agent", baseline_key),
            "agent_key": baseline_key,
        }
        run_name = template.format(**fmt)

        overrides: Dict[str, Any] = {}
        overrides.update(common_overrides)
        overrides.update(domain_overrides)
        overrides.update(model_overrides)
        overrides.update(baseline_overrides)
        overrides = {k: v for k, v in overrides.items() if v is not None}
        runs.append((run_name, overrides))

    return runs


def _build_legacy_runs(cfg: Dict[str, Any]) -> List[RunSpec]:
    runs: List[RunSpec] = []

    if "runs" in cfg:
        common_overrides = dict(cfg.get("common_overrides") or {})
        for r in cfg["runs"] or []:
            if not isinstance(r, dict):
                raise ValueError(f"Each entry under 'runs' must be a mapping: {r!r}")
            if r.get("enabled") is False:
                continue
            name = str(r.get("name"))
            overrides = dict(common_overrides)
            # Support both direct key/value overrides and an explicit overrides block.
            direct = {k: v for k, v in r.items() if k not in {"name", "enabled", "overrides"}}
            overrides.update(direct)
            overrides.update(dict(r.get("overrides") or {}))
            runs.append((name, {k: v for k, v in overrides.items() if v is not None}))
        return runs

    grid = cfg.get("grid") or {}
    if not isinstance(grid, dict) or not grid:
        raise ValueError(
            "Sweep config must contain either a matrix ('domains', 'models', 'baselines'), "
            "an explicit 'runs' list, or a non-empty 'grid'."
        )

    common_overrides = dict(cfg.get("common_overrides") or {})
    for i, overrides in enumerate(_expand_grid(grid)):
        merged = dict(common_overrides)
        merged.update(overrides)
        name = "__".join([f"{k}={merged[k]}" for k in sorted(merged.keys())])
        runs.append((f"run{i:04d}__{name}", merged))
    return runs


def _build_runs(
    cfg: Dict[str, Any],
    *,
    requested_domains: List[str] | None,
    requested_models: List[str] | None,
    requested_baselines: List[str] | None,
) -> List[RunSpec]:
    if {"domains", "models", "baselines"}.issubset(set(cfg.keys())):
        return _build_matrix_runs(
            cfg,
            requested_domains=requested_domains,
            requested_models=requested_models,
            requested_baselines=requested_baselines,
        )

    if requested_domains or requested_models or requested_baselines:
        raise ValueError(
            "--domains/--models/--baselines filters are only supported for matrix configs "
            "that define 'domains', 'models', and 'baselines'."
        )

    return _build_legacy_runs(cfg)


def _override_to_cli(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in overrides.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
            # False boolean flags are omitted because run_eval.py mostly uses store_true flags.
        elif isinstance(v, (list, tuple)):
            args.append(flag)
            args.extend(str(x) for x in v)
        else:
            args.extend([flag, str(v)])
    return args


def _format_cmd(cmd: List[str]) -> str:
    return shlex.join(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch GateMem experiment sweeps.")
    ap.add_argument("--config", required=True, help="Sweep YAML config")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    ap.add_argument("--domains", nargs="*", default=None, help="Subset of domain keys to run, e.g. medical office")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of model keys to run, e.g. gpt4omini")
    ap.add_argument("--baselines", nargs="*", default=None, help="Subset of baseline keys to run, e.g. rag_naive rag_policy")
    ap.add_argument(
        "--max_parallel_runs",
        type=int,
        default=None,
        help="Override max_parallel_runs in sweep config",
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue launching remaining runs when one run fails. Exit nonzero at the end if any failed.",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = _load_yaml(cfg_path)

    base_config = cfg.get("base_config")
    if not base_config:
        raise ValueError("Sweep config missing 'base_config'")
    base_config_path = Path(str(base_config))
    if not base_config_path.is_absolute():
        base_config_path = repo_root / base_config_path
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    run_name_prefix = str(cfg.get("run_name_prefix") or "").strip()
    out_dir = str(cfg.get("out_dir") or "outputs")
    max_parallel = int(args.max_parallel_runs or cfg.get("max_parallel_runs") or 1)
    max_parallel = max(1, max_parallel)

    run_eval = repo_root / "bench" / "scripts" / "run_eval.py"
    if not run_eval.exists():
        raise FileNotFoundError(f"run_eval.py not found: {run_eval}")

    runs = _build_runs(
        cfg,
        requested_domains=args.domains,
        requested_models=args.models,
        requested_baselines=args.baselines,
    )
    if not runs:
        raise ValueError("Sweep expands to zero runs. Check filters and config entries.")

    print(f"[INFO] repo_root     = {repo_root}")
    print(f"[INFO] sweep_config  = {cfg_path}")
    print(f"[INFO] base_config   = {base_config_path}")
    print(f"[INFO] out_dir       = {out_dir}")
    print(f"[INFO] n_runs        = {len(runs)}")
    print(f"[INFO] max_parallel  = {max_parallel}")
    print(f"[INFO] dry_run       = {args.dry_run}")

    def launch(run_name: str, overrides: Dict[str, Any]) -> int:
        final_name = f"{run_name_prefix}_{run_name}" if run_name_prefix else run_name
        cmd = [
            sys.executable,
            str(run_eval),
            "--config",
            str(base_config_path),
            "--out_dir",
            out_dir,
            "--run_name",
            final_name,
        ]
        cmd += _override_to_cli(overrides)

        print(f"\n[RUN] {final_name}")
        print(_format_cmd(cmd))

        if args.dry_run:
            return 0

        env = os.environ.copy()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
        return int(proc.returncode)

    failures: List[Tuple[str, int]] = []

    if args.dry_run or max_parallel == 1:
        for name, overrides in runs:
            rc = launch(name, overrides)
            if rc != 0:
                failures.append((name, rc))
                if not args.continue_on_error:
                    raise SystemExit(rc)
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {ex.submit(launch, name, overrides): name for name, overrides in runs}
            for fut in as_completed(futs):
                name = futs[fut]
                rc = fut.result()
                if rc != 0:
                    failures.append((name, rc))
                    if not args.continue_on_error:
                        raise SystemExit(rc)

    if failures:
        print("\n[ERROR] Some runs failed:")
        for name, rc in failures:
            print(f"  - {name}: exit code {rc}")
        raise SystemExit(1)

    print("\n[DONE] Sweep completed successfully." if not args.dry_run else "\n[DONE] Dry run completed.")


if __name__ == "__main__":
    main()

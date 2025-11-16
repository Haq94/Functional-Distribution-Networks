
from __future__ import annotations
import sys, json, importlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

param_counter = importlib.import_module("param_counter")

def _count(model_name: str, **kwargs) -> int:
    return param_counter.count_params(model_name, **kwargs)

def _binary_search_int(f, lo: int, hi: int, target: int):
    best_val, best_f, best_gap = lo, f(lo), abs(f(lo) - target)
    l, r = lo, hi
    while l <= r:
        m = (l + r) // 2
        fm = f(m)
        gap = abs(fm - target)
        if gap < best_gap or (gap == best_gap and fm < best_f):
            best_val, best_f, best_gap = m, fm, gap
        if fm < target:
            l = m + 1
        elif fm > target:
            r = m - 1
        else:
            return m, fm
    return best_val, best_f

def find_config_for_target(
    model_name: str,
    fixed_kwargs: Dict[str, Any],
    target: Optional[int] = None,
    target_range: Optional[Tuple[int, int]] = None,
    tune_order: Optional[List[str]] = None,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    max_passes: int = 2,
) -> Dict[str, Any]:
    assert (target is not None) ^ (target_range is not None), "Provide exactly one of target or target_range."
    default_tune_by_model = {
        "mlpnet": ["hidden_dim"],
        "mlpdropoutnet": ["hidden_dim"],
        "bayesnet": ["hidden_dim"],
        "hypernet": ["hyper_hidden_dim"],
        "gausshypernet": ["hyper_hidden_dim", "latent_dim"],
        "ic_fdnet": ["hyper_hidden_dim"],
        "lp_fdnet": ["hyper_hidden_dim"],
        "deepensemblenet": ["num_models"],
    }
    model_key = model_name.lower()
    if tune_order is None:
        tune_order = default_tune_by_model.get(model_key, ["hidden_dim"])
    default_bounds = {
        "hidden_dim": (1, 4096),
        "hyper_hidden_dim": (1, 8192),
        "latent_dim": (1, 512),
        "num_models": (1, 128),
    }
    if bounds is None:
        bounds = default_bounds
    else:
        for k, v in default_bounds.items():
            bounds.setdefault(k, v)

    def make_f(varname: str):
        def f_int(val: int) -> int:
            kwargs_local = dict(fixed_kwargs)
            if model_key == "deepensemblenet" and varname in ("hidden_dim", "dropout_rate"):
                base_kwargs = dict(kwargs_local.get("base_kwargs", {}))
                base_kwargs[varname] = val
                kwargs_local["base_kwargs"] = base_kwargs
            else:
                kwargs_local[varname] = val
            return _count(model_name, **kwargs_local)
        return f_int

    def in_range(cnt: int) -> bool:
        if target_range is None:
            return cnt == target
        lo, hi = target_range
        return lo <= cnt <= hi

    current = _count(model_name, **fixed_kwargs)
    if in_range(current):
        return dict(best_kwargs=fixed_kwargs, best_params=current, status="ok_in_range" if target_range else "ok_exact")

    for _ in range(max_passes):
        for var in tune_order:
            lo, hi = bounds[var]
            f = make_f(var)
            desired = target if target is not None else (target_range[0] + target_range[1]) // 2
            best_val, best_f = _binary_search_int(f, lo, hi, desired)
            if model_key == "deepensemblenet" and var in ("hidden_dim", "dropout_rate"):
                base_kwargs = dict(fixed_kwargs.get("base_kwargs", {}))
                base_kwargs[var] = best_val
                fixed_kwargs["base_kwargs"] = base_kwargs
            else:
                fixed_kwargs[var] = best_val
            current = best_f
            if in_range(current):
                return dict(best_kwargs=fixed_kwargs, best_params=current, status="ok_in_range" if target_range else "ok_exact")

    return dict(best_kwargs=fixed_kwargs, best_params=current, status="closest")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search for architecture hyperparameters to meet a parameter budget.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--target_min", type=int, default=None)
    parser.add_argument("--target_max", type=int, default=None)
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--output_dim", type=int)
    parser.add_argument("--hyper_hidden_dim", type=int)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--tune_order", type=str, nargs="*", default=None)
    parser.add_argument("--bounds", type=str, nargs="*", default=None,
                        help="Bounds like var:lo:hi (e.g., hidden_dim:1:4096)")
    args = parser.parse_args()

    if (args.target is None) == (args.target_min is None or args.target_max is None):
        parser.error("Provide either --target OR both --target_min and --target_max.")

    kwargs = {k: v for k, v in vars(args).items()
              if k not in {"model", "target", "target_min", "target_max", "tune_order", "bounds"} and v is not None}

    if args.model.lower() == "deepensemblenet":
        base_keys = dict(kwargs)
        for k in ["num_models", "base_model_name"]:
            base_keys.pop(k, None)
        kwargs["base_kwargs"] = base_keys

    trange = None
    tgt = args.target
    if tgt is None:
        trange = (args.target_min, args.target_max)

    bounds = None
    if args.bounds:
        bounds = {}
        for spec in args.bounds:
            name, lo, hi = spec.split(":")
            bounds[name] = (int(lo), int(hi))

    result = find_config_for_target(
        model_name=args.model,
        fixed_kwargs=kwargs,
        target=tgt,
        target_range=trange,
        tune_order=args.tune_order,
        bounds=bounds,
    )
    print(json.dumps(result, indent=2))

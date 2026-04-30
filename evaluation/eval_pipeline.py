#!/usr/bin/env python3
"""
Step 4 evaluation pipeline: zero-shot JSON → autoregressive rollout → RMSE / energy drift / ROI.

Usage (from repository root)::

    python evaluation/eval_pipeline.py --checkpoint data/interim/hierarchical_step3_model.pth

Requires eval IR from ``data_generation/generate_eval_data.jl``. Does **not** modify training code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surrogate_model"))

from evaluation.metrics import (  # noqa: E402
    compute_energy_drift,
    compute_rollout_rmse,
    effective_lambda_edges,
)
from evaluation.profiler import InferenceProfiler  # noqa: E402

from model import PhysicsGNNSurrogate  # noqa: E402
from model_hierarchical import HierarchicalPhysicsGNN  # noqa: E402
from utils.index_converter import (  # noqa: E402
    assert_valid_python_edge_index,
    convert_julia_sparse_coo_to_torch,
    convert_julia_to_python_indices,
)


def load_eval_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def series_to_tensor(series: Any, *, device: torch.device) -> Tensor:
    return torch.tensor(series, dtype=torch.float32, device=device)


def build_fine_coarse_edges(
    raw: Dict[str, Any], *, device: torch.device
) -> Tuple[Tensor, Tensor, int, int]:
    fg = raw["fine_graph"]
    cg = raw["coarse_graph"]
    nf = int(fg["num_nodes"])
    nc = int(cg["num_nodes"])
    ef = fg["edges"]
    ec = cg["edges"]
    ei_f = (
        convert_julia_to_python_indices(torch.tensor(ef, dtype=torch.long).t().contiguous(), num_nodes=nf)
        if len(ef) > 0
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    ei_c = (
        convert_julia_to_python_indices(torch.tensor(ec, dtype=torch.long).t().contiguous(), num_nodes=nc)
        if len(ec) > 0
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    assert_valid_python_edge_index(ei_f, num_nodes=nf)
    assert_valid_python_edge_index(ei_c, num_nodes=nc)
    return ei_f.to(device), ei_c.to(device), nf, nc


def build_sparse_rp(raw: Dict[str, Any], *, device: torch.device) -> Tuple[Tensor, Tensor]:
    rspec = raw["restriction"]
    pspec = raw["prolongation"]
    rows_r = torch.tensor(rspec["rows"], dtype=torch.long)
    cols_r = torch.tensor(rspec["cols"], dtype=torch.long)
    vals_r = torch.tensor(rspec["values"], dtype=torch.float32)
    r_mat = convert_julia_sparse_coo_to_torch(
        rows_r,
        cols_r,
        vals_r,
        size_julia_nrows=int(rspec["nrows"]),
        size_julia_ncols=int(rspec["ncols"]),
        dtype_values=torch.float32,
        device=device,
    )
    rows_p = torch.tensor(pspec["rows"], dtype=torch.long)
    cols_p = torch.tensor(pspec["cols"], dtype=torch.long)
    vals_p = torch.tensor(pspec["values"], dtype=torch.float32)
    p_mat = convert_julia_sparse_coo_to_torch(
        rows_p,
        cols_p,
        vals_p,
        size_julia_nrows=int(pspec["nrows"]),
        size_julia_ncols=int(pspec["ncols"]),
        dtype_values=torch.float32,
        device=device,
    )
    return r_mat, p_mat


def guess_architecture(meta: Dict[str, Any], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if "bond" in meta:
        return "hierarchical"
    return "heun"


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    *,
    arch: str,
    raw_eval: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    meta = ckpt.get("meta", {})
    state = ckpt["model"]
    resolved = guess_architecture(meta, arch)

    if resolved == "hierarchical":
        r_mat, p_mat = build_sparse_rp(raw_eval, device=device)
        model = HierarchicalPhysicsGNN(
            state_dim=2,
            hidden_dim=int(meta["hidden"]),
            bond_dim=int(meta["bond"]),
            sparse_r=r_mat,
            sparse_p=p_mat,
            num_fine_layers=int(meta.get("fine_layers", 2)),
            num_coarse_layers=int(meta.get("coarse_layers", 2)),
        ).to(device)
        model.load_state_dict(state)
        return model, resolved

    if resolved == "heun":
        dt = float(meta.get("dt", raw_eval["meta"]["dt"]))
        layers = int(meta.get("layers", meta.get("num_layers", 3)))
        model = PhysicsGNNSurrogate(
            state_dim=2,
            hidden_dim=int(meta["hidden"]),
            num_message_layers=layers,
            dt=dt,
        ).to(device)
        model.load_state_dict(state)
        return model, resolved

    raise ValueError(f"unknown architecture {resolved}")


@torch.no_grad()
def autoregressive_rollout_heun(
    model: PhysicsGNNSurrogate,
    h0: Tensor,
    edge_index: Tensor,
    *,
    steps: int,
) -> Tensor:
    traj = [h0]
    h = h0
    for _ in range(int(steps)):
        h = model(h, edge_index)
        traj.append(h)
    return torch.stack(traj, dim=0)


@torch.no_grad()
def autoregressive_rollout_hierarchical(
    model: HierarchicalPhysicsGNN,
    h0: Tensor,
    edge_f: Tensor,
    edge_c: Tensor,
    *,
    steps: int,
) -> Tensor:
    traj = [h0]
    h = h0
    for _ in range(int(steps)):
        h = model(h, edge_f, edge_c)
        traj.append(h)
    return torch.stack(traj, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4 zero-shot evaluation + ROI")
    ap.add_argument(
        "--eval-json",
        type=str,
        default=str(ROOT / "data" / "interim" / "eval_zero_shot_v1.json"),
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "data" / "interim" / "hierarchical_step3_model.pth"),
    )
    ap.add_argument("--architecture", type=str, default="auto", choices=["auto", "heun", "hierarchical"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-rollout-steps", type=int, default=0, help="0 = use full series length minus one")
    ap.add_argument("--report-json", type=str, default=str(ROOT / "reports" / "evaluation_results.json"))
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--benchmark-iters", type=int, default=30)
    args = ap.parse_args()

    eval_path = Path(args.eval_json)
    ckpt_path = Path(args.checkpoint)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing eval JSON: {eval_path} — run generate_eval_data.jl")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    raw = load_eval_json(eval_path)
    if raw.get("schema") != "physics_gnn_eval_v1":
        print("警告: schema が physics_gnn_eval_v1 でありません。")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meta_ev = raw["meta"]
    lam = effective_lambda_edges(c=float(meta_ev["c"]), dx=float(meta_ev["dx"]))

    u = series_to_tensor(raw["timeseries"]["u"], device=device)
    v = series_to_tensor(raw["timeseries"]["v"], device=device)
    gt = torch.stack([u, v], dim=-1)
    t_total = gt.size(0)
    nf = int(raw["fine_graph"]["num_nodes"])
    if gt.size(1) != nf:
        raise ValueError("timeseries node dimension mismatch")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    resolved_arch = guess_architecture(ckpt.get("meta", {}), args.architecture)
    if args.architecture == "auto":
        print(f"architecture (auto): {resolved_arch}")

    model, arch = build_model_from_ckpt(
        ckpt,
        arch=args.architecture,
        raw_eval=raw,
        device=device,
    )
    model.eval()

    ei_f, ei_c, _, _ = build_fine_coarse_edges(raw, device=device)

    max_steps = t_total - 1
    if args.max_rollout_steps > 0:
        max_steps = min(max_steps, int(args.max_rollout_steps))

    h0 = gt[0].clone()
    with torch.no_grad():
        if arch == "heun":
            pred = autoregressive_rollout_heun(model, h0, ei_f, steps=max_steps)
        else:
            pred = autoregressive_rollout_hierarchical(model, h0, ei_f, ei_c, steps=max_steps)

    tgt = gt[: pred.size(0)].clone()
    rmse = compute_rollout_rmse(pred, tgt, exclude_t0=True)
    drift = compute_energy_drift(pred, ei_f, reference_state_t0=gt[0], lambda_edges=lam)

    julia_per_step = float(meta_ev["julia_seconds_per_macro_step"])

    hBench = h0.clone()

    if arch == "heun":

        def forward_fn() -> Tensor:
            return model(hBench, ei_f)

    else:

        def forward_fn() -> Tensor:
            return model(hBench, ei_f, ei_c)

    profiler = InferenceProfiler(
        device=device,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    gnn_dt = profiler.measure_mean_seconds(forward_fn)
    speedup = julia_per_step / gnn_dt if gnn_dt > 0 else float("inf")

    report: Dict[str, Any] = {
        "schema_eval": raw.get("schema", ""),
        "eval_json": str(eval_path.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "architecture": arch,
        "rollout_rmse": rmse,
        "energy_drift_max_relative": drift,
        "julia_seconds_per_macro_step": julia_per_step,
        "julia_total_solve_seconds": float(meta_ev["julia_total_solve_seconds"]),
        "gnn_seconds_per_step_mean": gnn_dt,
        "speedup_vs_julia_macro_step": speedup,
        "max_rollout_steps": max_steps,
        "device": str(device),
        "profiler_warmup_iters": args.warmup_iters,
        "profiler_benchmark_iters": args.benchmark_iters,
    }

    out_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()

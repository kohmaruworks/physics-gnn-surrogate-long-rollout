"""
Step 3 training: load multigrid JSON (fine/coarse graphs + sparse ``R``, ``P``),
convert Julia indices once, fit ``HierarchicalPhysicsGNN`` to fine-grid rollout targets.

Run::

    python surrogate_model/train_step3.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from model_hierarchical import HierarchicalPhysicsGNN  # noqa: E402
from utils.index_converter import (  # noqa: E402
    assert_valid_python_edge_index,
    convert_julia_sparse_coo_to_torch,
    convert_julia_to_python_indices,
)


def project_root() -> Path:
    return _ROOT


def default_json_path() -> Path:
    return project_root() / "data" / "interim" / "multigrid_wave_v1.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def series_to_tensor(series: List[List[float]]) -> Tensor:
    return torch.tensor(series, dtype=torch.float32)


def build_sparse_operators(
    raw: Dict[str, Any], *, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tuple[Tensor, Tensor]:
    rspec = raw["restriction"]
    pspec = raw["prolongation"]
    rows_r = torch.tensor(rspec["rows"], dtype=torch.long)
    cols_r = torch.tensor(rspec["cols"], dtype=torch.long)
    vals_r = torch.tensor(rspec["values"], dtype=dtype)
    r_mat = convert_julia_sparse_coo_to_torch(
        rows_r,
        cols_r,
        vals_r,
        size_julia_nrows=int(rspec["nrows"]),
        size_julia_ncols=int(rspec["ncols"]),
        dtype_values=dtype,
        device=device,
    )

    rows_p = torch.tensor(pspec["rows"], dtype=torch.long)
    cols_p = torch.tensor(pspec["cols"], dtype=torch.long)
    vals_p = torch.tensor(pspec["values"], dtype=dtype)
    p_mat = convert_julia_sparse_coo_to_torch(
        rows_p,
        cols_p,
        vals_p,
        size_julia_nrows=int(pspec["nrows"]),
        size_julia_ncols=int(pspec["ncols"]),
        dtype_values=dtype,
        device=device,
    )
    return r_mat, p_mat


def build_edge_indices(raw: Dict[str, Any], *, device: torch.device) -> Tuple[Tensor, Tensor, int, int]:
    fg = raw["fine_graph"]
    cg = raw["coarse_graph"]
    nf = int(fg["num_nodes"])
    nc = int(cg["num_nodes"])
    ef = fg["edges"]
    ec = cg["edges"]
    ei_f = (
        convert_julia_to_python_indices(torch.tensor(ef, dtype=torch.long).t().contiguous(), num_nodes=nf)
        if len(ef) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )
    ei_c = (
        convert_julia_to_python_indices(torch.tensor(ec, dtype=torch.long).t().contiguous(), num_nodes=nc)
        if len(ec) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )
    assert_valid_python_edge_index(ei_f, num_nodes=nf)
    assert_valid_python_edge_index(ei_c, num_nodes=nc)
    return ei_f.to(device), ei_c.to(device), nf, nc


@torch.no_grad()
def validation_one_step_mse(
    model: HierarchicalPhysicsGNN,
    series: Tensor,
    ei_f: Tensor,
    ei_c: Tensor,
    *,
    max_start: int,
) -> float:
    errs = []
    for start in range(max_start):
        x_t = series[start]
        y = series[start + 1]
        pred = model(x_t, ei_f, ei_c)
        errs.append(F.mse_loss(pred, y))
    return float(torch.stack(errs).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: hierarchical tensor MP training")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bond", type=int, default=8)
    ap.add_argument("--fine-layers", type=int, default=2)
    ap.add_argument("--coarse-layers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.15)
    args = ap.parse_args()

    path = Path(args.json) if args.json else default_json_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Generate via:\n"
            "  julia --project=. data_generation/generate_multigrid_data.jl"
        )

    raw = load_json(path)
    if raw.get("schema") != "physics_gnn_multigrid_v1":
        print("警告: schema が physics_gnn_multigrid_v1 でありません。")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ei_f, ei_c, nf, nc = build_edge_indices(raw, device=device)
    r_mat, p_mat = build_sparse_operators(raw, device=device, dtype=torch.float32)

    u = series_to_tensor(raw["timeseries"]["u"])
    v = series_to_tensor(raw["timeseries"]["v"])
    state = torch.stack([u, v], dim=-1).to(device)
    t_total = state.size(0)
    if state.size(1) != nf:
        raise ValueError(f"fine trajectory length {state.size(1)} != fine_graph.num_nodes {nf}")

    t_train = max(2, int(math.floor(t_total * (1.0 - args.val_split))))
    train_s = state[:t_train]
    val_s = state[t_train:]

    model = HierarchicalPhysicsGNN(
        state_dim=2,
        hidden_dim=args.hidden,
        bond_dim=args.bond,
        sparse_r=r_mat,
        sparse_p=p_mat,
        num_fine_layers=args.fine_layers,
        num_coarse_layers=args.coarse_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for t in range(train_s.size(0) - 1):
            x_t = train_s[t]
            y = train_s[t + 1]
            opt.zero_grad(set_to_none=True)
            pred = model(x_t, ei_f, ei_c)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        av = tot / max(1, n)

        model.eval()
        if val_s.size(0) > 2:
            vm = validation_one_step_mse(
                model, val_s, ei_f, ei_c, max_start=int(val_s.size(0)) - 1
            )
            print(f"epoch {ep:03d} | train_mse={av:.6f} | val_one_step_mse={vm:.6f}")
        else:
            print(f"epoch {ep:03d} | train_mse={av:.6f}")

    out = project_root() / "data" / "interim" / "hierarchical_step3_model.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "schema": raw.get("schema", ""),
                "hidden": args.hidden,
                "bond": args.bond,
                "fine_layers": args.fine_layers,
                "coarse_layers": args.coarse_layers,
                "nf": nf,
                "nc": nc,
            },
        },
        out,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

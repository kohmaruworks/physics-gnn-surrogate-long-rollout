"""
Training loop: JSON load with Julia→PyG index conversion, teacher-forced +
autoregressive rollout, MSE + SymplecticLoss, validation rollout error.

Run from repository root::

    python surrogate_model/train.py

or from ``surrogate_model/``::

    python train.py
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

from model import PhysicsGNNSurrogate  # noqa: E402
from modules.physics_loss import SymplecticLoss  # noqa: E402
from utils.index_converter import (  # noqa: E402
    assert_valid_python_edge_index,
    convert_julia_to_python_indices,
)


def project_root() -> Path:
    return _ROOT


def default_json_path() -> Path:
    return project_root() / "data" / "interim" / "wave_rollout_step1.json"


def load_wave_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_edge_index_from_payload(raw: Dict[str, Any]) -> Tuple[Tensor, int]:
    num_nodes = int(raw["num_nodes"])
    edges = raw["edges"]
    if len(edges) == 0:
        ei = torch.empty((2, 0), dtype=torch.long)
    else:
        ei_julia = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = convert_julia_to_python_indices(ei_julia, num_nodes=num_nodes)
    assert_valid_python_edge_index(ei, num_nodes=num_nodes)
    return ei, num_nodes


def series_to_tensor(series: List[List[float]]) -> Tensor:
    """Shape ``[T, N]``."""
    return torch.tensor(series, dtype=torch.float32)


def effective_lambda_edges(meta: Dict[str, Any]) -> float:
    """Match discrete stiffness ``≈ c²/dx²``; divide by 2 if edges are bidirected."""
    c = float(meta["c"])
    dx = float(meta["dx"])
    base = (c / dx) ** 2
    # Grid JSON stores both directions; Hamiltonian PE should avoid double counting.
    return 0.5 * base


def train_step_multi(
    model: PhysicsGNNSurrogate,
    h_start: Tensor,
    targets: Tensor,
    edge_index: Tensor,
    sym_loss: SymplecticLoss,
    lambda_symp: float,
    *,
    lambda_edges: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    One optimization step: rollout ``targets.shape[0]-1`` steps from ``h_start``.

    ``targets``: ``[T, N, C]`` ground-truth trajectory (teacher states along time).
    """
    T = targets.size(0)
    pred_list = [h_start]
    h = h_start
    for _ in range(T - 1):
        h = model(h, edge_index)
        pred_list.append(h)
    pred = torch.stack(pred_list, dim=0)
    loss_data = F.mse_loss(pred, targets)
    loss_sym = sym_loss(
        pred,
        edge_index,
        u_channels=(0,),
        v_channels=(1,),
        lambda_edges=lambda_edges,
    )
    loss = loss_data + lambda_symp * loss_sym
    return loss, loss_data.detach(), loss_sym.detach()


@torch.no_grad()
def validation_rollout_mse(
    model: PhysicsGNNSurrogate,
    h0: Tensor,
    targets: Tensor,
    edge_index: Tensor,
    rollout_steps: int,
) -> float:
    """Autoregressive rollout MSE vs ``targets[:rollout_steps+1]``."""
    pred = model.rollout(h0, edge_index, steps=rollout_steps)
    tgt = targets[: rollout_steps + 1]
    return float(F.mse_loss(pred, tgt).item())


def main() -> None:
    p = argparse.ArgumentParser(description="Physics GNN long-rollout stabilization (Step 1)")
    p.add_argument("--json", type=str, default="", help="Wave dataset JSON (default: data/interim/...)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lambda-symp", type=float, default=0.05, dest="lambda_symp")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rollout-min", type=int, default=3, dest="rollout_min")
    p.add_argument("--rollout-max", type=int, default=16, dest="rollout_max")
    p.add_argument("--val-split", type=float, default=0.2, dest="val_split")
    args = p.parse_args()

    json_path = Path(args.json) if args.json else default_json_path()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {json_path}. Run Julia data_generation/generate_wave_data.jl first."
        )

    raw = load_wave_json(json_path)
    if raw.get("schema") != "physics_gnn_wave_rollout_step1_v1":
        print("警告: schema が physics_gnn_wave_rollout_step1_v1 でありません。")

    edge_index, num_nodes = build_edge_index_from_payload(raw)
    meta = raw["meta"]
    dt = float(meta["dt"])
    lambda_edges = effective_lambda_edges(meta)

    u_series = series_to_tensor(raw["timeseries"]["u"])
    v_series = series_to_tensor(raw["timeseries"]["v"])
    if u_series.shape != v_series.shape:
        raise ValueError("timeseries u/v shape mismatch")
    T_total, N = u_series.shape
    if N != num_nodes:
        raise ValueError(f"num_nodes mismatch: JSON {num_nodes}, series N={N}")

    state = torch.stack([u_series, v_series], dim=-1)  # [T, N, 2]

    T_train = max(3, int(math.floor(T_total * (1.0 - args.val_split))))
    train_series = state[:T_train]
    val_series = state[T_train:]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    edge_index = edge_index.to(device)

    model = PhysicsGNNSurrogate(
        state_dim=2,
        hidden_dim=args.hidden,
        num_message_layers=args.layers,
        dt=dt,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sym_loss = SymplecticLoss(lambda_edges=lambda_edges, reduction="mean")

    rollout_min = max(2, args.rollout_min)
    rollout_max = min(args.rollout_max, train_series.size(0) - 1)
    if rollout_max < rollout_min:
        rollout_max = rollout_min

    for ep in range(1, args.epochs + 1):
        progress = (ep - 1) / max(1, args.epochs - 1)
        horizon = int(round(rollout_min + progress * (rollout_max - rollout_min)))
        horizon = max(2, min(horizon, train_series.size(0) - 1))

        model.train()
        tot_loss = 0.0
        tot_data = 0.0
        tot_sym = 0.0
        n_chunks = 0
        for start in range(0, train_series.size(0) - horizon):
            batch = train_series[start : start + horizon + 1].to(device)
            h0 = batch[0]
            targets = batch
            opt.zero_grad(set_to_none=True)
            loss, ld, ls = train_step_multi(
                model,
                h0,
                targets,
                edge_index,
                sym_loss,
                args.lambda_symp,
                lambda_edges=lambda_edges,
            )
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
            tot_data += float(ld)
            tot_sym += float(ls)
            n_chunks += 1

        if n_chunks == 0:
            raise RuntimeError("No training windows; shorten horizon or generate longer series.")

        n_chunks = float(n_chunks)
        print(
            f"epoch {ep:03d} | horizon={horizon} | "
            f"L={(tot_loss / n_chunks):.6f} | "
            f"L_data={(tot_data / n_chunks):.6f} | "
            f"L_sym={(tot_sym / n_chunks):.6f}"
        )

        # Validation: long autoregressive rollout on held-out tail
        if val_series.size(0) > 2:
            model.eval()
            val_steps = min(val_series.size(0) - 1, rollout_max)
            h0v = val_series[0].to(device)
            tgtv = val_series[: val_steps + 1].to(device)
            err = validation_rollout_mse(model, h0v, tgtv, edge_index, val_steps)
            print(f"          val rollout MSE (steps={val_steps}): {err:.6f}")

    out_path = project_root() / "data" / "interim" / "wave_rollout_step1_model.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "schema": raw.get("schema", ""),
                "hidden": args.hidden,
                "layers": args.layers,
                "lambda_symp": args.lambda_symp,
                "lambda_edges": lambda_edges,
                "dt": dt,
            },
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

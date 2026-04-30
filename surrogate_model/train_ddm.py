"""
DDM training: subdomain micro-batching + gradient accumulation (OOM mitigation).

Default loss uses **teacher-forced halo** (ghost rows replaced by global GT before each
Heun step) so each subdomain contributes an independent loss fragment; gradients are
scaled and accumulated across subdomains before ``optimizer.step()``.

Optional ``--joint-ddm-loss`` uses a fully coupled ``rollout_ddm`` (single backward
per time window).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from model import PhysicsGNNSurrogateDDM  # noqa: E402
from modules.physics_loss import SymplecticLoss  # noqa: E402
from utils.index_converter import (  # noqa: E402
    assert_disjoint_core_masks,
    ddm_edge_index_from_julia,
    julia_indices_to_python,
)


def project_root() -> Path:
    return _ROOT


def default_json_path() -> Path:
    return project_root() / "data" / "interim" / "wave_rollout_ddm_v1.json"


def load_wave_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def effective_lambda_edges(meta: Dict[str, Any]) -> float:
    c = float(meta["c"])
    dx = float(meta["dx"])
    return 0.5 * (c / dx) ** 2


def series_to_tensor(series: List[List[float]]) -> Tensor:
    return torch.tensor(series, dtype=torch.float32)


@dataclass
class DDMBundle:
    num_global: int
    edge_indices: List[Tensor]
    global_ids: List[Tensor]
    is_ghost: List[Tensor]


def build_ddm_bundle(raw: Dict[str, Any], *, device: torch.device) -> DDMBundle:
    if raw.get("schema") != "physics_gnn_wave_rollout_ddm_v1":
        print("警告: schema が physics_gnn_wave_rollout_ddm_v1 でありません。")
    num_global = int(raw["global"]["num_nodes"])
    edge_indices: List[Tensor] = []
    global_ids: List[Tensor] = []
    is_ghost: List[Tensor] = []
    core_globals: List[Tensor] = []

    for sd in raw["subdomains"]:
        nloc = int(sd["num_local_nodes"])
        ej = torch.tensor(sd["edges_local"], dtype=torch.long)
        if ej.numel() == 0:
            ei = torch.empty((2, 0), dtype=torch.long)
        else:
            ei = ej.t().contiguous()
        edge_indices.append(ddm_edge_index_from_julia(ei, num_local_nodes=nloc).to(device))

        g_j = torch.tensor([int(n["global_id"]) for n in sd["nodes"]], dtype=torch.long)
        global_ids.append(julia_indices_to_python(g_j, upper_bound_julia=num_global).to(device))

        gh = torch.tensor([bool(n["is_ghost"]) for n in sd["nodes"]], dtype=torch.bool, device=device)
        is_ghost.append(gh)

        core_globals.append(global_ids[-1][~gh])

    assert_disjoint_core_masks(core_globals, num_global=num_global)

    return DDMBundle(
        num_global=num_global,
        edge_indices=edge_indices,
        global_ids=global_ids,
        is_ghost=is_ghost,
    )


def gather_nodes(global_state: Tensor, gid: Tensor) -> Tensor:
    """global_state: ``[N, F]`` → patch ``[N_local, F]``."""
    return global_state[gid.long()]


def train_loss_teacher_ghost_one_sd(
    model: PhysicsGNNSurrogateDDM,
    *,
    edge_index: Tensor,
    gid: Tensor,
    ghost: Tensor,
    targets_global: Tensor,
    start: int,
    horizon: int,
    sym_loss: SymplecticLoss,
    lambda_symp: float,
    lambda_edges: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Teacher-forced ghosts from global GT at each local time index."""
    chunk = targets_global[start : start + horizon + 1]
    preds: List[Tensor] = [gather_nodes(chunk[0], gid)]
    for _t in range(horizon):
        h = preds[-1].clone()
        gt_t = gather_nodes(chunk[_t], gid)
        h[ghost] = gt_t[ghost]
        preds.append(model.forward(h, edge_index))
    pred = torch.stack(preds, dim=0)
    tgt = torch.stack([gather_nodes(chunk[t], gid) for t in range(horizon + 1)], dim=0)
    loss_data = F.mse_loss(pred, tgt)
    loss_sym = sym_loss(
        pred,
        edge_index,
        u_channels=(0,),
        v_channels=(1,),
        lambda_edges=lambda_edges,
    )
    loss = loss_data + lambda_symp * loss_sym
    return loss, loss_data.detach(), loss_sym.detach()


def train_loss_joint_rollout(
    model: PhysicsGNNSurrogateDDM,
    bundle: DDMBundle,
    *,
    targets_global: Tensor,
    start: int,
    horizon: int,
    sym_loss: SymplecticLoss,
    lambda_symp: float,
    lambda_edges: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    h0_list = [gather_nodes(targets_global[start], bundle.global_ids[i]) for i in range(len(bundle.edge_indices))]
    preds = model.rollout_ddm(
        h0_list,
        bundle.edge_indices,
        bundle.global_ids,
        bundle.is_ghost,
        num_global=bundle.num_global,
        steps=horizon,
        sync_before_rollout=True,
        sync_before_each_heun=False,
    )
    tgt_list = [
        torch.stack(
            [gather_nodes(targets_global[start + t], bundle.global_ids[i]) for t in range(horizon + 1)],
            dim=0,
        )
        for i in range(len(preds))
    ]
    loss_data = torch.stack([F.mse_loss(preds[i], tgt_list[i]) for i in range(len(preds))]).mean()
    loss_sym = torch.stack(
        [
            sym_loss(
                preds[i],
                bundle.edge_indices[i],
                u_channels=(0,),
                v_channels=(1,),
                lambda_edges=lambda_edges,
            )
            for i in range(len(preds))
        ]
    ).mean()
    loss = loss_data + lambda_symp * loss_sym
    return loss, loss_data.detach(), loss_sym.detach()


@torch.no_grad()
def validation_ddm_rollout_mse(
    model: PhysicsGNNSurrogateDDM,
    bundle: DDMBundle,
    val_series: Tensor,
    rollout_steps: int,
) -> float:
    h0_list = [gather_nodes(val_series[0], gid) for gid in bundle.global_ids]
    preds = model.rollout_ddm(
        h0_list,
        bundle.edge_indices,
        bundle.global_ids,
        bundle.is_ghost,
        num_global=bundle.num_global,
        steps=rollout_steps,
    )
    errs = []
    for i, pred in enumerate(preds):
        tgt = torch.stack(
            [gather_nodes(val_series[t], bundle.global_ids[i]) for t in range(rollout_steps + 1)],
            dim=0,
        )
        errs.append(F.mse_loss(pred, tgt))
    return float(torch.stack(errs).mean().item())


def main() -> None:
    p = argparse.ArgumentParser(description="DDM subdomain training (Step 2)")
    p.add_argument("--json", type=str, default="")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lambda-symp", type=float, default=0.05, dest="lambda_symp")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rollout-min", type=int, default=3, dest="rollout_min")
    p.add_argument("--rollout-max", type=int, default=12, dest="rollout_max")
    p.add_argument("--val-split", type=float, default=0.2, dest="val_split")
    p.add_argument(
        "--microbatch-subdomains",
        type=int,
        default=0,
        dest="microbatch_subdomains",
        help="Random subset size per window (0 = all subdomains). Gradients still accumulate in-memory.",
    )
    p.add_argument(
        "--joint-ddm-loss",
        action="store_true",
        dest="joint_ddm_loss",
        help="Coupled rollout_ddm loss (single backward per window).",
    )
    args = p.parse_args()

    json_path = Path(args.json) if args.json else default_json_path()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {json_path}. Run: julia --project=. data_generation/generate_large_wave_data.jl"
        )

    raw = load_wave_json(json_path)
    meta = raw["meta"]
    dt = float(meta["dt"])
    lambda_edges = effective_lambda_edges(meta)

    u_series = series_to_tensor(raw["timeseries"]["u"])
    v_series = series_to_tensor(raw["timeseries"]["v"])
    state = torch.stack([u_series, v_series], dim=-1)
    T_total, N = state.shape[:2]
    if N != int(raw["global"]["num_nodes"]):
        raise ValueError("global num_nodes mismatch")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    bundle = build_ddm_bundle(raw, device=device)
    state = state.to(device)

    T_train = max(3, int(math.floor(T_total * (1.0 - args.val_split))))
    train_series = state[:T_train]
    val_series = state[T_train:]

    model = PhysicsGNNSurrogateDDM(
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

    num_sd = len(bundle.edge_indices)

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
            opt.zero_grad(set_to_none=True)

            if args.joint_ddm_loss:
                loss, ld, ls = train_loss_joint_rollout(
                    model,
                    bundle,
                    targets_global=train_series,
                    start=start,
                    horizon=horizon,
                    sym_loss=sym_loss,
                    lambda_symp=args.lambda_symp,
                    lambda_edges=lambda_edges,
                )
                loss.backward()
                opt.step()
                tot_loss += float(loss.detach())
                tot_data += float(ld)
                tot_sym += float(ls)
                n_chunks += 1
                continue

            sd_order = list(range(num_sd))
            random.shuffle(sd_order)
            micro_n = args.microbatch_subdomains if args.microbatch_subdomains > 0 else num_sd
            scale = float(num_sd)

            loss_mean_sum = 0.0
            ld_mean_sum = 0.0
            ls_mean_sum = 0.0
            count_sd = 0
            for batch_start in range(0, num_sd, micro_n):
                batch_sd = sd_order[batch_start : batch_start + micro_n]
                batch_loss = torch.zeros((), device=device)
                for sd_i in batch_sd:
                    li, ld, ls = train_loss_teacher_ghost_one_sd(
                        model,
                        edge_index=bundle.edge_indices[sd_i],
                        gid=bundle.global_ids[sd_i],
                        ghost=bundle.is_ghost[sd_i],
                        targets_global=train_series,
                        start=start,
                        horizon=horizon,
                        sym_loss=sym_loss,
                        lambda_symp=args.lambda_symp,
                        lambda_edges=lambda_edges,
                    )
                    batch_loss = batch_loss + li / scale
                    loss_mean_sum += float(li.detach().item())
                    ld_mean_sum += float(ld.item())
                    ls_mean_sum += float(ls.item())
                    count_sd += 1
                batch_loss.backward()

            opt.step()
            denom = float(max(1, count_sd))
            tot_loss += loss_mean_sum / denom
            tot_data += ld_mean_sum / denom
            tot_sym += ls_mean_sum / denom
            n_chunks += 1

        if n_chunks == 0:
            raise RuntimeError("No training windows.")

        n_chunks = float(n_chunks)
        mode = "joint" if args.joint_ddm_loss else "teacher-halo-accum"
        print(
            f"epoch {ep:03d} | horizon={horizon} | mode={mode} | "
            f"L={(tot_loss / n_chunks):.6f} | "
            f"L_data={(tot_data / n_chunks):.6f} | "
            f"L_sym={(tot_sym / n_chunks):.6f}"
        )

        if val_series.size(0) > 2:
            model.eval()
            vs = min(val_series.size(0) - 1, rollout_max)
            err = validation_ddm_rollout_mse(model, bundle, val_series, vs)
            print(f"          val DDM rollout MSE (mean over sd, steps={vs}): {err:.6f}")

    out_path = project_root() / "data" / "interim" / "wave_rollout_ddm_model.pth"
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
                "joint_ddm_loss": args.joint_ddm_loss,
            },
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

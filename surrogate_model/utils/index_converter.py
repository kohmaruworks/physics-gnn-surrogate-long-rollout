"""
Julia (1-based) → Python / PyTorch Geometric (0-based) conversion.

DDM extensions: global vertex ids and local patch indices from Julia are converted
with the same ``-1`` rule at the IR boundary; subgraph ``edge_index`` is validated
against ``num_local_nodes``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import warnings

import torch
from torch import Tensor

_sparse_invariant_policy_done = False


def _apply_sparse_invariant_policy_once() -> None:
    """
    PyTorch 2.x emits a UserWarning when sparse COO is created without explicitly
    choosing invariant checks. We opt out once (official way to silence it); COO
    from this loader is always `.coalesce()`'d.
    """
    global _sparse_invariant_policy_done
    if _sparse_invariant_policy_done:
        return
    _sparse_invariant_policy_done = True
    fn = getattr(torch.sparse, "set_sparse_tensor_invariants", None)
    if callable(fn):
        try:
            fn(False)
        except Exception:
            pass


class JuliaIndexError(ValueError):
    """Raised when Julia-origin indices are invalid or out of range."""


def convert_julia_to_python_indices(
    edge_index_julia: Tensor,
    *,
    num_nodes: Optional[int] = None,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    Subtract 1 from every endpoint and optionally validate against ``num_nodes``.

    Parameters
    ----------
    edge_index_julia:
        Shape ``[2, E]`` or ``[E, 2]``. Integer tensor of **1-based** endpoints.
    num_nodes:
        If given, validates ``0 <= idx_python < num_nodes`` for all entries.
    dtype:
        Output dtype (PyG typically uses ``torch.long``).

    Returns
    -------
    torch.Tensor
        Shape ``[2, E]``, **0-based**, contiguous long tensor.

    Raises
    ------
    JuliaIndexError
        Non-integer, negative Julia indices, zeros, or out-of-range values.
    """
    if edge_index_julia.dim() != 2:
        raise JuliaIndexError(f"edge_index must be 2D, got shape {tuple(edge_index_julia.shape)}")
    if edge_index_julia.shape[0] == 2:
        ei = edge_index_julia
    elif edge_index_julia.shape[1] == 2:
        ei = edge_index_julia.t().contiguous()
    else:
        raise JuliaIndexError(
            "edge_index must be [2, E] or [E, 2]; "
            f"got {tuple(edge_index_julia.shape)}"
        )

    if not torch.is_floating_point(ei):
        conv = ei.to(dtype=torch.int64)
    else:
        raise JuliaIndexError("edge_index must be integral (no floating endpoints).")

    if conv.numel() > 0:
        min_j = int(conv.min().item())
        max_j = int(conv.max().item())
        if min_j < 1:
            raise JuliaIndexError(
                f"Julia 1-based indices require >= 1; got min={min_j}"
            )
        if num_nodes is not None and max_j > num_nodes:
            raise JuliaIndexError(
                f"Julia index max={max_j} exceeds num_nodes={num_nodes}"
            )

    out = conv - 1
    if num_nodes is not None and out.numel() > 0:
        min_p = int(out.min().item())
        max_p = int(out.max().item())
        if min_p < 0 or max_p >= num_nodes:
            raise JuliaIndexError(
                f"Converted indices out of range for num_nodes={num_nodes}: "
                f"[{min_p}, {max_p}]"
            )

    return out.to(dtype=dtype).contiguous()


def julia_indices_to_python(
    x: Tensor | Sequence[int],
    *,
    upper_bound_julia: Optional[int] = None,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    Map Julia 1-based indices (vertex ids or local ids) to 0-based ``torch.long``.

    Parameters
    ----------
    x:
        1D tensor or sequence of integers ``>= 1`` (Julia convention).
    upper_bound_julia:
        If set, validates ``max(x) <= upper_bound_julia``.
    """
    if isinstance(x, Tensor):
        t = x.to(dtype=torch.int64).reshape(-1).contiguous()
    else:
        t = torch.tensor(list(x), dtype=torch.int64)
    if t.numel() == 0:
        return t.to(dtype=dtype)
    if int(t.min().item()) < 1:
        raise JuliaIndexError(f"Julia indices must be >= 1; min={int(t.min().item())}")
    if upper_bound_julia is not None and int(t.max().item()) > upper_bound_julia:
        raise JuliaIndexError(
            f"max Julia index {int(t.max().item())} exceeds bound {upper_bound_julia}"
        )
    return (t - 1).to(dtype=dtype)


def python_global_to_local_maps(
    global_ids_python: Tensor,
    *,
    num_global: int,
) -> Tuple[Dict[int, int], Tensor]:
    """
    Functor-style bookkeeping from global vertex space to positions in a patch list.

    Builds ``global_to_row[g] = row`` for rows that correspond to **unique** globals
    in ``global_ids_python`` (order-preserving first occurrence). Also returns an
    inverse view ``local_rows_to_global`` same shape as ``global_ids_python``.

    Parameters
    ----------
    global_ids_python:
        Shape ``[N_local]``, entries in ``[0, num_global)``.
    num_global:
        Total global vertex count ``N`` (Python 0-based universe size).

    Returns
    -------
    global_to_row :
        Maps global id → contiguous row index in a gathered core-only buffer (optional use).
    local_rows_to_global :
        Same shape as input; ``local_rows_to_global[r]`` is global id at patch row ``r``.
    """
    if global_ids_python.dim() != 1:
        raise JuliaIndexError(f"global_ids must be 1D, got {tuple(global_ids_python.shape)}")
    g = global_ids_python.to(dtype=torch.int64).cpu()
    if g.numel() > 0:
        if int(g.min().item()) < 0 or int(g.max().item()) >= num_global:
            raise JuliaIndexError(
                f"global_ids out of range for num_global={num_global}: "
                f"[{int(g.min().item())}, {int(g.max().item())}]"
            )
    global_to_row: Dict[int, int] = {}
    for row in range(g.numel()):
        gid = int(g[row].item())
        global_to_row.setdefault(gid, row)
    return global_to_row, global_ids_python.to(dtype=torch.int64).clone()


def ddm_edge_index_from_julia(
    edges_local_julia: Tensor,
    *,
    num_local_nodes: int,
) -> Tensor:
    """
    Convert subdomain ``edges_local`` from JSON (Julia 1-based local endpoints)
    to PyG ``[2, E]`` validated against extended patch size ``num_local_nodes``.
    """
    ei = convert_julia_to_python_indices(edges_local_julia, num_nodes=num_local_nodes)
    assert_valid_python_edge_index(ei, num_nodes=num_local_nodes)
    return ei


def assert_disjoint_core_masks(
    subdomain_core_globals: Sequence[Tensor],
    *,
    num_global: int,
) -> None:
    """Sanity check: core (non-ghost) globals partition ``0..num_global-1`` without overlap."""
    seen = torch.zeros(num_global, dtype=torch.bool)
    for gid_core in subdomain_core_globals:
        if gid_core.numel() == 0:
            continue
        g = gid_core.detach().to(dtype=torch.long).cpu().view(-1)
        if int(g.min().item()) < 0 or int(g.max().item()) >= num_global:
            raise JuliaIndexError("core global ids out of range")
        dup = seen[g]
        if dup.any():
            raise JuliaIndexError("overlapping core global ids across subdomains")
        seen[g] = True
    if not bool(seen.all()):
        raise JuliaIndexError("some global vertices are not covered by any subdomain core")


def convert_julia_sparse_coo_to_torch(
    rows_julia: Tensor,
    cols_julia: Tensor,
    values: Tensor,
    *,
    size_julia_nrows: int,
    size_julia_ncols: int,
    dtype_values: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Convert Julia-exported COO triples (1-based row/col) into a **coalesced**
    ``torch.sparse_coo_tensor`` of shape ``(nrows, ncols)`` (0-based layout).

    Parameters
    ----------
    rows_julia, cols_julia:
        1D integer tensors (length ``nnz``), ≥ 1.
    values:
        1D tensor, same length as rows (real coefficients).
    size_julia_nrows, size_julia_ncols:
        Matrix shape ``(nrows, ncols)`` — dimension count matches Julia.
    """
    rows_py = julia_indices_to_python(
        rows_julia, upper_bound_julia=size_julia_nrows, dtype=torch.long
    )
    cols_py = julia_indices_to_python(
        cols_julia, upper_bound_julia=size_julia_ncols, dtype=torch.long
    )
    v = values.reshape(-1).to(dtype=dtype_values)
    if not (rows_py.numel() == cols_py.numel() == v.numel()):
        raise JuliaIndexError("rows, cols, values must have the same nnz length")
    nrow, ncol = int(size_julia_nrows), int(size_julia_ncols)
    idx = torch.stack([rows_py, cols_py], dim=0)
    dev = device if device is not None else v.device
    _apply_sparse_invariant_policy_once()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled",
            category=UserWarning,
        )
        s = torch.sparse_coo_tensor(
            idx.to(dev),
            v.to(dev),
            size=(nrow, ncol),
            dtype=dtype_values,
            device=dev,
        ).coalesce()
    return s


def assert_valid_python_edge_index(
    edge_index: Tensor, *, num_nodes: int
) -> None:
    """Validate PyG-style ``[2, E]`` indices lie in ``[0, num_nodes)``."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise JuliaIndexError(f"Expected [2, E], got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise JuliaIndexError(f"num_nodes must be non-negative, got {num_nodes}")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
        raise JuliaIndexError(
            f"edge_index out of range for num_nodes={num_nodes}: "
            f"min={int(edge_index.min().item())}, max={int(edge_index.max().item())}"
        )

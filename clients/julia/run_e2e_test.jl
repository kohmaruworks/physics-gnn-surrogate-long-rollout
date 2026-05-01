#!/usr/bin/env julia
# End-to-end: load Julia-generated IR JSON → FastAPI `/health` → `/predict_step`.
#
# Prerequisites:
#   - Repository root JSON at `data/interim/wave_rollout_step1.json` (same layout as IR export).
#   - Trained checkpoint at `data/interim/wave_rollout_step1_model.pth` (or SURROGATE_* on server).
#
# Usage (from repo root, with API running separately):
#   julia --project=. clients/julia/run_e2e_test.jl
#
# Optional: `julia --project=. clients/julia/run_e2e_test.jl http://127.0.0.1:9000`

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

using JSON3
include(joinpath(@__DIR__, "APIClient.jl"))
using .APIClient

function wave_payload_from_disk(path::AbstractString)::Tuple{Int,Any,Any,Float64}
    raw = JSON3.read(read(path, String))
    N::Int = Int(raw.num_nodes)
    indexing = raw.indexing.edges
    indexing == "julia_1_based" || @warn("unexpected indexing.edges; expected \"julia_1_based\", got \"$indexing\"")

    u_series = raw.timeseries.u
    v_series = raw.timeseries.v
    length(u_series) ≥ 2 || throw(ArgumentError("timeseries must contain at least 2 snapshots"))
    u0 = u_series[1]
    v0 = v_series[1]
    length(u0) == N ||
        throw(ArgumentError("len(timeseries.u[1]) mismatch with num_nodes ($N)"))
    length(v0) == N ||
        throw(ArgumentError("len(timeseries.v[1]) mismatch with num_nodes ($N)"))

    node_features =
        Vector{Any}([[Float64(u0[i]), Float64(v0[i])] for i in eachindex(u0)])
    edges = Vector{Any}(raw.edges)
    dt = Float64(raw.meta.dt)
    return N, node_features, edges, dt
end

function main()
    base_url =
        isempty(ARGS) ? "http://127.0.0.1:8000" : ARGS[1]
    data_path =
        isempty(ARGS) ? joinpath(REPO_ROOT, "data", "interim", "wave_rollout_step1.json") :
            (length(ARGS) ≥ 2 ? ARGS[2] : joinpath(REPO_ROOT, "data", "interim", "wave_rollout_step1.json"))

    ispath(data_path) ||
        throw(ArgumentError("input JSON missing: $(data_path) — run data_generation/generate_wave_data.jl first"))

    N, feats, edges, dt = wave_payload_from_disk(data_path)

    client = SurrogateClient(; base_url=base_url)

    println("—— health ——")
    h = health_json(client)
    println("status: ", h.status)
    println("model_loaded: ", h.model_loaded)
    println("checkpoint: ", h.checkpoint === nothing ? "<none>" : h.checkpoint)

    dv = h.detail
    if dv !== nothing && !isempty(String(dv))
        ln = split(String(dv), '\n'; limit = 2)
        println("detail(first line): ", ln[begin])
    end

    string(h.status) != "ok" &&
        println(
            "(warning) `/health` is not fully healthy; proceeding with POST anyway for diagnostics.",)
    Bool(h.model_loaded) != true &&
        println("(warning) `model_loaded` is false — `/predict_step` may respond with HTTP 503.",)

    println()
    println("—— predict_step ——")
    println("payload: num_nodes=", N, "  state_rows=", length(feats), "  edges=", length(edges))
    println("Using dt_override from IR meta: ", dt)

    resp = predict_step_json(client; num_nodes=N, node_features=feats, edges=edges, dt_override=dt)

    next_feats = resp.node_features_next
    echo_edges = resp.edges_julia
    meta = resp.meta

    length(next_feats) == N ||
        error("unexpected node_features_next length: $(length(next_feats)), expected $(N)")
    all(r -> isa(r, AbstractVector) && length(r) == length(next_feats[1]), next_feats) ||
        error("node_features_next rows must be uniform-length vectors")

    println("received node_features_next: ", length(next_feats), " × ", length(next_feats[1]))
    println("edges_julia count: ", length(echo_edges), " (request had ", length(edges), ")")
    println("meta: ", meta)

    println()
    println("sample node 1: before = ", feats[1], "  after = ", next_feats[1])
    println("sample last node (node ", N, "): before = ", feats[end], "  after = ", next_feats[end])

    println()
    println("E2E client path completed successfully.")
    return nothing
end

main()

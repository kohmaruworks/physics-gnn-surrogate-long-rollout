#!/usr/bin/env julia
# generate_large_wave_data.jl — Higher-res grid wave + Metis domain decomposition + halo patches (DDM).
# Outputs physics_gnn_wave_rollout_ddm_v1 JSON; indices Julia 1-based.

using JSON3
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Metis

function node_id(i::Int, j::Int, nx::Int)::Int
    (j - 1) * nx + i
end

function grid_edges_directed(nx::Int, ny::Int)::Vector{Vector{Int}}
    out = Vector{Int}[]
    for j in 1:ny, i in 1:nx
        n = node_id(i, j, nx)
        if i < nx
            m = node_id(i + 1, j, nx)
            push!(out, [n, m])
            push!(out, [m, n])
        end
        if j < ny
            m = node_id(i, j + 1, nx)
            push!(out, [n, m])
            push!(out, [m, n])
        end
    end
    out
end

"""Undirected adjacency as symmetric sparse matrix (METIS input)."""
function grid_adjacency_symmetric(nx::Int, ny::Int)::SparseMatrixCSC{Float64, Int}
    N = nx * ny
    Is = Int[]
    Js = Int[]
    function add_edge(a::Int, b::Int)
        push!(Is, a)
        push!(Js, b)
        push!(Is, b)
        push!(Js, a)
    end
    for j in 1:ny, i in 1:nx
        n = node_id(i, j, nx)
        if i < nx
            add_edge(n, node_id(i + 1, j, nx))
        end
        if j < ny
            add_edge(n, node_id(i, j + 1, nx))
        end
    end
    sparse(Is, Js, ones(Float64, length(Is)), N, N)
end

function neighbor_lists(nx::Int, ny::Int)::Vector{Vector{Int}}
    N = nx * ny
    adj = [Int[] for _ in 1:N]
    for e in grid_edges_directed(nx, ny)
        a, b = e[1], e[2]
        push!(adj[a], b)
    end
    adj
end

function laplacian!(Lu::AbstractVector{Float64}, u::AbstractVector{Float64}, adj::Vector{Vector{Int}})
    N = length(u)
    @assert length(Lu) == N
    @inbounds for i in 1:N
        s = 0.0
        for j in adj[i]
            s += u[j] - u[i]
        end
        Lu[i] = s
    end
    Lu
end

function wave_rhs!(dz::Vector{Float64}, z::Vector{Float64}, p, _t::Float64)
    N = p.N
    c = p.c
    dx = p.dx
    adj = p.adj
    u = @view z[1:N]
    v = @view z[N+1:2N]
    du = @view dz[1:N]
    dv = @view dz[N+1:2N]
    @. du = v
    laplacian!(dv, u, adj)
    coeff = (c * c) / (dx * dx)
    @. dv = coeff * dv
    nothing
end

function initial_gaussian!(u::AbstractVector{Float64}, nx::Int, ny::Int, dx::Float64; sigma::Float64 = 3.0)
    cx = (nx - 1) * dx / 2
    cy = (ny - 1) * dx / 2
    for j in 1:ny, i in 1:nx
        n = node_id(i, j, nx)
        x = (i - 1) * dx
        y = (j - 1) * dx
        u[n] = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
    end
    u
end

"""Metis returns part id 1..K per global vertex."""
function metis_partition(A::SparseMatrixCSC{Float64, Int}, K::Int)::Vector{Int}
    parts_int = Metis.partition(A, K)
    [Int(parts_int[i]) for i in 1:length(parts_int)]
end

"""Build extended patch for Metis part `p` (1-based): core ∪ 1-hop halo from other parts."""
function build_subdomain_patch(
    p::Int,
    parts::Vector{Int},
    adj::Vector{Vector{Int}},
    directed_edges::Vector{Vector{Int}},
)::NamedTuple
    N = length(parts)
    core_vs = [v for v in 1:N if parts[v] == p]
    core_set = Set(core_vs)
    halo_vs = Set{Int}()
    for v in core_vs
        for nb in adj[v]
            if parts[nb] != p
                push!(halo_vs, nb)
            end
        end
    end
    ordered = sort(collect(core_set))
    append!(ordered, sort(collect(halo_vs)))
    local_of = Dict(g => i for (i, g) in enumerate(ordered))
    edges_local = Vector{Vector{Int}}()
    for e in directed_edges
        a, b = e[1], e[2]
        la = get(local_of, a, nothing)
        lb = get(local_of, b, nothing)
        if la !== nothing && lb !== nothing
            push!(edges_local, [la, lb])
        end
    end
    nodes = Vector{Dict{String, Any}}()
    for g in ordered
        lid = local_of[g]
        is_g = !(g in core_set)
        owner = is_g ? parts[g] : nothing
        push!(
            nodes,
            Dict{String, Any}(
                "global_id" => g,
                "local_id" => lid,
                "is_ghost" => is_g,
                "owner_subdomain" => owner === nothing ? nothing : owner,
            ),
        )
    end
    (
        subdomain_id = p,
        num_local_nodes = length(ordered),
        nodes = nodes,
        edges_local = edges_local,
    )
end

function main(;
    nx::Int = 32,
    ny::Int = 32,
    dx::Float64 = 1.0,
    c::Float64 = 1.0,
    dt_save::Float64 = 0.05,
    tspan::Tuple{Float64, Float64} = (0.0, 2.0),
    abstol::Float64 = 1e-9,
    reltol::Float64 = 1e-9,
    num_parts::Int = 4,
    out_path::String = joinpath(@__DIR__, "..", "data", "interim", "wave_rollout_ddm_v1.json"),
)
    N = nx * ny
    K = num_parts
    if K < 2
        error("num_parts must be >= 2")
    end

    adj = neighbor_lists(nx, ny)
    dir_edges = grid_edges_directed(nx, ny)

    A = grid_adjacency_symmetric(nx, ny)
    parts = metis_partition(A, K)

    subdomains = [build_subdomain_patch(p, parts, adj, dir_edges) for p in 1:K]

    z0 = zeros(2N)
    u0 = @view z0[1:N]
    v0 = @view z0[N+1:2N]
    initial_gaussian!(u0, nx, ny, dx)
    fill!(v0, 0.0)

    p_ode = (; N, c, dx, adj)
    prob = ODEProblem(wave_rhs!, z0, tspan, p_ode)
    sol = solve(prob, Tsit5(); abstol = abstol, reltol = reltol, saveat = dt_save)

    nt = length(sol.t)
    u_series = Vector{Vector{Float64}}()
    v_series = Vector{Vector{Float64}}()
    for k in 1:nt
        z = sol.u[k]
        push!(u_series, collect(z[1:N]))
        push!(v_series, collect(z[N+1:2N]))
    end

    payload = Dict{String, Any}(
        "schema" => "physics_gnn_wave_rollout_ddm_v1",
        "indexing" => Dict(
            "edges_local" => "julia_1_based_within_extended_patch",
            "global_node_ids" => "julia_1_based",
            "local_indices" => "julia_1_based_per_subdomain_node_list",
        ),
        "meta" => Dict(
            "nx" => nx,
            "ny" => ny,
            "dt" => dt_save,
            "c" => c,
            "dx" => dx,
            "num_steps" => nt,
            "num_parts" => K,
            "description" => "DDM via Metis + 1-hop halo; discrete Hamiltonian wave reference.",
        ),
        "global" => Dict{String, Any}(
            "num_nodes" => N,
            "edges" => dir_edges,
        ),
        "subdomains" => [
            Dict{String, Any}(
                "subdomain_id" => sd.subdomain_id,
                "num_local_nodes" => sd.num_local_nodes,
                "nodes" => sd.nodes,
                "edges_local" => sd.edges_local,
            ) for sd in subdomains
        ],
        "timeseries" => Dict("u" => u_series, "v" => v_series),
    )

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON3.write(io, payload)
    end
    println("Wrote: ", abspath(out_path))
    nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

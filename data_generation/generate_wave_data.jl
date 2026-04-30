#!/usr/bin/env julia
# generate_wave_data.jl — 2D discrete wave (Hamiltonian) + JSON export per schema.
# Edge indices are Julia 1-based; Python uses convert_julia_to_python_indices.

using JSON3
using DifferentialEquations
using LinearAlgebra

function node_id(i::Int, j::Int, nx::Int)::Int
    (j - 1) * nx + i
end

"""Build directed 4-neighbor edges (both directions) on an nx×ny grid; endpoints 1-based."""
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

function node_positions(nx::Int, ny::Int, dx::Float64)::Vector{Vector{Float64}}
    pos = Vector{Vector{Float64}}()
    for j in 1:ny, i in 1:nx
        push!(pos, [(i - 1) * dx, (j - 1) * dx])
    end
    pos
end

"""Neighbor lists (1-based) for Laplacian via Σ_j (u_j - u_i)."""
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

"""
Semi-discrete wave: u′ = v, v′ = c² * ∇²_h u / dx² with graph Laplacian
(Σ_{j∼i} (u_j - u_i)) / dx².
"""
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

function initial_gaussian!(u::AbstractVector{Float64}, nx::Int, ny::Int, dx::Float64; sigma::Float64 = 2.5)
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

"""
    main(; kwargs...) -> Nothing

Generate `data/interim/wave_rollout_step1.json` by default.
"""
function main(;
    nx::Int = 16,
    ny::Int = 16,
    dx::Float64 = 1.0,
    c::Float64 = 1.0,
    dt_save::Float64 = 0.05,
    tspan::Tuple{Float64, Float64} = (0.0, 4.0),
    abstol::Float64 = 1e-9,
    reltol::Float64 = 1e-9,
    out_path::String = joinpath(@__DIR__, "..", "data", "interim", "wave_rollout_step1.json"),
)
    N = nx * ny
    adj = neighbor_lists(nx, ny)
    edges = grid_edges_directed(nx, ny)
    pos = node_positions(nx, ny, dx)

    z0 = zeros(2N)
    u0 = @view z0[1:N]
    v0 = @view z0[N+1:2N]
    initial_gaussian!(u0, nx, ny, dx)
    fill!(v0, 0.0)

    p = (; N, c, dx, adj)
    prob = ODEProblem(wave_rhs!, z0, tspan, p)
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
        "schema" => "physics_gnn_wave_rollout_step1_v1",
        "indexing" => Dict("edges" => "julia_1_based"),
        "meta" => Dict(
            "nx" => nx,
            "ny" => ny,
            "dt" => dt_save,
            "c" => c,
            "dx" => dx,
            "num_steps" => nt,
            "description" => "2D discrete Hamiltonian wave on grid; Tsit5 reference trajectory.",
        ),
        "num_nodes" => N,
        "edges" => edges,
        "node_positions" => pos,
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

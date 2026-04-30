#!/usr/bin/env julia
# generate_eval_data.jl — Zero-shot style eval mesh (non-square, asymmetric IC) + wave GT + wall-clock timing for ROI baseline.
# Schema: physics_gnn_eval_v1. Indices Julia 1-based. Optional 2:1 multigrid operators when nf_x, nf_y even.

import Pkg
Pkg.activate(dirname(@__DIR__))

using JSON3
using DifferentialEquations
using LinearAlgebra

function fine_node_id(i::Int, j::Int, nx::Int)::Int
    (j - 1) * nx + i
end

function coarse_node_id(I::Int, J::Int, ncx::Int)::Int
    (J - 1) * ncx + I
end

function grid_edges_directed(nx::Int, ny::Int)::Vector{Vector{Int}}
    out = Vector{Int}[]
    for j in 1:ny, i in 1:nx
        n = fine_node_id(i, j, nx)
        if i < nx
            m = fine_node_id(i + 1, j, nx)
            push!(out, [n, m])
            push!(out, [m, n])
        end
        if j < ny
            m = fine_node_id(i, j + 1, nx)
            push!(out, [n, m])
            push!(out, [m, n])
        end
    end
    out
end

function neighbor_lists(nx::Int, ny::Int)::Vector{Vector{Int}}
    N = nx * ny
    adj = [Int[] for _ in 1:N]
    for e in grid_edges_directed(nx, ny)
        push!(adj[e[1]], e[2])
    end
    adj
end

function laplacian!(Lu::AbstractVector{Float64}, u::AbstractVector{Float64}, adj::Vector{Vector{Int}})
    @inbounds for i in eachindex(u)
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

"""Asymmetric Gaussian (offset peak) — distinct from typical centered training ICs."""
function initial_gaussian_asymmetric!(u::AbstractVector{Float64}, nx::Int, ny::Int, dx::Float64)
    cx = (nx - 1) * dx * 0.33
    cy = (ny - 1) * dx * 0.61
    sigma = 2.8
    for j in 1:ny, i in 1:nx
        n = fine_node_id(i, j, nx)
        x = (i - 1) * dx
        y = (j - 1) * dx
        u[n] = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
    end
    u
end

function build_restriction_coo(nf_x::Int, nf_y::Int, nc_x::Int, nc_y::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    @assert nf_x == 2 * nc_x && nf_y == 2 * nc_y
    for J in 1:nc_y, I in 1:nc_x
        c = coarse_node_id(I, J, nc_x)
        fi0 = 2 * I - 1
        fj0 = 2 * J - 1
        for dj in 0:1, di in 0:1
            f = fine_node_id(fi0 + di, fj0 + dj, nf_x)
            push!(rows, c)
            push!(cols, f)
            push!(vals, 0.25)
        end
    end
    (rows = rows, cols = cols, vals = vals)
end

function build_prolongation_coo(nf_x::Int, nf_y::Int, nc_x::Int, nc_y::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    @assert nf_x == 2 * nc_x && nf_y == 2 * nc_y
    for J in 1:nc_y, I in 1:nc_x
        c = coarse_node_id(I, J, nc_x)
        fi0 = 2 * I - 1
        fj0 = 2 * J - 1
        for dj in 0:1, di in 0:1
            f = fine_node_id(fi0 + di, fj0 + dj, nf_x)
            push!(rows, f)
            push!(cols, c)
            push!(vals, 1.0)
        end
    end
    (rows = rows, cols = cols, vals = vals)
end

function main(;
    nf_x::Int = 44,
    nf_y::Int = 32,
    dx::Float64 = 1.0,
    c::Float64 = 1.0,
    dt_save::Float64 = 0.05,
    tspan::Tuple{Float64, Float64} = (0.0, 2.5),
    abstol::Float64 = 1e-9,
    reltol::Float64 = 1e-9,
    out_path::String = joinpath(@__DIR__, "..", "data", "interim", "eval_zero_shot_v1.json"),
)
    if nf_x % 2 != 0 || nf_y % 2 != 0
        error("nf_x and nf_y must be even for embedded 2:1 multigrid operators in eval IR")
    end
    nc_x = div(nf_x, 2)
    nc_y = div(nf_y, 2)
    Nf = nf_x * nf_y
    Nc = nc_x * nc_y

    adj = neighbor_lists(nf_x, nf_y)
    fine_edges = grid_edges_directed(nf_x, nf_y)
    coarse_edges = grid_edges_directed(nc_x, nc_y)
    Rcoo = build_restriction_coo(nf_x, nf_y, nc_x, nc_y)
    Pcoo = build_prolongation_coo(nf_x, nf_y, nc_x, nc_y)

    z0 = zeros(2Nf)
    initial_gaussian_asymmetric!(@view(z0[1:Nf]), nf_x, nf_y, dx)
    fill!(@view(z0[Nf+1:2Nf]), 0.0)

    prob = ODEProblem(wave_rhs!, z0, tspan, (; N = Nf, c, dx, adj))

    elapsed_total = @elapsed begin
        sol = solve(prob, Tsit5(); abstol = abstol, reltol = reltol, saveat = dt_save)
    end

    nt = length(sol.t)
    per_macro = elapsed_total / max(1, nt - 1)

    u_series = Vector{Vector{Float64}}()
    v_series = Vector{Vector{Float64}}()
    for k in 1:nt
        z = sol.u[k]
        push!(u_series, collect(z[1:Nf]))
        push!(v_series, collect(z[Nf+1:2Nf]))
    end

    payload = Dict{String, Any}(
        "schema" => "physics_gnn_eval_v1",
        "meta" => Dict(
            "nf_x" => nf_x,
            "nf_y" => nf_y,
            "nc_x" => nc_x,
            "nc_y" => nc_y,
            "dt" => dt_save,
            "c" => c,
            "dx" => dx,
            "num_macro_steps" => nt,
            "julia_total_solve_seconds" => elapsed_total,
            "julia_seconds_per_macro_step" => per_macro,
            "description" => "Zero-shot eval: rectangular grid + asymmetric IC; timing is wall-clock for full Tsit5 solve with saveat.",
        ),
        "fine_graph" => Dict("num_nodes" => Nf, "edges" => fine_edges),
        "coarse_graph" => Dict("num_nodes" => Nc, "edges" => coarse_edges),
        "restriction" => Dict(
            "nrows" => Nc,
            "ncols" => Nf,
            "rows" => Rcoo.rows,
            "cols" => Rcoo.cols,
            "values" => Rcoo.vals,
        ),
        "prolongation" => Dict(
            "nrows" => Nf,
            "ncols" => Nc,
            "rows" => Pcoo.rows,
            "cols" => Pcoo.cols,
            "values" => Pcoo.vals,
        ),
        "timeseries" => Dict("u" => u_series, "v" => v_series),
    )

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON3.write(io, payload)
    end
    println("Wrote: ", abspath(out_path))
    println("julia_total_solve_seconds=", elapsed_total, "  per_macro_step≈", per_macro)
    nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

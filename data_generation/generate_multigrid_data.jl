#!/usr/bin/env julia
# generate_multigrid_data.jl — Fine/coarse 2:1 tensor-product grids, Galerkin-style R and piecewise-constant P (COO, Julia 1-based).
# Reference trajectory on the fine grid only (wave equation). Schema: physics_gnn_multigrid_v1.

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

function initial_gaussian!(u::AbstractVector{Float64}, nx::Int, ny::Int, dx::Float64; sigma::Float64 = 2.5)
    cx = (nx - 1) * dx / 2
    cy = (ny - 1) * dx / 2
    for j in 1:ny, i in 1:nx
        n = fine_node_id(i, j, nx)
        x = (i - 1) * dx
        y = (j - 1) * dx
        u[n] = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
    end
    u
end

"""Full-weight averaging restriction: each coarse cell averages four fine vertices."""
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

"""Piecewise-constant prolongation: each fine inherits its coarse cell value."""
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
    nf_x::Int = 16,
    nf_y::Int = 16,
    dx::Float64 = 1.0,
    c::Float64 = 1.0,
    dt_save::Float64 = 0.05,
    tspan::Tuple{Float64, Float64} = (0.0, 2.0),
    abstol::Float64 = 1e-9,
    reltol::Float64 = 1e-9,
    out_path::String = joinpath(@__DIR__, "..", "data", "interim", "multigrid_wave_v1.json"),
)
    if nf_x % 2 != 0 || nf_y % 2 != 0
        error("nf_x and nf_y must be even for 2:1 coarsening")
    end
    nc_x = div(nf_x, 2)
    nc_y = div(nf_y, 2)

    Nf = nf_x * nf_y
    Nc = nc_x * nc_y

    fine_edges = grid_edges_directed(nf_x, nf_y)
    coarse_edges = grid_edges_directed(nc_x, nc_y)

    Rcoo = build_restriction_coo(nf_x, nf_y, nc_x, nc_y)
    Pcoo = build_prolongation_coo(nf_x, nf_y, nc_x, nc_y)

    adj = neighbor_lists(nf_x, nf_y)
    z0 = zeros(2Nf)
    initial_gaussian!(@view(z0[1:Nf]), nf_x, nf_y, dx)
    fill!(@view(z0[Nf+1:2Nf]), 0.0)

    prob = ODEProblem(wave_rhs!, z0, tspan, (; N = Nf, c, dx, adj))
    sol = solve(prob, Tsit5(); abstol = abstol, reltol = reltol, saveat = dt_save)

    nt = length(sol.t)
    u_series = Vector{Vector{Float64}}()
    v_series = Vector{Vector{Float64}}()
    for k in 1:nt
        z = sol.u[k]
        push!(u_series, collect(z[1:Nf]))
        push!(v_series, collect(z[Nf+1:2Nf]))
    end

    payload = Dict{String, Any}(
        "schema" => "physics_gnn_multigrid_v1",
        "indexing" => Dict(
            "edges" => "julia_1_based",
            "sparse_coo" => "julia_1_based_rows_cols",
        ),
        "meta" => Dict(
            "nf_x" => nf_x,
            "nf_y" => nf_y,
            "nc_x" => nc_x,
            "nc_y" => nc_y,
            "dt" => dt_save,
            "c" => c,
            "dx" => dx,
            "num_steps" => nt,
            "description" => "2:1 structured multigrid; full-weight R and piecewise-constant P.",
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
    nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

"""
HTTP client for the Python FastAPI surrogate inference service.

Convention: payloads use **Julia-style 1-based** edge endpoints as stored in exported JSON IR.
This module does **not** apply −1/+1 tweaks; indices are forwarded verbatim to `/predict_step` and echoed back.

See `api/schemas.py` for the authoritative contract.
"""
module APIClient

using HTTP
using JSON3

export SurrogateClient, health_json, predict_step_json

mutable struct SurrogateClient
    base_url::String
end

"""Construct a client; `base_url` defaults to ``http://127.0.0.1:8000``."""
function SurrogateClient(; base_url::AbstractString="http://127.0.0.1:8000")
    u = String(rstrip(base_url))
    isempty(u) && throw(ArgumentError("base_url must be non-empty"))
    SurrogateClient(u)
end

function _joined(c::SurrogateClient, path::AbstractString)
    p = startswith(path, "/") ? path : "/" * path
    string(c.base_url, p)
end

function _require_ok(resp::HTTP.Messages.Response)::Nothing
    if resp.status < 300
        return nothing
    end
    bod = isempty(resp.body) ? "" : String(resp.body)
    error("HTTP $(resp.status): $(bod)")
end

"""GET `/health`; returns parsed JSON (`JSON3.Object` or readable via property access)."""
function health_json(c::SurrogateClient)
    uri = _joined(c, "/health")
    resp = HTTP.get(uri)
    _require_ok(resp)
    JSON3.read(String(resp.body))
end

"""POST `/predict_step` with Julia-style payloads; returns parsed JSON."""
function predict_step_json(
    c::SurrogateClient;
    num_nodes::Int,
    node_features::Any,
    edges::Any,
    dt_override::Union{Nothing,Float64}=nothing,
)::JSON3.Object
    num_nodes ≥ 1 || throw(ArgumentError("num_nodes must be ≥ 1"))
    body_dict = Dict{String,Any}(
        "num_nodes" => num_nodes,
        "node_features" => node_features,
        "edges" => edges,
    )
    if dt_override !== nothing
        body_dict["dt_override"] = Float64(dt_override)
    end
    uri = _joined(c, "/predict_step")
    resp = HTTP.post(
        uri;
        headers=["Content-Type" => "application/json"],
        body=JSON3.write(body_dict),
    )
    _require_ok(resp)
    JSON3.read(String(resp.body))
end

end # module APIClient


struct UGCNConv{W <: AbstractMatrix, B, F} <: GNNLayer
    weight::W
    bias::B
    σ::F
    add_self_loops::Bool
    use_edge_weight::Bool
end

Flux.@functor UGCNConv

function UGCNConv(ch::Pair{Int, Int}, σ = identity;
                 init = glorot_uniform,
		 aggr=mean,
                 bias::Bool = true,
                 add_self_loops = true,
                 use_edge_weight = false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    UGCNConv(W, b, σ, add_self_loops, use_edge_weight)
end

check_ugcnconv_input(g::GNNGraph{<:GraphNeuralNetworks.ADJMAT_T}, edge_weight::AbstractVector) = 
    throw(ArgumentError("Providing external edge_weight is not yet supported for adjacency matrix graphs"))

function check_ugcnconv_input(g::GNNGraph, edge_weight::AbstractVector)
    if length(edge_weight) !== g.num_edges 
        throw(ArgumentError("Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"))
    end
end

check_ugcnconv_input(g::GNNGraph, edge_weight::Nothing) = nothing



function (l::UGCNConv)(g::GNNGraph, 
                      x::AbstractMatrix{T},
                      edge_weight::EW = nothing
                      ) where {T, EW <: Union{Nothing, AbstractVector}}

    check_ugcnconv_input(g, edge_weight)

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            # Pad weights with ones
            # TODO for ADJMAT_T the new edges are not generally at the end
            edge_weight = [edge_weight; fill!(similar(edge_weight, g.num_nodes), 1)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if Dout < Din
        # multiply before convolution if it is more convenient, otherwise multiply after
        x = l.weight * x
    end
    if edge_weight !== nothing
        d = GraphNeuralNetworks.degree(g, T; dir = :in, edge_weight)
    else
        d = GraphNeuralNetworks.degree(g, T; dir = :in, edge_weight = l.use_edge_weight)
    end
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +, xj = x, e = edge_weight)
    elseif l.use_edge_weight
        x = propagate(w_mul_xj, g, +, xj = x)
    else
        x = propagate(copy_xj, g, +, xj = x)
    end
    
    if Dout >= Din
        x = l.weight * x
    end
    return l.σ.(x .+ l.bias)
end

function (l::UGCNConv)(g::GNNGraph{<:GraphNeuralNetworks.ADJMAT_T}, x::AbstractMatrix,
                      edge_weight::AbstractVector)
    g = GNNGraph(edge_index(g)...; g.num_nodes)  # convert to COO
    return l(g, x, edge_weight)
end

function Base.show(io::IO, l::UGCNConv)
    out, in = size(l.weight)
    print(io, "UGCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


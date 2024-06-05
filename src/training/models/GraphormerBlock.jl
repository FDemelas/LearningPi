"""
Structure that implement the basic main machine-learning block of this .

# Fields:
- `Convolution`: a Graph Convolutional Neural Network that performs one graph-message passing,
- `MLP`: a Multi-Layer-Perceptron that implement the non-linear part in parallel over all the node-hidden-features. 

The first constructor takes as input the following

# Arguments:
- `hidden_sample`: a structure composed by three boolean fields to handle the sampling positions in the  model,
- `inpOut`: the size  of the hidden space, 
- `init`: initialization for the parameters of the models,
- `rng`: random number generator for the sampler, dropout, and all the other random components of the model,
- `pDrop`: dropout probability,
- `h_MLP`: a vector containing at each component the number of nodes in the associated layer of the hidden multi-layer-perceptron,
- `ConvLayer`: convolutional layer, by default is `GraphConv`,
- `act`: activation function for the Multi-Layer-Perceptron, by default is `relu`,
- `act_conv`: activation function for the Graph Convolutional Part, by default is  `identity`,
- `aggr`: aggregation function for the Graph Convolutional Part, by default is `mean`.

The second constructor directly takes as input the Fields of the structure.
"""
struct GraphormerBlock <: AbstractModel
	Convolution::Any
	MLP::Any
	GraphormerBlock(hidden_sample, inpOut::Int, init, rng, pDrop, h_MLP::Vector{Int}, ConvLayer = GraphConv, act::Any = relu, act_conv = identity, aggr = mean) =
		(
			r1 = LayerNorm(1.0f-5, inpOut);
			r2 = LayerNorm(1.0f-5, inpOut);

			hMLP = cat([inpOut], h_MLP, dims = 1);
			layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)];
			Convolution = GNNChain(Parallel(+, GNNChain(r1, ConvLayer(inpOut => inpOut, act_conv, ; aggr, init), Dropout(pDrop; rng)), identity));

			MLP = Parallel(+, Chain(r2, layersMLP..., Dense(hMLP[end] => inpOut, act; init = init), Dropout(pDrop; rng)), identity);
			MLP = hidden_sample ? Chain(MLP, Dense(inpOut => 2 * inpOut; init)) : MLP;
			new(Convolution, MLP)
		)
	GraphormerBlock(Convolution, MLP) = (new(Convolution, MLP))
end

"""
# Arguments:
- `x`: a `GNNGraph`,
- `h`: a features matrix associated to the nodes of `x`.

Computes the forward for the `GraphormerBlock`.
The backward is automatically computed as long as all the operation in the forward are differentiable by `Zygote`.
"""
function (m::GraphormerBlock)(x, h)
	h = m.Convolution(x, h)
	return m.MLP(h)
end

# declare the GraphormerBlock a Flux functor
Flux.@functor GraphormerBlock
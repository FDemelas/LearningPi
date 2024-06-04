"""
# Arguments:
- `where_sample`: a structure composed by three booleans to say where perform sampling, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `nBlocks`: number of main-blocks that compose the core part of the encoder,
- `nNodes`: dimension of the hidden-space for the features representation,
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `act`: activation function for the parallel MLP, by default `relu`,
- `act_conv`: activation function for the Graph-Convolutional Layers, by default `relu`,
- `seed`: random generation seed, by default `1`, 
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default `[100,250,500]`,
- `hH`:   a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, by default `[500]` ,
- `hF`:  a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `pDrop`: drop-out parameter, by default  `0.001`,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default `0.00001`,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `aggr`: the aggregation function, by default `mean`,
- `prediction_layers`: a vector that contains the indexes of the layers in which we want perform a prediction, by default `[]`, in this case we use the decoder only in the last main-block of the Graphormer.

returns a model as defined in `Graphormer.jl` using the provided hyper-parameters.
"""
function create_model(
	where_sample,
	in,
	nBlocks,
	nNodes,
	out = 1,
	act = relu,
	act_conv = relu,
	seed = 1,
	hI = [100, 250, 500],
	hH = [500],
	hF = [500, 250, 100],
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.00001,
	norm = true,
	aggr = mean,
	prediction_layers = [],
)
	CLayer = norm ? GCNConv : UGCNConv

	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

	hI = cat([in], hI, dims = 1)
	layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] => nNodes, act; init = init)))

	Graphormers = GraphormerBlock[]
	for _ in 1:nBlocks
		push!(Graphormers, model_device(GraphormerBlock(where_sample.hidden_state, nNodes, init, rng, pDrop, hH, CLayer, act, act_conv, aggr)))
	end

	Graphormers = GNNChain(Graphormers...)

	if isempty(prediction_layers)
		prediction_layers = [nBlocks]
	end

	Decoders = []
	for i in prediction_layers
		inpL = where_sample.before_decoding ? Int64(nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
		outL = where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
	Decoders = Chain(Decoders...)

	Sampling = Sampler(rng)
	m = Graphormer(HiddenMap, Graphormers, Decoders, Sampling, where_sample, prediction_layers, dt)
	return m
end

"""
# Arguments:
- `where_sample`: a structure composed by three booleans to say where perform sampling, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `nBlocks`: number of main-blocks that compose the core part of the encoder,
- `nNodes`: dimension of the hidden-space for the features representation,
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `act`: activation function for the parallel MLP, by default `relu`,
- `act_conv`: activation function for the Graph-Convolutional Layers, by default `relu`,
- `seed`: random generation seed, by default `1`, 
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default `[100,250,500]`,
- `hH`:   a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, by default `[500]` ,
- `hF`:  a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `pDrop`: drop-out parameter, by default  `0.001` (unused in this implementation, will be removed soon),
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default `0.00001`,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `aggr`: the aggregation function, by default `mean`,
- `prediction_layers`: a vector that contains the indexes of the layers in which we want perform a prediction, by default `[]`, in this case we use the decoder only in the last main-block of the Graphormer.

returns a model as defined in `Graphormer.jl` using the provided hyper-parameters.
"""
function create_model_gasse(
	where_sample,
	in,
	nBlocks,
	nNodes,
	out = 1,
	act = relu,
	act_conv = relu,
	seed = 1,
	hI = [100, 250, 500],
	hH = [500],
	hF = [500, 250, 100],
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.00001,
	norm = true,
	aggr = mean,
	prediction_layers = [],
)
	ConvLayer = norm ? GCNConv : UGCNConv

	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

	hI = cat([in], hI, dims = 1)
	layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] => nNodes, act; init = init)))

	Graphormers = GraphormerBlock[]
	for _ in 1:nBlocks
		hMLP = cat([nNodes], hH, dims = 1)
		layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)]
		Convolution = ConvLayer(nNodes => nNodes, act_conv, ; aggr, init)
		MLP = identity
		push!(Graphormers, model_device(GraphormerBlock(Convolution, MLP)))
	end

	Graphormers = GNNChain(Graphormers...)

	if isempty(prediction_layers)
		prediction_layers = [nBlocks]
	end

	Decoders = []
	for i in prediction_layers
		inpL = where_sample.before_decoding ? Int64(nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
		outL = where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
	Decoders = Chain(Decoders...)

	Sampling = Sampler(rng)
	m = Graphormer(HiddenMap, Graphormers, Decoders, Sampling, where_sample, prediction_layers, dt)
	return m
end

"""
# Arguments:
- `where_sample`: a structure composed by three booleans to say where perform sampling, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `nBlocks`: number of main-blocks that compose the core part of the encoder,
- `nNodes`: dimension of the hidden-space for the features representation,
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `act`: activation function for the parallel MLP, by default `relu`,
- `act_conv`: activation function for the Graph-Convolutional Layers, by default `relu` (unused in this implementation, will be removed soon),
- `seed`: random generation seed, by default `1`, 
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default `[100,250,500]`,
- `hH`:   a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, by default `[500]` ,
- `hF`:  a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `pDrop`: drop-out parameter, by default  `0.001` (unused in this implementation, will be removed soon),
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default `0.00001`,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `aggr`: the aggregation function, by default `mean`,
- `prediction_layers`: a vector that contains the indexes of the layers in which we want perform a prediction, by default `[]`, in this case we use the decoder only in the last main-block of the Graphormer.

returns a model as defined in `Graphormer.jl` using the provided hyper-parameters.
"""
function create_model_nair(
	where_sample,
	in,
	nBlocks,
	nNodes,
	out = 1,
	act = relu,
	act_conv = relu,
	seed = 1,
	hI = [100, 250, 500],
	hH = [500],
	hF = [500, 250, 100],
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.00001,
	norm = true,
	aggr = mean,
	prediction_layers = [],
)
	ConvLayer = norm ? GCNConv : UGCNConv

	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

	hI = cat([in], hI, dims = 1)
	layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] => nNodes, act; init = init)))

	r1 = LayerNorm(1.0f-5, nNodes)
	Graphormers = GraphormerBlock[]
	for _ in 1:nBlocks
		hMLP = cat([nNodes], hH, dims = 1)
		layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)]
		Convolution = GNNChain(Parallel(+, GNNChain(Chain(layersMLP..., Dense(hMLP[end] => nNodes, act; init = init)), ConvLayer(nNodes => nNodes; aggr, init), r1), identity))

		MLP = identity

		push!(Graphormers, model_device(GraphormerBlock(Convolution, MLP)))
	end

	Graphormers = GNNChain(Graphormers...)

	if isempty(prediction_layers)
		prediction_layers = [nBlocks]
	end

	Decoders = []
	for i in prediction_layers
		inpL = where_sample.before_decoding ? Int64(nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
		outL = where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
	Decoders = Chain(Decoders...)

	Sampling = Sampler(rng)
	m = Graphormer(HiddenMap, Graphormers, Decoders, Sampling, where_sample, prediction_layers, dt)
	return m
end

"""
Struct to easily construct a neural network architecture inspired from:

Nair, V., Bartunov, S., Gimeno, F., von Glehn, I., Lichocki, P., Lobov, I., O’Donoghue, B., Sonnerat, N., Tjandraatmadja, C., Wang, P., Addanki, R., Hapuarachchi, T., Keck, T., Keeling, J., Kohli, P., Ktena, I., Li, Y., Vinyals, O., and Zwols, Y. Solving mixed integer programs using neural networks. CoRR, abs/2012.13349, 2020.

Subtype of `learningBlockGNN`.
"""
struct learningSampleNair <: learningBlockGNN end


"""
Struct to easily construct a neural network architecture inspired from:

Gasse, M., Chételat, D., Ferroni, N., Charlin, L., and Lodi, A. Exact Combinatorial Optimization with Graph Convolutional Neural Networks. In Wallach, H., Larochelle, H., Beygelzimer, A., Alché-Buc, F. d., Fox, E., and Garnett,R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Subtype of `learningBlockGNN`.
"""
struct learningSampleGasse <: learningBlockGNN end

"""
Struct to easily construct a neural network architecture presented in:

F. Demelas, J. Le Roux, M. Lacroix, A. Parmentier "Predicting Lagrangian Multipliers for Mixed Integer Linear Programs", ICML 2024.

Subtype of `learningBlockGNN`.
"""
struct learningSampleTransformer <: learningBlockGNN end

"""
Struct to easily construct a neural network architecture similar to `learningSampleTransformer`, but that perform instead the sampling in the lagrangian multipliers output space.
"""
struct learningSampleOutside <: learningBlockGNN end

"""
Struct to easily construct a neural network architecture similar to `learningSampleTransformer`, but it does not perform sampling at all.
"""
struct learningTransformer <: learningBlockGNN end

"""
Struct to easily construct a neural network architecture similar to `learningMultiPresSample`, but it does not perform sampling at all.
"""
struct learningMultiPredTransformer <: learningBlockGNN end

"""
Struct to easily construct a neural network architecture similar to `learningSampleTransformer`, but predict multiple deviation using different decoders enbedded at the end of given main-blocks.
"""
struct learningMultiPredSample <: learningBlockGNN end

"""
# Arguments:
- `lType`: learning type, should be `learningMultiPredSample`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningMultiPredSample` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningMultiPredSample,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.00001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, true)
	return create_model(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, collect(1:block_number))
end


"""
# Arguments:
- `lType`: learning type, should be `learningMultiPredTransformer`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningMultiPredTransformer` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningMultiPredTransformer,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.00001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, false)
	return create_model(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, collect(1:block_number))
end

"""
# Arguments:
- `lType`: learning type, should be `learningSampleTransformer`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningSampleTransformer` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningSampleTransformer,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.0001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, true)
	return create_model(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, [])
end

"""
# Arguments:
- `lType`: learning type, should be `learningSampleGasse`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningSampleGasse` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningSampleGasse,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.0001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, true)
	return create_model_gasse(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, [])
end


"""
# Arguments:
- `lType`: learning type, should be `learningSampleNair`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningSampleNair` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningSampleNair,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.0001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, true)
	return create_model_nair(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, [])
end

"""
# Arguments:
- `lType`: learning type, should be `learningTransformer`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningTransformer` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningTransformer,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.0001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(false, false, false)
	return create_model(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, [])
end

"""
# Arguments:
- `lType`: learning type, should be `learningSampleOutside`, 
- `in`: size of the input of the neural network, for each node in the bipartite graph-representation, 
- `h`:  a vector containing the number of nodes in the hidden layers that composes the  MLP inside the main-blocks of the Encoder, 
- `out`: the dimention of the output of the neural network model, for each dualized constraint, by default 1,
- `a`: activation function, by default `relu`,
- `seed`: random generation seed, by default `1`,
- `hI`: a vector containing the number of nodes in the hidden layers that composes the initial MLP that send the features into the hidden space representation, by default  `[500, 250, 100]`,
- `hF`: a vector containing the number of nodes in the hidden layers that composes the final MLP in the Decoder, by default `[500, 250, 100]`,
- `block_number`: number of main-blocks that compose the core part of the encoder, by default `5`,
- `nodes_number`: dimension of the hidden-space for the features representation, by default `500`,
- `pDrop`: drop-out parameter, by default 0.001,
- `dt` : deviation type, by default  cr_deviation(),
- `std`: standard deviation used for the initialization of the nn parameters, by default 0.00001,
- `norm`: a boolean to say if normalize or not during the GNN message passing, by default true;
- `final_A`: final activation function (in the space of Lagrangian multipliers, but before deviation), by default identity

Returns the neural network model for `learningSampleOutside` and the other provided hyper-parameters.
"""
function create_model(
	lType::learningSampleOutside,
	in,
	h,
	out = 1,
	a = relu,
	seed = 1,
	hI = [500, 250, 100],
	hF = [500, 250, 100],
	block_number::Int64 = 5,
	nodes_number::Int64 = 500,
	pDrop = 0.001,
	dt::abstract_deviation = cr_deviation(),
	std = 0.0001,
	norm = true;
	final_A = identity,
)
	where_sample = SamplingPosition(true, false, false)
	return create_model(where_sample, in, block_number, nodes_number, out, a, a, seed, hI, h, hF, pDrop, dt, std, norm, mean, [])
end

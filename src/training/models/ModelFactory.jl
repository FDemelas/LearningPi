function create_model(where_sample, in, nBlocks,nNodes, out = 1, act = relu,act_conv=relu, seed = 1, hI = [100,250,500], hH = [500] , hF = [500, 250, 100],  pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true,aggr=mean,prediction_layers=[]) 
    CLayer = norm ? GCNConv : UGCNConv
          
    init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

    hI = cat([in], hI, dims = 1)
    layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] =>nNodes, act; init = init)))
    
    Graphormers = GraphormerBlock[]
    for _ in 1:nBlocks
        push!(Graphormers, model_device(GraphormerBlock( where_sample.hidden_state,nNodes,init,rng,pDrop,hH,CLayer,act,act_conv,aggr)))
    end
    
    Graphormers=GNNChain(Graphormers...)
    
	 if isempty(prediction_layers) 
        prediction_layers=[nBlocks]
    end
    
    Decoders = []
    for i in prediction_layers
        inpL =  where_sample.before_decoding ? Int64( nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
        outL =  where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
    Decoders=Chain(Decoders...)

    Sampling = Sampler(rng)
	m = Graphormer(HiddenMap,Graphormers,Decoders,Sampling, where_sample, prediction_layers, dt) 
	return m
end

function create_model_gasse(where_sample, in, nBlocks,nNodes, out = 1, act = relu,act_conv=relu, seed = 1, hI = [100,250,500], hH = [500] , hF = [500, 250, 100],  pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true,aggr=mean,prediction_layers=[]) 
    ConvLayer = norm ? GCNConv : UGCNConv
          
    init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

    hI = cat([in], hI, dims = 1)
    layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] =>nNodes, act; init = init)))
    
    Graphormers = GraphormerBlock[]
    for _ in 1:nBlocks
        hMLP = cat([nNodes], hH, dims = 1);
        layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)];
        Convolution = ConvLayer(nNodes => nNodes,act_conv,; aggr, init);
        MLP = identity;
        push!(Graphormers, model_device(GraphormerBlock(Convolution,MLP)));
    end
    
    Graphormers=GNNChain(Graphormers...)
    
	if isempty(prediction_layers) 
        prediction_layers=[nBlocks]
    end
    
    Decoders = []
    for i in prediction_layers
        inpL =  where_sample.before_decoding ? Int64( nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
        outL =  where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
    Decoders=Chain(Decoders...)

    Sampling = Sampler(rng)
	m = Graphormer(HiddenMap,Graphormers,Decoders,Sampling, where_sample, prediction_layers, dt) 
	return m
end

function create_model_nair(where_sample, in, nBlocks,nNodes, out = 1, act = relu,act_conv=relu, seed = 1, hI = [100,250,500], hH = [500] , hF = [500, 250, 100],  pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true,aggr=mean,prediction_layers=[]) 
    ConvLayer = norm ? GCNConv : UGCNConv
          
    init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

    hI = cat([in], hI, dims = 1)
    layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] =>nNodes, act; init = init)))
    
    r1 = LayerNorm(1.0f-5, nNodes);
    Graphormers = GraphormerBlock[]
    for _ in 1:nBlocks    
        hMLP = cat([nNodes], hH, dims = 1);
        layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)];
        Convolution = GNNChain(Parallel(+, GNNChain( Chain( layersMLP..., Dense(hMLP[end] => nNodes, act; init = init)), ConvLayer(nNodes => nNodes; aggr, init),r1), identity));
        
        MLP = identity;
        
        push!(Graphormers, model_device(GraphormerBlock(Convolution,MLP)))
    end
    
    Graphormers=GNNChain(Graphormers...)
    
	 if isempty(prediction_layers) 
        prediction_layers=[nBlocks]
    end
    
    Decoders = []
    for i in prediction_layers
        inpL =  where_sample.before_decoding ? Int64( nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
        outL =  where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
    Decoders=Chain(Decoders...)

    Sampling = Sampler(rng)
	m = Graphormer(HiddenMap,Graphormers,Decoders,Sampling, where_sample, prediction_layers, dt) 
	return m
end

struct learningSampleNair <: learningBlockGNN end

struct learningSampleGasse <: learningBlockGNN end

struct learningSampleTransformer <: learningBlockGNN end

struct learningSampleOutside <: learningBlockGNN end

struct learningTransformer <: learningBlockGNN end

struct learningMultiPredTransformer <: learningBlockGNN end

struct learningMultiPredSample <: learningBlockGNN end

function create_model(lType::learningMultiPredSample, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true; final_A = identity)
    where_sample = SamplingPosition(false,false,true)
    return create_model(where_sample, in, block_number,nodes_number, out, a,a, seed, hI, h , hF,  pDrop, dt, std, norm,mean,collect(1:block_number))
end

function create_model(lType::learningMultiPredTransformer, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true; final_A = identity)
    where_sample = SamplingPosition(false,false,false)
    return create_model(where_sample, in, block_number,nodes_number, out,a,a, seed, hI, h , hF,  pDrop, dt, std, norm,mean,collect(1:block_number))
end

function create_model(lType::learningSampleTransformer, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001,dt::abstract_deviation=cr_deviation(), std = 0.0001, norm = true;final_A=identity)
    where_sample = SamplingPosition(false,false,true)
    return create_model(where_sample, in, block_number,nodes_number, out, a, a, seed, hI, h , hF,  pDrop, dt, std, norm, mean, [])
end

function create_model(lType::learningSampleGasse, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001,dt::abstract_deviation=cr_deviation(), std = 0.0001, norm = true;final_A=identity)
    where_sample = SamplingPosition(false,false,true)
    return create_model_gasse(where_sample, in, block_number,nodes_number, out, a, a, seed, hI, h , hF,  pDrop, dt, std, norm, mean, [])
end

function create_model(lType::learningSampleNair, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001,dt::abstract_deviation=cr_deviation(), std = 0.0001, norm = true;final_A=identity)
    where_sample = SamplingPosition(false,false,true)
    return create_model_nair(where_sample, in, block_number,nodes_number, out, a, a, seed, hI, h , hF,  pDrop, dt, std, norm, mean, [])
end

function create_model(lType::learningTransformer, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001,dt::abstract_deviation=cr_deviation(), std = 0.0001, norm = true;final_A=identity)
    where_sample = SamplingPosition(false,false,false)
    return create_model(where_sample, in, block_number,nodes_number, out, a, a, seed, hI, h , hF,  pDrop, dt, std, norm, mean, [])
end

function create_model(lType::learningSampleOutside, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001,dt::abstract_deviation=cr_deviation(), std = 0.0001, norm = true;final_A=identity)
    where_sample = SamplingPosition(true,false,false)
    return create_model(where_sample, in, block_number,nodes_number, out, a, a, seed, hI, h , hF,  pDrop, dt, std,norm, mean, [])
end

function create_model_v2(where_sample, in, nBlocks,nNodes, out = 1, act = relu,act_conv=relu, seed = 1, hI = [100,250,500], hH = [500] , hF = [500, 250, 100],  pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true,aggr=mean,prediction_layers=[]) 
    CLayer = norm ? GraphConv : UGCNConv
          
    init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

    hI = cat([in], hI, dims = 1)
    layersI = [Dense(hI[i] => hI[i+1], act; init = init) for i in 1:(length(hI)-1)]
	HiddenMap = model_device(Chain(layersI..., Dense(hI[end] =>nNodes, act; init = init)))
    
    Graphormers = Any[]
    for _ in 1:nBlocks
        grf = GraphormerBlock( where_sample.hidden_state,nNodes,init,rng,pDrop,hH,CLayer,act,act_conv,aggr)
        push!(Graphormers, GNNChain(grf.Convolution,grf.MLP))
    end
    
    Graphormers=GNNChain(Graphormers...)
    
	 if isempty(prediction_layers) 
        prediction_layers=[nBlocks]
    end
    
    Decoders = []
    for i in prediction_layers
        inpL =  where_sample.before_decoding ? Int64( nNodes / 2) : nNodes
		hF_i = cat([Int64(inpL)], hF, dims = 1)
		layersF = [Dense(hF_i[j] => hF_i[j+1], act; init = init) for j in 1:(length(hF_i)-1)]
        outL =  where_sample.outside ? 2 * out : out
		final = model_device(Chain(layersF..., Dense(hF_i[end] => outL; init = init)))
		push!(Decoders, final)
	end
    Decoders=Chain(Decoders...)

    Sampling = Sampler(rng)
	
    im1 = length(prediction_layers) > 1 ? prediction_layers[end-1] : 1
    
    sampling1 = where_sample.hidden_state ? Sampling : identity
    sampling2 = where_sample.before_decoding ? Sampling : identity
    sampling3 = where_sample.outside ? Sampling : identity

    model = GNNChain(sampling1,Graphormers[im1:end],sampling2,Decoders[end],sampling3)
    
    if im1 > 1
        idx = length(prediction_layers)-1
        for i in prediction_layers[end-1:-1:1]
            decoder = Chain(sampling2,Decoders[idx],sampling3) 
            model = GNNChain(sampling1,Graphormers[im1:i], Parallel(model,GNNChain(decoder),vcat))
            idx-=1
        end
    end
    model = GNNChain(HiddenMap,model)
    
    
    return Graphormer_v2(model_device(model),length(prediction_layers),aggr,dt)
end

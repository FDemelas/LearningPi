struct learningMultiPredSampleFeat <: learningBlockGNN
end

mutable struct FeatSampleMultiPred
	Encoder::Any
	TransformerBlocks::Any
	Decoders::Any
	dt::abstract_deviation
	rng::Any
end

mutable struct multiFeatExamples_instance <: abstract_example
	instance::Any
	features::Any
    features_vector::Any
	gold::Any
	linear_relaxation::Any
    max_layers::Int
end

mutable struct multiFeatDataSet <: abstract_dataset
	examples_list::Vector{multiFeatExamples_instance}
end

"""
createEmptyDataset(lt::learningMultiPredSampleFeat)

# Arguments:
- `lt`: learning Multi Layer Perceptron type.

Create an empty dataset for the Graph Neural network learning type.
"""
function createEmptyDataset(lt::learningMultiPredSampleFeat)
	return multiFeatDataSet([])
end


"""
create_example(lt::learningMultiPredSampleFeat, fileName::String, factory::abstractInstanceFactory)

#Arguments:
- `lt`: learning Type, this function works for all the learning types that use a graph representation of the instance
- `fileName`: path to the json that contains all the information to construct the learning sample starting from an instance, its features and the labels.
- `factory`: instance factory, it works for all the factory

returns an gnnExample_instance with all the information useful for the training.
"""
function create_example(lt::learningMultiPredSampleFeat, fileName::String, factory::abstractInstanceFactory, fmt::abstract_features_matrix)
	instance, gold, featuresObj = dataLoader(fileName, factory)
	features = featuresExtraction(lt, featuresObj, instance, fmt)
	linear_relaxation = featuresObj.objLR
    
    z = cpu(deviationFrom(features, typeof(fmt) == cr_features_matrix ? cr_deviation() : zero_deviation()))
	obj, predict_x, _ = LR(instance,z)
    
    lp=Instances.lengthLM(instance)
    pi_feat = zeros(size(features.ndata.x,2))
    pi_feat[1:lp]=reshape(z,:)
    x_feat = zeros(size(features.ndata.x,2))
    x_feat[lp+1:lp+length(predict_x)]=reshape(predict_x,:)
    base_features = vcat(pi_feat',x_feat')
    
    features_vector = Any[base_features]
    return multiFeatExamples_instance(instance, features,features_vector, gold, linear_relaxation,1)
end

"""
function (m::FeatSampleMultiPred)(x)

# Arguments:
   - `x`: input of the NN model of type FeatSampleMultiPred    

Forward computation of a FeatSampleMultiPred m, the output is the concatenation of all the multipliers predicted by the model   
"""
function (m::FeatSampleMultiPred)(x, data, train_mode=true)
	sizes = cpu(x.gdata.u)
	sizes = (sizes[1],sizes[2]) 
	f = m.Encoder(x.ndata.x)
	xHidden = m.TransformerBlocks[1](x, vcat(f, model_device(data.features_vector[1] )))
	if train_mode
		z = sample_latent(m,xHidden,sizes)
	else
		z,_ = MLUtils.chunk(xHidden, 2; dims = 1)
	end
	base = reshape(m.Decoders[1](z)[1:prod(sizes)], sizes) +  model_device(deviationFrom(x,m.dt))
	o = copy(base)
	n_tot_blocks=length(m.TransformerBlocks)
	for i in 2:min(n_tot_blocks,data.max_layers)
		xHidden = (m.TransformerBlocks[i](x, vcat(xHidden, model_device(data.features_vector[i]))))
		if train_mode
			z=sample_latent(m,xHidden,sizes)
		else
			z,_=MLUtils.chunk(xHidden, 2; dims = 1)
		end
		pred = reshape(m.Decoders[i](z)[1:prod(sizes)], sizes)
		o = vcat(o, base +  pred) # deviationFrom(x,m.dt))
		base += copy(pred)
	end
	return o
end



"""
function (m::FeatSampleMultiPred)(x)

# Arguments:
   - `x`: input of the NN model of type FeatSampleMultiPred    

Forward computation of a FeatSampleMultiPred m, the output is the concatenation of all the multipliers predicted by the model   
"""
function initialize_model(m,dataset)
	for (train_mode, data) in [(true,dataset.train),(false,dataset.val),(false,dataset.test)]
		for sample in data.examples_list
			x=model_device(sample.features)
			sizes = cpu(x.gdata.u)
			sizes = (sizes[1],sizes[2]) 
			f = m.Encoder(x.ndata.x)
			xHidden = m.TransformerBlocks[1](x, vcat(f, model_device(sample.features_vector[1] )))
			if train_mode
				z = sample_latent(m,xHidden,sizes)
			else
				z, _ = MLUtils.chunk(xHidden, 2; dims = 1)
			end
			base = reshape(m.Decoders[1](z)[1:prod(sizes)], sizes) +  model_device(deviationFrom(x,m.dt))
			o = copy(base)
			n_tot_blocks=length(m.TransformerBlocks)
			update_features_vector(sample,o)
			for i in 2:n_tot_blocks
				xHidden = (m.TransformerBlocks[i](x, vcat(xHidden, model_device(sample.features_vector[i]))))
				if train_mode
					z=sample_latent(m,xHidden,sizes)
				else
					z,_=MLUtils.chunk(xHidden, 2; dims = 1)
				end
				pred = reshape(m.Decoders[i](z)[1:prod(sizes)], sizes)
				o = vcat(o, base +  pred) # deviationFrom(x,m.dt))
				base += copy(pred)
				update_features_vector(sample,o)
				sample.max_layers+=1
			end
		end
	end
end

# Call @functor to allow for training. Described below in more detail.
Flux.@functor FeatSampleMultiPred

"""
function create_transformer_block(inOut, hMLP, act, pDrop, init, rng, norm)

	#Arguments:
	- `inOut`: the input ( = the ouput) size of the transformer block, i.e the size of the hidden features representation
	- `hMLP`: the size of the MLP considered for the non-linear part.
	- `act`: the activation function for the non-linear layers
	- `pDrop`: the drop-out probability in the DropOut Layers
	- `init`: the initialiser for the model parameters
 
	Creates and returns a transformer block with the provided characteristics.
"""
function create_transformer_block_dynFeat(inOut, hMLP, act, pDrop, init,rng, norm)
	hMLP = cat([inOut], hMLP, dims = 1)
	layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)]

	r1 = LayerNorm(1.0f-5, inOut)
	r2 = LayerNorm(1.0f-5, inOut)
	
	if norm
#		CLayer=GCN_basic()
#		CLayer=GCNConv(inOut => inOut; init = init)
#		CLayer=TransformerConv((inOut,0)=>inOut;init=init)
		CLayer=GraphConv(inOut+2 => inOut, identity; aggr=mean, init=init) |> model_device
	else
		CLayer=UGCNConv(inOut+2 => inOut; init = init) |> model_device
	end	
	
	return GNNChain(
		Parallel(+, GNNChain( r1,CLayer, Dropout(pDrop;rng)), Dense(inOut+2 => inOut,act)),
		Parallel(+, Chain( r2, layersMLP..., Dense(hMLP[end] => inOut, act; init = init), Dropout(pDrop;rng)), identity),
	)
end

"""
function create_model(lType::learningMultiPredSampleFeat, in, h, out=1, a=relu, seed=1, hI=[500, 250, 100], hF=[500, 250, 100], block_number::Int64=5, nodes_number::Int64=500, pDrop=0.001, variance=0.000001)

#Arguments:
- `lType`: the learning type.
- `in`: the input size (size of the input features for each node of the bipartite graph representation)
- `h`: the size of the hidden MLP, used in the transformer blocks
- `out`: the output size [by default 1]
- `a`: the activation function [ by default relu] 
- `seed`: the random generation seed for the initialiser of the model parameters [ by default 1], 
- `hI`: the size of the initial MLP [by default [500, 250, 100]] 
- `hF`: the size of the final MLP [by default [500, 250, 100]] 
- `block_number`: the number of transformer blocks [by default 5], 
- `nodes_number`:  the size of the hidden features representation, i.e. the input/output size of each transformer block [by default 500] 
- `pDrop`: probability of drop-out in the DropOut Layers [by default 0.001] 
- `std`: the standard deviation used for the initialiser of the model parameters [by default 0.0001]

Create a learningMultiPredSampleFeat model, that is a model composed of transformer block, where  we make a Lagrangian multipliers prediction after each block.
Each prediction will be considered in the correspondent loss.
"""
function create_model(lType::learningMultiPredSampleFeat, in, h, out = 1, a = relu, seed = 1, hI = [500, 250, 100], hF = [500, 250, 100], block_number::Int64 = 5, nodes_number::Int64 = 500, pDrop = 0.001, dt::abstract_deviation=cr_deviation(), std=0.00001, norm=true; final_A = identity)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = std)
	rng = Random.default_rng(seed)

	hI = cat([in], hI, dims = 1)
	layersI = [Dense(hI[i] => hI[i+1], a; init = init) for i in 1:(length(hI)-1)]
	encoder = Chain(layersI..., Dense(hI[end] => nodes_number, a; init = init))

	Decoders = []
	TransformerBlocks = []
	for _ in 1:block_number
		hF = cat([Int64(nodes_number/2)], hF, dims = 1)
		layersF = [Dense(hF[i] => hF[i+1], a; init = init) for i in 1:(length(hF)-1)]
		final = Chain(layersF..., Dense(hF[end] => out; init = init)) |> model_device

		block = create_transformer_block_dynFeat(nodes_number, h, a, pDrop, init, rng, norm) |> model_device
		push!(Decoders, final)
		push!(TransformerBlocks, block)
	end

	m = FeatSampleMultiPred(encoder, TransformerBlocks, Decoders, dt, rng) |> model_device

	return m
end


"""
forwardBackward(lossParam::abstractLoss,trainSet,gnn,currentMetrics,opt,loss,epoch::Int64,lt::learningMultiPredSampleFeat, dt::abstract_deviation)
	
# Arguments:

- `lossParam`: a structure that contains the parameters α and β of the loss.
- `trainSet`: the training dataset structure.
- `nn`: the neural network model.
- `currentMetrics`: a dictionary of Float 
- `opt`: the optimizer used for the training.
- `loss`: the loss function
- `epoch`: the current epoch.
- `lt`: graph neural network learning type object, should be learningMultiPredSampleFeat.   
- `dt`: deviation type (0 or duals of the continuous relaxation)

This function performs the forward-backward pass for the training considering a generic loss and for the learningMultiPredSampleFeat graph neural network model.

"""
function forwardBackward(lossParam::abstractLoss, trainSet, gnn, currentMetrics, opt, loss, epoch, lt::learningMultiPredSampleFeat,dt::abstract_deviation)
	nInst = length(trainSet)
	currentMetrics["loss training"] = 0
	currentMetrics["GAP training"] = 0
	par = Flux.params(gnn)
	lossValue = 0.0f0
	GAPValue = 0.0f0
	GAP_CR_Value = 0.0f0
	device = get_device(lossParam)

	for (idx,example) in enumerate(trainSet)
		#CUDA.seed!(idx)

		feat = model_device(example.features)
		
		v, grad = Flux.withgradient(() -> loss(device(gnn(feat,example)); hei = example, lossPar = lossParam), par)

		lossValue += v / nInst

        πs = (device(gnn( feat,example)))

		π = reshape(πs[1:Instances.sizeLM(example.instance)[1],:],Instances.sizeLM(example.instance))
		objPred = sub_problem_value(π, v, example, lossParam)
		
		objGold = example.gold.objLR
		GAPValue += (objGold - objPred) / (objGold * nInst)
		GAP_CR_Value += (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) / nInst
		Flux.update!(opt, par, grad)

        update_features_vector(example,πs)
	end
	currentMetrics["loss training"] = lossValue |> device
	currentMetrics["GAP training"] = GAPValue * 100 |> device
	currentMetrics["GAP CR training"] = GAP_CR_Value * 100 |> device
end

function update_features_vector(sample,o)
	o=cpu(o)
    lo = Int64(size(o)[1] / sample.features.gdata.u[1])
    #grad = zeros(Float32, Instances.sizeLM(sample.instance))
    sample.features_vector=[sample.features_vector[1]]
        for i in 1:lo
            sIdx = ((i - 1) * sample.features.gdata.u[1] + 1)
            eIdx = i * sample.features.gdata.u[1]
            obj, predict_x, _ = LR(sample.instance, o[sIdx:eIdx, :] |> cpu)
            #value += (sample.gold.objLR - obj) * ((lossPar.α)^(lo - i))
            #gradientX(grad, sIdx, predict_x, sample.instance, (lossPar.α)^(lo - i))
            pi_feat = zeros(size(sample.features.ndata.x,2))
            lp=Instances.lengthLM(sample.instance)
            pi_feat[1:lp]=reshape(o[sIdx:eIdx, :],:)
            x_feat = zeros(size(sample.features.ndata.x,2))
            x_feat[lp+1:lp+length(predict_x)]=reshape(predict_x,:)
            new_feat = hcat(cpu(pi_feat),cpu(x_feat))
            push!(sample.features_vector,new_feat')
        end
    end


function validation(currentMetrics::Dict, valSet, nn::FeatSampleMultiPred, lossParam::abstractLoss, lt::learningType,dt::abstract_deviation)
	loss=createLoss(lossParam)
	nInstVal = length(valSet)
	Flux.testmode!(nn)
	device = get_device(lossParam)
	for example in valSet
		feat = model_device(example.features)
		πs = (device(nn( feat,example)))

		s1,s2 = sizeLM(example.instance)

        update_features_vector(example,πs)
		π = πs[1:s1,1:s2]

		v = loss(device(π); hei = example, lossPar = lossParam)

		objPred = sub_problem_value(π, v, example, lossParam)
		objGold = example.gold.objLR
		sp_sign=is_min_SP(example.instance) ? 1 : -1

		currentMetrics["loss validation"] += sp_sign * v / (nInstVal)
		currentMetrics["GAP validation"] += sp_sign * (objGold - objPred) / (objGold * nInstVal) * 100
		currentMetrics["GAP CR validation"] += (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) * 100 / nInstVal
	end
end


function testAndPrint(currentMetrics::Dict, testSet, nn::FeatSampleMultiPred, lossParam::abstractLoss, lt::learningType,dt::abstract_deviation)

	loss=createLoss(lossParam)

	nInstTest = length(testSet)

	Flux.testmode!(nn)
	device = get_device(lossParam)
	
	times=[]
	for example in testSet
		t=0
		t0=time()
		CR(example.instance)
		t+=time()-t0

		feat = model_device(example.features)

		t0=time()
		println(feat)
		
		πs = (device(nn( feat,example)))

		s1,s2 = sizeLM(example.instance)

        update_features_vector(example,πs)
		π = πs[1:s1,1:s2]

		t+=time()-t0
#		π = prediction(nn, feat, example, lt,dt)
		v = loss(π; hei = example, lossPar = lossParam)

		objPred = sub_problem_value(π, v, example, lossParam)
			
		t0=time()
		objPred,_=LR(example.instance, π)
		t+=time()-t0

		sp_sign=is_min_SP(example.instance) ? 1 : -1

		objGold = example.gold.objLR

		currentMetrics["loss test"]   += sp_sign*v / (nInstTest)
		currentMetrics["GAP test"]    += sp_sign*(objGold - objPred) / (objGold * nInstTest) * 100
		currentMetrics["GAP CR test"] += (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) * 100 / nInstTest
		append!(times,t)
	end
	println("-----------------")
	println("--TEST--RESULTS--")
	println("-----------------")

	for key in keys(currentMetrics)
		if occursin("test", key)
			println(key, ": ", currentMetrics[key])
		end
	end
	println("time : ", mean(times))
end


"""
function get_model(nn::FeatSampleMultiPred)

#Arguments:
	-`nn`: neural network model

returns a cpu version of the model that can be saved using a bson file.    
"""
function get_model(nn::FeatSampleMultiPred)
	return nn # [nn.Encoder, nn.TransformerBlocks, nn.Decoders,nn.dt, nn.rng]
end



function prediction(model, x, data, lt::learningMultiPredSampleFeat,dt::abstract_deviation)
	πs = cpu(model(x,data.features_vector,false))
	out = πs[(size(πs)[1]-cpu(x.gdata.u)[1]+1):end, :]
	return out
end

function load_model(nn,lt::learningMultiPredSampleFeat)
	return nn # FeatSampleMultiPred(nn[1], nn[2], nn[3],nn[4], nn[5])
end
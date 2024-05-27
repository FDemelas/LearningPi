struct learningMLP <: learningType end

"""
create_model(lType::learningMLP, in, h, out=1, a=relu)

# Arguments:
- `lType`: general learningType.
- `in`: size of the input layer.
- `h`: a vector with the same length as the desired number of hidden layers and each component say how many nodes we want in the correspondent hidden layer.
- `out`: size of the output layer, by default is equal to one.
- `a`: the activation function for the hidden layers, by default is `relu`.

This function creates a model for the provided learning type (in order to use this variant it should be learningArc).
The model is a multi layer perceptron with `in` nodes in the first layer, `length(h)` hidden layers (the i-th layer has `h[i]` nodes) and `out` nodes in the output layer (by default 1).
By default each hidden layer use a relu activation function (the input and output layers have no activation function).   
"""
function create_model(lType::learningMLP, in, h, out = 1, a = relu, seed = 1;final_A=identity)
	# The last layers has no activation function, the others relu
	h = cat([in], h, dims = 1)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.0001)
	layers = [Dense(h[i] => h[i+1], a; init = init) for i in 1:(length(h)-1)]# init=Flux.truncated_normal(std=0.1,lo=-0.1,hi=0.1)
	model = Chain(layers..., Dense(h[end] => out, final_A; init = init), X -> dropdims(X, dims = 1)) |> model_device
	return model
end

"""
prediction(nn, f,ins,lt::learningMLP)

# Arguments:
-`nn`: neural network model.
-`f`: features matrix.
-`ins`: structure containing the instance informations.
-`lt`: learning type (general).

"""
function prediction(nn::Chain, f, data, lt::learningMLP, dt::abstract_deviation)
	π = dropdims(nn(f), dims = 1)
	zCR = f[1, :, :]
	return sign(π .+ zCR,data)
end

"""
sizeFeatures(lt::learningType,dS)

# Arguments:
- `lt` : learning type (general).
- `dS` : dataset (corpus structure)

"""
function sizeFeatures(lt::learningMLP, dS)
	return size(dS.train.examples_list[1].features, 1)
end

"""
createEmptyDataset(lt::learningMLP)

# Arguments:
- lt: learning Multi Layer Perceptron type.

Create an empty dataset for the  Multi Layer Perceptron learning type.
"""
function createEmptyDataset(lt::learningMLP)
	return gnn_dataset([])
end

function adj_var_constr(ins::abstractInstanceMCND)
	G = zeros(sizeK(ins),sizeV(ins),sizeE(ins))
	for ij in 1:sizeE(ins)
		for k in 1:sizeK(ins)
			G[k, head(ins, ij), ij] = -1 * isInKij(ins, k, ij)
			G[k, tail(ins, ij), ij] = 1 * isInKij(ins, k, ij)
		end
	end
	return G
end	

function adj_var_constr(ins::instanceGA)
	G = zeros( 1,ins.I, ins.J)
	for i in 1:ins.I
		for j in 1:ins.J
			G[1, i , j] = 1
			G[1, i , j] = 1
		end
	end
	return G
end	

function adj_var_constr(ins::instanceCWL)
        G = zeros( 1,ins.J, ins.I)
        for i in 1:ins.J
                for j in 1:ins.I
                        G[1, i , j] = 1
                end
        end
        return G
end

function features_variables(ins::abstractInstanceMCND,featObj,G)
	f = zeros(10,sizeK(ins),sizeE(ins))
	for ij in 1:sizeE(ins)
		for k in 1:sizeK(ins)
			f[1, k, ij] = mean(G[k,:,ij])
			f[2, k, ij] = var(G[k,:,ij])
			f[3, k, ij] = 1
			f[4, k, ij] = 0
			f[5, k, ij] = featObj.xCR[k,ij]
			f[6, k, ij] = ins.r[k,ij]
			f[7, k, ij] = featObj.xRC[k,ij]
			f[8, k, ij] = 0
			f[9, k, ij] = 0
			f[10, k, ij] = ins.c[ij] < ins.K[k][3] ?  ins.c[ij] : ins.K[k][3]
		end
	end
	return f
end

function features_variables(ins::instanceCWL,featObj,G)
        f = zeros(10,ins.J,ins.I)
        for i in 1:ins.J
                for j in 1:ins.I
                        f[1, i, j] = mean(G[1,i,j])
                        f[2, i, j] = 0
                        f[3, i, j] = ins.q[j]
                        f[4, i, j] = 0
                        f[5, i, j] = featObj.xCR[j,i]
                        f[6, i, j] = ins.c[j,i]
                        f[7, i, j] = featObj.xRC[j,i]
                        f[8, i, j] = 1
                        f[9, i, j] = 0
                        f[10, i, j] = 1
                end
        end
        return f
end

function features_variables(ins::instanceGA,featObj,G)
	f = zeros(10,ins.I,ins.J)
	for i in 1:ins.I
		for j in 1:ins.J
			f[1, i, j] = mean(G[1,i,j])
			f[2, i, j] = 0
			f[3, i, j] = ins.w[i,j]
			f[4, i, j] = 0
			f[5, i, j] = featObj.xCR[i,j]
			f[6, i, j] = ins.p[i,j]
			f[7, i, j] = featObj.xRC[i,j]
			f[8, i, j] = 1
			f[9, i, j] = 0
			f[10, i, j] = 1
		end
	end
	return f
end

function aggregate_features(ins::abstractInstance,varFeatures,G)
	S1,S2 = sizeLM(ins)
	conFeat=zeros(size(varFeatures,1),S1,S2)
	for f in 1:size(varFeatures,1)
		for k in 1:S1
			conFeat[f,k,:] = G[k,:,:]*varFeatures[f,k,:]
		end
	end
	return conFeat
end

function rhs(ins::abstractInstanceMCND,k,i)
	return b(ins,i,k)
end

function rhs(ins::instanceGA,k,i)
	return ins.c[i]
end

function rhs(ins::instanceCWL,k,i)
        return ins.q[i]
end

"""
featuresExtraction(featType::learningMLP, featObj, ins)

Vectorization function for the features when we consider a learningNodeDemand encoding.

# Arguments:
- `featType` features type
- `features` a features matrix
- `nbFeatures` the number of features
"""
function featuresExtraction(featType::learningMLP, featObj, ins::abstractInstance)
	
	λ = featObj.λ

	nbFeatures = 14

	features = zeros((nbFeatures, sizeLM(ins)...))

	G = adj_var_constr(ins)


	varFeatures = features_variables(ins, featObj, G)
	conFeatures = aggregate_features(ins, varFeatures, G)

	S1,S2 = sizeLM(ins)
	for k in 1:S1
		for i in 1:S2
			features[:,k,i]= [λ[k,i],featObj.λSP[k,i],1,rhs(ins,k,i),conFeatures[:,k,i]...]
		end
	end

	return Array{Float32, 3}(features) 
end


"""
create_example(lt::learningArc, fileName::String,factory::abstractInstanceFactory)
	
# Arguments:
- lt: learning type, it should be learningMLP.
- fileName: the name of the file json that contains the informations about the instance, its features and its labels.
- factory: type of instance (it handle both with the normalized and un-normalized instances).

Create a structure containing the instance, the extracted features and the associated labels.
"""
function create_example(lt::learningMLP, fileName::String, factory::abstractInstanceFactory,cr_features_matrix)
	instance, gold, featuresObj = dataLoader(fileName, factory)
	features = featuresExtraction(lt, featuresObj, instance)
	linear_relaxation = featuresObj.objLR
	return example_gnn(instance, features, gold, linear_relaxation)
end


"""
forwardBackward(trainSet,nn,currentMetrics,opt,loss,epoch::Int64,lt::learningType,dt::abstract_deviation)

# Arguments:

- `loss`: a structure that contains the parameters α and β of the loss.
- `trainSet`: the training dataset structure.
- `nn`: the neural network model.
- `currentMetrics`: a dictionary of Float 
- `opt`: the optimizer used for the training.
- `loss`: the loss function
- `epoch`: the current epoch.
- `lt`: learning type object.   
- `dt`: deviation type (0 or duals of the continuous relaxation)

This function performs the forward-backward pass for the training considering a generic loss and a generic learning type.
"""

function forwardBackward(trainSet, nn, currentMetrics, opt, loss, epoch, lt::learningMLP, dt::abstract_deviation,bs=1)	
	nInst = length(trainSet)
	currentMetrics["loss training"] = 0
	currentMetrics["GAP training"] = 0
	currentMetrics["GAP CR training"] = 0
	device = get_device(loss)
	par = Flux.params(nn)
	for (iter, example) in enumerate(trainSet)
#		CUDA.seed!(iter)
		feat = model_device(example.features)
		zCR = example.features[1, :, :] |> model_device
		let π, v
			grad = gradient(par) do
				π = dropdims(nn(feat), dims = 1)
				π += zCR
				π = device(is_min_SP(example.instance)*π)
				v = loss(π |> device; example)
			end

			objPred = sub_problem_value(π, v, example, loss)

			objGold = example.gold.objLR

			currentMetrics["loss training"] += v / (nInst)
			currentMetrics["GAP training"] += (objGold - objPred) / (objGold * nInst) * 100
			currentMetrics["GAP CR training"] += (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) / nInst * 100

			Flux.update!(opt, par, grad)
		end
	end
end

function get_λ(x::CuArray)
	return model_device(x[1,:,:])
end

function load_model(nn,lt::learningMLP)
        return nn
end
using ArgParse
using LearningPi
using Random: MersenneTwister
using Flux
using Dates
using JSON
using CUDA

MaxEp = 300
maxInst = [-1, -1, -1]

"""
main(args)

Main function to perform the training using the k-fold, for a given k as test.
This function takes some arguments as input considering ArgParse, the description of that parameters 
can be found in the body of this function.    
	
"""
function main(args)
	s = ArgParseSettings("Training a model with k-fold: " *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option
	@add_arg_table! s begin
		"--opt"
		nargs = '?'
		arg_type = String
		default = "ADAM"
		help = "optimizer for the training"
		"--lr"
		required = true
		arg_type = Float32
		help = "learning rate"
		"--seed"
		required = true
		arg_type = Int64
		help = "random seed"
		"--decay"
		required = true
		arg_type = Float32
		help = "exponential decay for the learning rate"
		"--MLP_i"
		nargs = '*'
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the initial MLP"
		"--MLP_h"
		nargs = '*'
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the hiddens MLP"
		"--MLP_f"
		nargs = '*'
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the final MLP"
		"--maxEp"
		nargs = 1
		arg_type = Int64
		help = "Maximum number of epochs"
		"--stepSize"
		nargs = '?'
		arg_type = Int64
		default = 1000
		help = "Step size for the decay"
		"--lossType"
		nargs = 1
		arg_type = String
		help = "Type of loss used for the training"
		"--kFold"
		required = true
		arg_type = Int64
		help = "k to choose the test folder"
		"--lossParams"
		nargs = '*'
		arg_type = Float32
		help = "Parameters for the loss"
		"--data"
		arg_type = String
		default = "/users/demelas/tmp50/"
		help = "dataset folder for the training"
		"--pDrop"
		required = true
		arg_type = Float32
		help = "Dropping probability"
		"--block_number"
		required = true
		arg_type = Int64
		help = "Number of transformer block layers"
		"--nodes_number"
		required = true
		arg_type = Int64
		help = "Number of nodes as input/output for the Transformer block layers"
		"--learningType"
		required = true
		arg_type = String
		help = "Learning Type, for the selection of the model type"
		"--factory"
		required = true
		arg_type = String
		help = "Factory, for the selection of the instance "
		"--ones_weights"
		nargs = 1
		arg_type = Bool
		default = true
		help = "If true the weights of the edges in the bipartite graph representation of the instance are only ones"
		"--beta"
		nargs = 1
		default=-1
		arg_type = Float32
		help = "Another parameter for the weights of the edges in the bipartite graph representation of the instance, if >0 then we consider as preprocessing w->exp(beta*w) for the weights."
	end

	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)

	#path to the training data
	data = parsed_args["data"]

	#construct the correspondent learning type starting from the provided string
	lt = getLT(parsed_args["learningType"])
	
	dt = LearningPi.cr_deviation()
	
	fmt = LearningPi.cr_features_matrix(parse_args["ones_weights"],parse_args["beta"])
		
	#construct the correspondent instance factory starting from the provided string
	factory = getFactory(parsed_args["factory"])

	# construct the three arrays of hidden dimension for the MLP encoder, the MLPs in the hidden Transformer
	# blocks and the final MLP decoder
	iMLP = [parse(Int64, parsed_args["MLP_i"][i]) for i in 1:length(parsed_args["MLP_i"])]
	hMLP = [parse(Int64, parsed_args["MLP_h"][i]) for i in 1:length(parsed_args["MLP_h"])]
	fMLP = [parse(Int64, parsed_args["MLP_f"][i]) for i in 1:length(parsed_args["MLP_f"])]

	#construct the datasets: training, validation and test
	#l, dS = createKfold(lt, fmt, data, maxInst, parsed_args["seed"]; factory = factory, k = parsed_args["kFold"])


	println(readdir(data))
	#construct the correspondent loss from the preovided imput parameters
	lossParam = lossFromParams(parsed_args["lossType"][1], parsed_args["lossParams"])
	loss = createLoss(lossParam)

	# construct the optimizer with exponential decay
	lrDecay = Flux.Optimise.ExpDecay(1., parsed_args["decay"], parsed_args["stepSize"], 1e-8, 1)
	opt = Flux.Optimiser(RAdam(parsed_args["lr"]), ClipNorm(1), lrDecay)

	# construction of the neural network model
	println("Memory status before training tasks")
	CUDA.memory_status()

	println("Creating the model")
	CUDA.@time nn = create_model(lt, size_nn_input(fmt), hMLP, size_nn_output(lt), Flux.relu, parsed_args["seed"], iMLP, fMLP, parsed_args["block_number"], parsed_args["nodes_number"], parsed_args["pDrop"]) |> gpu
	nn = create_model(lt, size_nn_input(fmt), hMLP, size_nn_output(lt), Flux.relu, parsed_args["seed"], iMLP, fMLP, parsed_args["block_number"], parsed_args["nodes_number"], parsed_args["pDrop"]) |> gpu	
	
	ins, gold, feat = dataLoader(data*readdir(data)[1],cpuMCNDinstanceFactory())
	f = featuresExtraction(LearningPi.learningSampleTransformer(), feat, ins, fmt)
	linear_relaxation = feat.objLR
	example = LearningPi.example_gnn(ins, f, gold, linear_relaxation)

	println("Before charging the features on gpu")
	CUDA.@time feat = gpu(example.features)
	feat = gpu(example.features)

	println("During the forward phase")
	CUDA.@time z = prediction(nn,feat,example.instance,lt,dt)	

	println("Memory status")
	CUDA.memory_status()


	println("During the backward phase")

	par = Flux.params(nn)
	if lt==LearningPi.learningTransformer()
		CUDA.@time ( z = prediction(nn, feat, example.instance, lt, dt);v = loss(cpu(z); hei = example, lossPar = lossParam ) )	
#	else 
#		CUDA.@time (nbpreds = prod(LearningPi.sizeLM(example.instance));zCR = deviationFrom(example.features.gdata.u[3],dt) |> CPU;
#					(v1, mLR), grad1 = Flux.withgradient(() ->
#						losses(nn, feat, nbpreds, -1, LearningPi.sizeLM(example.instance)[1], LearningPi.sizeLM(example.instance)[2], zCR, loss, example, lossParam), par))
#			)
	end

	println("Memorty consuption at the end")
	CUDA.memory_status()

end


function getFactory(factory::String)
	if factory == "cpuCWLinstanceFactory"
		return LearningPi.cpuCWLinstanceFactory()
	end
	if factory == "cpuMCNDinstanceFactory"
		return LearningPi.cpuMCNDinstanceFactory()
	end
	if factory == "gpuMCNDinstanceFactory"
		return LearningPi.gpuMCNDinstanceFactory()
	end
end


"""
lossFromParams(lossType::String,lossParams::Vector{Float32})

Construct a loss object taking as inputs a string for the loss type and a vector for the loss parameters.

rguments:
- `lt`: a string to indicate the learning type. The possible choices are: learningTransformer,learningSampleTransformer,learningMultiPredTransformer

"""
function getLT(lt::String)
	if lt == "learningTransformer"
		return LearningPi.learningTransformer()
	elseif lt == "learningSampleTransformer"
		return LearningPi.learningSampleTransformer()
	elseif lt == "learningMultiPredTransformer"
		return LearningPi.learningMultiPredTransformer()
	elseif lt=="learningSampleOutside"
		return LearningPi.learningSampleOutside()
	end
end

"""
lossFromParams(lossType::String,lossParams::Vector{Float32})

Construct a loss object taking as inputs a string for the loss type and a vector for the loss parameters.

Arguments:
- `lossType`: a string to indicate the loss type. 
- `lossParams`: a vector of float with the same size of the parameters required for the chosen loss,
			  see the documentation of the loss strucures to know more about it.

"""
function lossFromParams(lossType::String, lossParams::Vector{Float32})
	if lossType == "MSELoss"
	    return LearningPi.MSELoss(lossParams[1])
	end
	if lossType == "HingeLoss"
	    return LearningPi.HingeLoss(lossParams[1])
	end
	if lossType == "LRloss"
		return LearningPi.LRloss(lossParams[1])
	end

	if lossType == "gpuLRloss"
		return LearningPi.gpuLRloss(lossParams[1])
	end

	if lossType == "MultiPredLRloss"
		return LearningPi.MultiPredLRloss(lossParams[1])
	end

	if lossType == "GAPcloseLoss"
		return LearningPi.GAPcloseLoss(lossParams[1])
	end

	if lossType == "GAPloss"
		return LearningPi.GAPloss(lossParams[1])
	end
end

function size_nn_input(fmt::LearningPi.cr_features_matrix)
	return 20
end

function size_nn_input(fmt::LearningPi.without_cr_features_matrix)
	return 12
end

function size_nn_output(lt::LearningPi.learningType)
	return 1
end	

function size_nn_output(lt::LearningPi.learningSampleOutside)
	return 2
end	

main(ARGS)

using ArgParse
using LearningPi
using Random: MersenneTwister, Random
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
		nargs='?'
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
		arg_type = Int64
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the initial MLP"
		"--MLP_h"
		nargs = '*'
                arg_type = Int64
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the hiddens MLP"
		"--MLP_f"
		nargs = '*'
                arg_type = Int64
		default = [500, 250, 100]
		help = "number of nodes in each hidden layer for the final MLP"
		"--maxEp"
		required = true
		arg_type = Int64
		help = "Maximum number of epochs"
		"--stepSize"
		nargs = '?'
		arg_type = Int64
		default = 1000
		help = "Step size for the decay"
                "--lossType"
                nargs = '?'
                arg_type = String
                default = "loss_LR"
                help = "Type of loss used for the training"
		"--kFold"
		required = true
		arg_type = Int64
		help = "k to choose the test folder"
		"--lossParams"
		nargs = '?'
		arg_type = Float32
		help = "Parameters for the loss"
                "--data"
                arg_type = String
                default = "/users/demelas/MCNDsmallCom40/"
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
		arg_type = String
		default = "LearningSampleTransformer"
		help = "Learning Type, for the selection of the model type"
		"--factory"
		arg_type = String
		default = "MCNDinstance"
		help = "Factory, for the selection of the instance type"
		"--cr_deviation"
		default = true
		arg_type = Bool
		help = "Learn the deviation from zero (if false) or to the dual variables of the continuous relaxation"
		"--cr_features"
		default = true
		arg_type = Bool
		help = "Use (true) or not (false), the features associated to the continuous relaxation"
		"--norm"
		default = true
		arg_type = Bool
		help = "Normalize or Not the features"
		"--prefix"
		nargs = '?'
		arg_type = String
                default = ""
		help = "Memorization String Prefix"
		"--ones_weights"
		arg_type = Bool
		default = true
		help = "If true the weights of the edges in the bipartite graph representation of the instance are only ones"
		"--beta"
		default=  - 1.0f0
		arg_type = Float32
		help = "Another parameter for the weights of the edges in the bipartite graph representation of the instance, if >0 then we consider as preprocessing w->exp(beta*w) for the weights."
	end
	
		CUDA.math_mode()
	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)
	println(parsed_args)

	ones_weights = true # parsed_args["ones_weights"]
	beta = -1 # parsed_args["beta"]
	
	seed=parsed_args["seed"]
	CUDA.seed!(seed)
	CURAND.seed!(seed)

	Random.seed!(seed)
	#path to the training data
	data = parsed_args["data"]
	norm = parsed_args["norm"]

	#construct the correspondent learning type starting from the provided string
	lt = getLT(parsed_args["learningType"])



	if parsed_args["cr_deviation"]
		dt = LearningPi.cr_deviation()
	else
		dt = LearningPi.zero_deviation()
	end


	if parsed_args["cr_features"]
		fmt = LearningPi.cr_features_matrix(ones_weights,beta)
	else
		fmt = LearningPi.without_cr_features_matrix(ones_weights,beta)
	end

	#construct the correspondent instance factory starting from the provided string
	factory = getFactory(parsed_args["factory"])

	# construct the three arrays of hidden dimension for the MLP encoder, the MLPs in the hidden Transformer
	# blocks and the final MLP decoder
	iMLP = [parsed_args["MLP_i"][i] for i in 1:length(parsed_args["MLP_i"])]
	hMLP = [parsed_args["MLP_h"][i] for i in 1:length(parsed_args["MLP_h"])]
	fMLP = [parsed_args["MLP_f"][i] for i in 1:length(parsed_args["MLP_f"])]

println(1)
println(lt," ", typeof(fmt)," ", data," ", maxInst," ", seed, " ",factory, " ", parsed_args["kFold"])
	#construct the datasets: training, validation and test
	corpus_info, dS = createKfold(lt, fmt, data, maxInst, seed; factory, k = parsed_args["kFold"])

	#construct the correspondent loss from the preovided imput parameters
	loss = lossFromParams(parsed_args["lossType"],  parsed_args["lossParams"])
	
	# construct the optimizer with exponential decay
	lrDecay = Flux.Optimise.ExpDecay(1.0, parsed_args["decay"], parsed_args["stepSize"], 1e-10, 1)
	opt = Flux.Optimiser(Adam(parsed_args["lr"]), ClipNorm(5), lrDecay)

	# construct the string used to memorize the results
	endString =
		parsed_args["prefix"] * string(lt) * "model" * string(parsed_args["MLP_i"]) * "+" * string(parsed_args["block_number"]) * "x(" * string(parsed_args["nodes_number"]) * "," * string(parsed_args["MLP_h"]) * ", " * string(parsed_args["nodes_number"]) *
		")+" * "MLP" *
		string(parsed_args["MLP_f"]) * "_" * string(parsed_args["lossType"]) *"_"*string(parsed_args["lossParams"])* "_)_ADAM_" * string(parsed_args["lr"]) * "_" * string(parsed_args["decay"]) * "_" *
		string(Dates.format(today(), "yyyymmdd")) * "_DO" * string(parsed_args["pDrop"]) * "_" * string(seed) * "_" * string(seed) * "_" * string(parsed_args["factory"]) * "_fold" * string(parsed_args["kFold"]) * "_dev_" *
		string(parsed_args["cr_deviation"]) * "_feat_" * string(parsed_args["cr_features"]) * "_" * string(split(parsed_args["data"], "/")[end-1]) * "_norm_" * string(norm)

	mkpath("run/HL_" * endString)
println(2)
	# save the hyper-parameters
	LearningPi.saveHP(
		endString,
		parsed_args["lr"],
		parsed_args["decay"],
		hMLP,
		opt,
		lt,
		fmt,
		dt,
		loss,
		seed,
		seed,
		parsed_args["stepSize"],
		parsed_args["nodes_number"],
		parsed_args["block_number"],
		iMLP,
		fMLP,
		data,
		string(factory))

	open("run/HL_" * endString * "/dataset.json", "w") do f
		JSON.print(f, corpus_info)  # parse and transform data
	end

	act(x) = -softplus(x)
	# construction of the neural network model
	#final_A = (factory == LearningPi.cpuGAinstanceFactory()) ? act : identity
	final_A = identity
	println(lt)
	nn = create_model(lt, size_nn_input(fmt), hMLP, size_nn_output(lt), Flux.relu, seed, iMLP, fMLP, parsed_args["block_number"], parsed_args["nodes_number"], parsed_args["pDrop"], dt, 0.0001, norm; final_A) |> gpu
println(3)
	#if lt == LearningPi.learningMultiPredSampleFeat()
	#	LearningPi.initialize_model(nn,dS)
	#end 
	
	# train the model
	train(parsed_args["maxEp"][1], dS, nn, opt, loss; printEpoch = 100, endString = endString, lt = lt, dt = dt, seed) 
end


function getFactory(factory::String)
	if factory == "cpuCWLinstanceFactory"
		return LearningPi.cpuCWLinstanceFactory()
	end
	if factory == "cpuGAinstanceFactory"
		return LearningPi.cpuGAinstanceFactory()
	end
	if factory == "cpuMCNDinstanceFactory"
		return LearningPi.cpuMCNDinstanceFactory()
	end
	if factory == "cpuMCNDinstanceFactory"
		return LearningPi.cpuGAinstanceFactory()
	end
	if factory == "gpuMCNDinstanceFactory"
		return LearningPi.gpuMCNDinstanceFactory()
	end
	if factory == "cpuGAinstanceFactory"
		return LearningPi.cpuGAinstanceFactory()
	end

end


"""
lossFromParams(lt::String})

Construct a loss object taking as inputs a string for the loss type and a vector for the loss parameters.

rguments:
- `lt`: a string to indicate the learning type. The possible choices are: learningTransformer,learningSampleTransformer,learningMultiPredTransformer

"""
function getLT(lt::String)
	if lt == "learningTransformer"
		return LearningPi.learningTransformer()
	elseif lt == "learningSampleNair"
		return LearningPi.learningSampleNair()
	elseif lt == "learningSampleGasse"
		return LearningPi.learningSampleGasse()
	elseif lt == "learningSampleTransformer"
		return LearningPi.learningSampleTransformer()
	elseif lt == "learningMultiPredTransformer"
		return LearningPi.learningMultiPredTransformer()
	elseif lt == "learningMultiPredSampleFeat"
		return LearningPi.learningMultiPredSampleFeat()
	elseif lt == "learningMultiPredSample"
		return LearningPi.learningMultiPredSample()
	elseif lt == "learningSampleOutside"
		return LearningPi.learningSampleOutside()
	elseif lt == "learningBM"
		return LearningPi.learningBM()
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
function lossFromParams(lossType::String, lossParams::Float32)
	if lossType == "MSELoss"
		return LearningPi.loss_mse()
	end
	if lossType == "HingeLoss"
		return LearningPi.loss_hinge()
	end
	if lossType == "LRloss"
		return LearningPi.loss_LR()
	end
	if lossType == "gpuLRloss"
		return LearningPi.loss_LR_gpu()
	end
	if lossType == "MultiPredLRloss"
		return LearningPi.loss_multi_LR(lossParams)
	end
	if lossType == "GAPcloseLoss"
		return LearningPi.loss_GAP_closure()
	end
	if lossType == "GAPloss"
		return LearningPi.loss_GAP()
	end
end

function size_nn_input(fmt::LearningPi.cr_features_matrix)
	return 20
end

function size_nn_input(fmt::LearningPi.without_cr_features_matrix)
	return 12
end

function size_nn_input(fmt::LearningPi.lr_features_matrix)
	return 16
end

function size_nn_output(lt::LearningPi.learningType)
	return 1
end

function size_nn_output(lt::LearningPi.learningSampleOutside)
	return 2
end

main(ARGS)

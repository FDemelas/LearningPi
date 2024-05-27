using ArgParse
using LearningPi
using Random: MersenneTwister
using Flux
using Dates
using JSON

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
		"--MLP"
		nargs = '*'
		default = Any["no_arg_given"]
		help = "number of nodes in each hidden layer for the MLP"
		"--maxEp"
		nargs = '?'
		arg_type = Int64
		default = 1000
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
		default = "../data/"
		help = "dataset folder for the training"
		"--normalizeEdges"
		arg_type = Bool
		default = true
		help = "If normalize or not the edge features of the graph"
		"--factory"
		arg_type = String
		default =  "cpuMCNDinstanceFactory"
		help=""
	end

	parsed_args = parse_args(args, s)

	data = parsed_args["data"]

	lt = LearningPi.learningMLP()
	dt = LearningPi.cr_deviation()
	fmt = LearningPi.cr_features_matrix()
	
	factory = getFactory(parsed_args["factory"])

	h = [parse(Int64, parsed_args["MLP"][i]) for i in 1:length(parsed_args["MLP"])]

	corpus_info, dS = createKfold(lt,fmt, data, maxInst, parsed_args["seed"]; factory = factory, k = parsed_args["kFold"])

	lossParam = lossFromParams(parsed_args["lossType"][1], parsed_args["lossParams"])
	loss = createLoss(lossParam)

	# construct the optimizer with exponential decay
	lrDecay = Flux.Optimise.ExpDecay(1, parsed_args["decay"], parsed_args["stepSize"], 1e-8, 1)
	opt = Flux.Optimiser(RAdam(parsed_args["lr"]), ClipNorm(1), lrDecay)

	endString = "MLP_"*string(parsed_args["MLP"]) * "_" * string(parsed_args["lossType"]) * "(" * string(parsed_args["lossParams"]) * ")_ADAM_" * string(parsed_args["lr"]) * "_" * string(parsed_args["decay"]) * "_" *
		Dates.format(today(), "yyyymmdd") * "_" * string(parsed_args["seed"]) * "_" * string(parsed_args["seed"]) * "_MCNDfactory" * "_fold" * string(parsed_args["kFold"])*"_"*string(split(data,"/")[end-1])
        
	mkpath("run/HL_" * endString)
	open("run/HL_" * endString * "/dataset.json", "w") do f
                JSON.print(f, corpus_info)  # parse and transform data
    end


	LearningPi.saveHP(endString, parsed_args["lr"], parsed_args["decay"], h, opt, lt, lossParam, loss, parsed_args["seed"], parsed_args["seed"], parsed_args["stepSize"])

	#acct(x)=-Flux.softplus(x)	
	#final_A=contains(data,"GA") ? acct : identity
	
	final_A=identity
	nn = create_model(lt, LearningPi.sizeFeatures(lt, dS), h, 1, relu, parsed_args["seed"];final_A)
	
	train(parsed_args["maxEp"], dS, nn, opt, lossParam; printEpoch = 100, endString = endString, lt,dt,seed=parsed_args["seed"])

end

"""
lossFromParams(lossType::String,lossParams::Vector{Float32})

Construct a loss object taking as inputs a string for the loss type and a vector for the loss parameters.

#Arguments:
- `lossType`: a string to indicate the loss type. The possible choices are: hingeMSE,ZKRloss,SVR,hingeZKR,pertZKRadd and SVR
- `lossParams`: a vector of float with the same size of the parameters required for the chosen loss,
			  see the documentation of the loss strucures to know more about it.
"""
function lossFromParams(lossType::String, lossParams::Vector{Float32})
	if lossType == "GAPloss"
		return LearningPi.GAPloss(lossParams[1])
	end
	if lossType == "LRloss"
		return LearningPi.LRloss(lossParams[1])
	end
	if lossType == "GPULRloss"
		return LearningPi.GPULRloss(lossParams[1])
	end
	if lossType == "GAPcloseLoss"
		return LearningPi.GAPcloseLoss(lossParams[1])
	end
	if lossType == "MSE"
		return LearningPi.MSEloss(lossParams[1])
	end
	if lossType == "HingeLoss"
		return LearningPi.HingeLoss(LearningPi.perturbationAdditive(lossParams[1], lossParams[2], MersenneTwister(1), lossParams[3]))
	end
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
        if factory == "cpuGAinstanceFactory"
                return LearningPi.cpuGAinstanceFactory()
        end

end



main(ARGS)

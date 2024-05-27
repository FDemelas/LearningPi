# currentMetrics dictionary, used to memorize 
# different values for the current epoch
currentMetrics = Dict(
	"loss training" => 0.0,
	"GAP training" => 0.0,
	"GAP CR training" => 0.0,
	"loss validation" => 0.0,
	"GAP validation" => 0.0,
	"GAP CR validation" => 0.0,
	"loss test" => 0.0,
	"GAP test" => 0.0,
	"GAP CR test" => 0.0,
	"forward time" => 0.0,
)

is_min_SP(ins::abstractInstance) = true

is_min_SP(ins::cpuInstanceGA) = false

# best metrics dictionary, used to memorize 
# different values for the best models found so far
bestModels = Dict{String,Any}("loss training"=>Chain(),"GAP training"=>Chain(),"GAP CR training"=>Chain(), "loss validation"=>Chain(),"GAP validation"=>Chain(),"GAP CR validation"=>Chain())

"""
function validation(currentMetrics::Dict, valSet, nn,loss::abstractLoss,  lt::learningType,dt::abstract_deviation)

# Arguments:
- `currentMetrics`: a dictionary of Float 
- `valSet` : a vector of gnn_dataset that correspond to the validation set.
- `nn` : a neural network model.
- `loss`: a structure with the parameters of the loss.
		   For other details of the parameters of a certain loss see the definition of the particular structure of the loss.
- `loss`: loss function.
- `lt`: learning type object.       
- `dt`: deviation type (0 or dual of the CR)        

This function compute different metrics over the validation set.
The values are memorized in the dictionary.
"""
function validation(currentMetrics::Dict, valSet, nn, loss, lt::learningType,dt::abstract_deviation)
	nInstVal = length(valSet)
	Flux.testmode!(nn)
	device = get_device(loss)
        act_o = is_min_SP(valSet[1].instance) ? identity : (f(x) = -softplus(-x))
	for example in valSet
		feat = model_device(example.features)
		π = act_o(nn(feat)) #prediction(nn, feat, example, lt,dt)

		v = loss(device(π); example)

		objPred = sub_problem_value(π, v, example, loss)
		objGold = example.gold.objLR
		sp_sign = is_min_SP(example.instance) ? 1 : -1

		currentMetrics["loss validation"] += sp_sign * v / (nInstVal)
		currentMetrics["GAP validation"] += sp_sign * (objGold - objPred) / (objGold * nInstVal) * 100
		currentMetrics["GAP CR validation"] += (1 - sp_sign * (objGold - objPred) / (objGold - example.linear_relaxation)) * 100 / nInstVal
	end
end

#function validation(currentMetrics::Dict, valSet, nn, loss, lt::learningMultiPredTransformer)
#	nInstVal = length(valSet)
#	for example in valSet
#
#		feat = model_device(example.features)
#		πs = nn(feat)
#		π = πs[(size(πs)[1]-feat.gdata.u[2][1]+1):end, :]
#
#		v = loss(πs; example)
#		objPred = sub_problem_value(π, v, example, loss)
#		objGold = example.gold.objLR
#		sp_sign=is_min_SP(example.instance) ? 1 : -1
#
#
#		currentMetrics["loss validation"] +=sp_sign* v / (nInstVal)# *lengthLM(example.instance))
#		currentMetrics["GAP validation"] += sp_sign*(objGold - objPred) / (objGold * nInstVal) * 100
#		currentMetrics["GAP CR validation"] += (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) * 100 / nInstVal
#	end
#end

"""
function testAndPrint(currentMetrics::Dict,testSet,nn,loss,loss,lt::learningType)

# Arguments:
- `currentMetrics`: a dictionary of Float 
- `testSet` : a vector of hingeExamples_instance that correspond to the test set.
- `nn` : a neural network model.
- `loss`: a structure with the parameters of the loss.
			   For other details of the parameters of a certain loss see the definition of the particular structure of the loss.
- `lt`: learning type object.     

This function compute different metrics over the validation set.
The values are memorized in the dictionary and print them in the standard output.
"""
function testAndPrint(currentMetrics::Dict, testSet, nn, loss, lt::learningType,dt::abstract_deviation)

	nInstTest = length(testSet)

	Flux.testmode!(nn)
	device = get_device(loss)
	
	times=[]
        act_o = is_min_SP(testSet[1].instance) ? identity : (f(x) = -softplus(-x))

	for example in testSet
		t=0
		t0=time()
		CR(example.instance)
		t+=time()-t0

		feat = model_device(example.features)

		t0=time()
		π = act_o(nn(feat)) #prediction(nn, feat, example, lt,dt)
		v = loss(device(π); example)
		t += time()-t0

		objPred = sub_problem_value(π, v, example,loss)
		objGold = example.gold.objLR
		sp_sign = 1
		
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
function compareWithBests(currentMetrics::Dict,bestMetrics::Dict,nn,endString::String)

# Arguments:
- `currentMetrics` : a dictionary of Float 
- `bestMetrics` : a dictionary of Float 
- `nn` : a neural network
- `endString` : a string used to memorize the best models

This function compare all the values in `bestMetrics` with the ones in `currentMetrics` (that corresponds to the same key).
If some value in `currentMetrics` is better, then we update the correspondent value in `bestMetrics` and we save the model
in a bson file.
"""
function compareWithBests(currentMetrics::Dict, bestMetrics::Dict, nn, endString)
	let nn = cpu(get_model(nn))

		#if currentMetrics["loss training"] < bestMetrics["loss training"]
		#	bestMetrics["loss training"] = currentMetrics["loss training"]
		#	@save "/data-a40/demelas/run/HL_" * endString * "/bestTrainLoss_" * endString * ".bson" nn
		#end
		#if currentMetrics["loss validation"] < bestMetrics["loss validation"]
		#	bestMetrics["loss validation"] = currentMetrics["loss validation"]
		#	@save "run/HL_" * endString * "/bestValLoss_" * endString * ".bson" nn
		#end
		if currentMetrics["GAP validation"] < bestMetrics["GAP validation"]
			bestMetrics["GAP validation"] = currentMetrics["GAP validation"]
			@save "run/HL_" * endString * "/bestValGap_" * endString * ".bson" nn
		end
		#if currentMetrics["GAP CR validation"] > bestMetrics["GAP CR validation"]
		#	bestMetrics["GAP CR validation"] = currentMetrics["GAP CR validation"]
		#end
		#if currentMetrics["GAP CR training"] > bestMetrics["GAP CR training"]
		#	bestMetrics["GAP CR training"] = currentMetrics["GAP CR training"]
		#end
		if currentMetrics["GAP training"] < bestMetrics["GAP training"]
			bestMetrics["GAP training"] = currentMetrics["GAP training"]
#		for key in keys(bestModels)
#			if contains(key,"CR") ? currentMetrics[key] > bestMetrics[key] : currentMetrics[key] < bestMetrics[key]
#				bestModels[key] = deepcopy(nn)
#				bestMetrics[key] = currentMetrics[key]
#			end
		end
	end
end

"""
function print_best_models(endString::String, bestModels::Dict)

#Arguments:
- `endString`: Path where save the models 
- `bestModels`: Dictionary of the best models (w.r.t different metrics) found so far

Print in a file BSON, located in the folder `endString` the best model found so far 
"""
function print_best_models(endString::String, bestModels::Dict)
	for key in keys(bestModels)
		sKey=""
		for s in split(key," ")
			sKey *= s[1:min(3,end)]
		end
		sKey*="_"
		@save "run/HL_" * endString * "/"* sKey * endString * ".bson" nn=bestModels[key]
	end
end	

"""
function printMetrics(currentMetrics::Dict)

# Arguments:
- `currentMetrics`: a dictionary of Float

This function takes as input the dictionary of the metrics and print the values
associated to training and validation sets.    
"""
function printMetrics(currentMetrics::Dict)
	println("loss training: ", currentMetrics["loss training"])
	println("loss validation: ", currentMetrics["loss validation"])
	println("GAP training: ", currentMetrics["GAP training"])
	println("GAP validation: ", currentMetrics["GAP validation"])
	println("GAP CR training: ", currentMetrics["GAP CR training"])
	println("GAP CR validation: ", currentMetrics["GAP CR validation"])
	println()
end

"""
function printBests(bestMetrics::Dict, path::String)

# Arguments:
- `bestMetrics`: a dictionary of float that contains the best values
			   find in the training for altypel the considered metrics.
- `path`: location where print the results in a file.				

Takes as input the dictionary of the best metrics and print the values in standard output and in a file defined by the `path`.                
"""
function printBests(bestMetrics::Dict, path::String)
	f = open(path, "w")
	println(f, "------------------------------")
	println(f, "--------BEST-RESULTS----------")
	println(f, "------------------------------")
	println(f, "Best loss training: ", bestMetrics["loss training"])
	println(f, "Best loss validation: ", bestMetrics["loss validation"])
	println(f, "Best GAP training: ", bestMetrics["GAP training"])
	println(f, "Best GAP validation: ", bestMetrics["GAP validation"])
	println(f, "Best GAP CR training: ", bestMetrics["GAP CR training"])
	println(f, "Best GAP CR validation: ", bestMetrics["GAP CR validation"])
	println(f)
	close(f)

	println("------------------------------")
	println("--------BEST-RESULTS----------")
	println("------------------------------")
	println("Best loss training: ", bestMetrics["loss training"])
	println("Best loss validation: ", bestMetrics["loss validation"])
	println("Best GAP training: ", bestMetrics["GAP training"])
	println("Best GAP validation: ", bestMetrics["GAP validation"])
	println("Best GAP CR training: ", bestMetrics["GAP CR training"])
	println("Best GAP CR validation: ", bestMetrics["GAP CR validation"])
	println()
end

"""
function saveHP(endString::String,lr::Float32,decay::Float32,h::Vector{Int64}, opt, lt::learningType, loss,seedNN::Int64, seedDS::Int64, stepSize::Int64)
	
# Arguments:    
- `endString`: a string used as name for the output file.
- `lr`: learning rate of the algorithm.
- `decay`: decay for the learning rate.
- `h`: a list of #(hidden layers), each component of h contains the number of nodes in
	 the associated hidden layer.
- `opt`: optimizer
- `lt`: learning type object.     
- `loss`: loss function.
- `seedDS`: random seed for the dataset generation.
- `seedNN`: random seed for the neural network parameters.
- `stepSize`: the step size for the decay scheduler of the optimizer

This function memorize all this hyper parameters in a JSON file.     
"""
function saveHP(endString::String, lr::Float32, decay::Float32, h::Vector{Int64}, opt, lt::learningType, loss, seedNN::Int64, seedDS::Int64, stepSize)
	hyperParams = Dict(
		"learnign rate" => lr,
		"decay" => decay,
		"step size" => stepSize,
		"optimizer" => opt,
		"model" => h,
		"learning type" => string(lt),
		"loss" => string(loss),
		"seed neural network" => seedNN,
		"seed Data Set" => seedDS,
		"date" => Dates.format(today(), "yyyy-mm-dd"),)
	fileName = "run/HL_" * endString * "/config.json"
	open(fileName, "w") do f
		JSON.print(f, hyperParams)  # parse and transform data
	end
end

"""
function saveHP(endString::String,lr::Float32,decay::Float32,h::Vector{Int64},opt,lt::learningType,loss,seedNN::Int64,seedDS::Int64,stepSize::Int64)
	
# Arguments:    
- `endString`: a string used as name for the output file.
- `lr`: learning rate of the algorithm.
- `decay`: decay for the learning rate.
- `h`: a list of #(hidden layers), each component of h contains the number of nodes in
	 the associated hidden layer.
- `opt`: optimizer
- `lt`: learning type object.     
- `fmt`:
- `dt`:
- `loss`: loss function.
- `seedDS`: random seed for the dataset generation.
- `seedNN`: random seed for the neural network parameters.
- `stepSize`: the step size for the decay scheduler of the optimizer
- `nodes_number`: size (number of nodes) in the hidden reprensentation between each layer
- `block_number`: number of blocks in the model
- `hI`: sizes of Dense layers in the first part, where the nodes features are sent in the hidden space
- `hF`: sizes of Dense layers in the final
- `dataPath`: path to the instances used in the dataset
- `factory`: instance factory type

This function memorize all this hyper parameters in a JSON file.     
"""
function saveHP(endString::String, lr::Float32, decay::Float32, h::Vector{Int64}, opt, lt::learningGNN,fmt::abstract_features_matrix,dt::abstract_deviation, loss, seedNN::Int64, seedDS::Int64, stepSize, nodes_number, block_number, hI, hF, dataPath, factory)
	hyperParams = Dict(
		"learnign rate" => lr,
		"decay" => decay,
		"step size" => stepSize,
		"optimizer" => opt,
		"model" => h,
		"learning type" => string(lt),
		"features matrix type" => string(fmt),
		"deviation type" => string(dt),
		"loss" => string(loss),
		"seed neural network" => seedNN,
		"seed Data Set" => seedDS,
		"date" => Dates.format(today(), "yyyy-mm-dd"),
		"nodes number hr" => nodes_number,
		"block number" => block_number,
		"encoder sizes" => hI,
		"decoder sizes" => hF,
		"data path" => dataPath,
		"factory" => factory,
	)
	fileName = "run/HL_" * endString * "/config.json"
	open(fileName, "w") do f
		JSON.print(f, hyperParams)  # parse and transform data
	end
end

"""
function get_parameters(nn, lt::learningType, f = 0)

- `nn` : neural network
- `lt` : learning type
- `f` :  ???

Returns the model parameters of `nn` in the case in which nn belsong to `lt` learning type.
"""
function get_parameters(nn, lt::learningType, f = 0)
	return Flux.params(nn)
end

"""
function train(maxEp::Int64, dS::Corpus, nn, opt::Optimiser, loss::abstractLoss; printEpoch::Int64=10, endString::String, lt::learningType, dt::abstract_deviation,seed::Int64,bs::Int64=1)

# Arguments:

- `maxEp`: the maximum number of epochs for the learning algorithm.
- `dS`: the Corpus structure that contains the training, validation and test sets.
- `nn`: the neural network model.
- `opt`: the optimizer used for the training.
- `loss`: a structure that contains the parameters α and β of the loss.
- `printEpoch`: the number of epochs in which print the metrics of the training.
- `endString`: the string used to memorize the output files as best models and tensorboard logs.
- `dt`: deviation type, it could deviate from zero or the duals of the continuous relaxation.
- `lt`: learning type
- `seed`: random seed for the random generators
- `bs` batch size

This function performs the learning with the provided inputs and save the best models in a bson file.
"""
function train(maxEp::Int64, dS::Corpus, nn, opt::Optimiser, loss::abstract_loss; printEpoch::Int64 = 10, endString::String, lt::learningType, dt::abstract_deviation,seed::Int64,bs::Int64=1)
	# Create training, validation and test set
	trainSet = dS.train.examples_list
	valSet = dS.val.examples_list
	testSet = dS.test.examples_list


	# best metrics dictionary
	# all values are initialized to infinity
	bestMetrics = Dict(
		"loss training" => Inf,
		"GAP training" => Inf,
		"GAP CR training" => -Inf,
		"loss validation" => Inf,
		"GAP validation" => Inf,
		"GAP CR validation" => -Inf,
	)
	
	rng = MersenneTwister(seed)
	lg = TBLogger("run/HL_" * endString, min_level = Logging.Info)
	with_logger(lg) do
		for epoch in 1:maxEp
			shuffle!(rng,trainSet)

			for key in keys(currentMetrics)
				currentMetrics[key] = 0.0
			end

			t0=time()	
			forwardBackward(trainSet, nn, currentMetrics, opt, loss, epoch, lt, dt)
			#println("Training time",time()-t0)


			#t0=time()	
			validation(currentMetrics, valSet, nn, loss, lt, dt)
			#println("Validation time",time()-t0)

			#t0=time()	
			compareWithBests(currentMetrics, bestMetrics, nn, endString)
			#println("Comparison time",time()-t0)

			#t0=time()	
			if epoch % printEpoch == 0
				println(epoch)
				printMetrics(currentMetrics)
			end
			#println("Comparison time",time()-t0)

			@info " Train " loss = currentMetrics["loss training"] log_step_increment = 0
			@info " Train " GAP = currentMetrics["GAP training"] log_step_increment = 0
			@info " Validation " loss = currentMetrics["loss validation"] log_step_increment = 0
			@info " Validation " GAPCR = currentMetrics["GAP CR validation"] log_step_increment = 0
			@info " Validation " GAP = currentMetrics["GAP validation"]
		end

		print_best_models(endString,bestModels)

		printBests(bestMetrics, "run/HL_" * endString * "/results.dat")

		println("test with the final model (best training loss model)")
		testAndPrint(currentMetrics, testSet, nn, loss, lt,dt)
		
		@load "run/HL_" * endString * "/bestValGap_" * endString * ".bson" nn
		nn_test = model_device(load_model(nn,lt))
		
		println("test with the best validation gap model")
		testAndPrint(currentMetrics, testSet, nn_test, loss, lt,dt)
	end
end

"""
	Abstract type for the dataset
"""
abstract type abstract_dataset end

""" 
	Abstract type for an element of a dataset 
"""
abstract type abstract_example end

"""
	Corpus structure contains the three datasets: training, validation and test set 
"""
struct Corpus
	train::abstract_dataset
	val::abstract_dataset
	test::abstract_dataset
end

"""
# Arguments:
- `lt`: the learning type.
- `directory`: a list of paths to the instances.
- `maxInstance`: the maximum number of instances that we wat consider in the provided directory. By default is equal to `-1`, that means consider all the instances in the directory.
- `factory`: type of instance, the possibilities in this moment are cpuMCNDinstanceFactory() (that is Multi-Commodity Network-Design instances) and cpuCWLinstanceFactory() (for the Bin Packing instances).

Create the dataset for the provided (general) learning type.
return a dataSet structure of a proper type.
"""
function createDataSet(lt::learningType, fmt::abstract_features_matrix, directory, maxInstance::Int64 = -1, factory::abstractInstanceFactory = cpuMCNDinstanceFactory())
	dS = createEmptyDataset(lt)

	for (index, file) in enumerate(directory)
		push!(dS.examples_list, @views create_example(lt, file, factory, fmt))

		if (maxInstance >= 0) && (index > maxInstance)
			break
		end
	end
	return dS
end

"""
# Arguments:

- `featType` : the type of the features instance.
- `folder` : the path to the directory that contains the json files that defines the instances (and the associated features and labels).
- `maxInstance` : a vector with three components. 
   that say how many instances take for the training/validation/test set.
- `seed` : a random seed used to select which instaces consider in the training/validation/test sets.
- `factory`: type of instance.
- `pTrain` : The percentage of training instances in the provided folder.
- `pVal` : The percentage of validation instances in the provided folder.

Create a Corpus, that is a structure with three DataSet field for training,validation and test dataset.
Note: The percentage for the test set will be 1-pTrain-pVal. It is important to select the percentage of training and validation in such a way that pTrain+pVal<1 and both will be non-negative.
"""
function createCorpus(featType::learningType, fmt::abstract_features_matrix, folder::String, maxInstance::Vector{Int64} = [-1, -1, -1], seed::Int64 = 0; factory::abstractInstanceFactory = cpuMCNDinstanceFactory(), pTrain = 0.8, pVal = 0.1)
	directory = folder .* readdir(folder)

	shuffle!(directory)

	eT = round(Int64, length(directory) * pTrain)
	eV = round(Int64, length(directory) * (pTrain + pVal))
	trainInstances = @views directory[1:eT]
	valInstances = @views directory[eT+1:eV]
	testInstances = @views directory[eV+1:end]

	corpus_info = Dict(
		"train" => trainInstances,
		"val" => valInstances,
		"test" => testInstances)

	# create training set
	train = createDataSet(featType, fmt, trainInstances, maxInstance[1], factory)

	# create validation set
	val = createDataSet(featType, fmt, valInstances, maxInstance[2], factory)

	# create test set
	test = createDataSet(featType, fmt, testInstances, maxInstance[3], factory)

	return corpus_info, Corpus(train, val, test)
end


"""
# Arguments:
- `featType` : the type of the features instance.
- `folder` : the path to the directory that contains the json files that defines the instances (and the associated features and labels).
- `maxInstance` : a vector with three components. 
   that say how many instances take for the training/validation/test set.
- `seed` : a random seed used to select which instaces consider in the training/validation/test sets.
- `factory`: type of instance.
- `k`: the fold that we want select as test set. Note: 1 <= k <= 10.

Create a Corpus, that is a struct with three DataSet field for training/validation/test set
"""
function createKfold(featType::learningType, fmt::abstract_features_matrix, folder::String, maxInstance::Vector{Int64} = [-1, -1, -1], seed::Int64 = 0; factory::abstractInstanceFactory = cpuMCNDinstanceFactory(), k::Int64 = 1)
	directory = folder .* readdir(folder)

	rng = Random.MersenneTwister(seed)
	shuffle!(rng, directory)

	leng1set = floor(Int64, length(directory) / 10)

	kfold = [directory[i*leng1set+1:(i+1)*leng1set] for i in 0:9]

	testInstances = kfold[k]
	deleteat!(kfold, k)

	valIdx = rand(MersenneTwister(seed + k), 1:9, 1)[1]
	valInstances = kfold[valIdx]
	deleteat!(kfold, valIdx)

	trainInstances = reduce(vcat, kfold)

	corpus_info = Dict(
		"train" => trainInstances,
		"val" => valInstances,
		"test" => testInstances)

	# create training set
	train = createDataSet(featType, fmt, trainInstances, maxInstance[1], factory)

	# create validation set
	val = createDataSet(featType, fmt, valInstances, maxInstance[2], factory)

	# create test set
	test = createDataSet(featType, fmt, testInstances, maxInstance[3], factory)

	return corpus_info, Corpus(train, val, test)
end

"""
# Arguments:
- `example`: the current example (dataset point).
- `objPred`: the current obective for the example.
- `objGold`: the optimal value of the Lagrangian Dual.
- `nInst`: the number of the instances in the set.

compute the GAP of the instance in the `example` using the predicted objective `objPred`.
"""
function gap(example::abstract_example, objPred::Real, objGold::Real, nInst::Real)
	if typeof(example.instance) <: instanceGA
		return (objPred - objGold) / (objGold * nInst) * 100
	end
	return (objGold - objPred) / (objGold * nInst) * 100
end


"""
# Arguments:
- `example`: the current example (dataset point).
- `objPred`: the current obective for the example.
- `objGold`: the optimal value of the Lagrangian Dual.
- `nInst`: the number of the instances in the set.

compute the closure GAP of the instance in the `example` using the predicted objective `objPred`.
The closure is w.r.t. the value of the Lagrangian Sub-Problem, solved with the dual variables of the continuous relaxation.
"""
function gap_closure(example::abstract_example, objPred::Real, objGold::Real, nInst::Real)
	if typeof(example.instance) <: instanceGA
		return (1 - (objPred - objGold) / (example.linear_relaxation - objGold)) * 100 / nInst
	end
	return (1 - (objGold - objPred) / (objGold - example.linear_relaxation)) * 100 / nInst
end

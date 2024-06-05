"""
Structure to encode the examples for the training when we whant use a GNN model.

# Fields:
- `instance`: an instance,
- `features`: the features associated to the instance,	
- `gold`: the labels,
- `linear_relaxation`: the Lagrangian Subproblem value associated to the dual variable of the continuous relaxation.
"""
mutable struct example_gnn <: abstract_example
	instance::Any
	features::Any
	gold::Any
	linear_relaxation::Any
end

"""
Structure to encode the dataset composed by `example_gnn`.
"""
mutable struct gnn_dataset <: abstract_dataset
	examples_list::Vector{example_gnn}
end

"""
# Arguments:
- `lt`: learning Multi Layer Perceptron type.

Create an empty dataset for the Graph Neural network learning type.
"""
function createEmptyDataset(lt::learningGNN)
	return gnn_dataset([])
end

"""
# Arguments:
- `lt`: learning Type, this function works for all the learning types that use a graph representation of the instance,
- `fileName`: path to the json that contains all the information to construct the learning sample starting from an instance, its features and the labels,
- `factory`: instance factory, it works for all the factory.

Returns an `gnnExample_instance` with all the information useful for the training.
"""
function create_example(lt::learningGNN, fileName::String, factory::abstractInstanceFactory, fmt::abstract_features_matrix)
	instance, gold, featuresObj = dataLoader(fileName, factory)
	features = featuresExtraction(lt, featuresObj, instance, fmt)
	linear_relaxation = featuresObj.objLR
	return example_gnn(instance, features, gold, linear_relaxation)
end

"""
# Arguments:
		- `lt`:learning type, it should be a sub-type of `learningGNN`, 
		- `directory`: the path to the directory containing instances,
		- `maxInstance`: maximum instance number, 
		- `factory`: instance factory, generic sub-type of `abstractInstanceFactory`.
	
	Create and return a dataset for the provided learning type `lt`, considering `maxInst` instances of the factory `factory`, contained in `directory`  
"""
function createDataSet(lt::learningGNN, fmt::abstract_features_matrix, directory::String,  factory::abstractInstanceFactory, maxInstance::Int64 = -1 )
	l = enumerate(readdir(directory))
	if (maxInstance >= 0) && (length(l) > maxInstance)
		l = @views l[1:maxInstance]
	end

	return gnn_dataset(#@showprogress
		map((index, file) -> create_example(lt, directory * file, factory, fmt), l))
end

"""
# Arguments:
- `lt`: learning type, it should be a sub-type of `learningGNN`,
- `dS`: a dataset.

Return the size of the features matrix.
"""
function sizeFeatures(lt::learningGNN, dS)
	return size(dS.train.examples_list[1].features.ndata.x, 1)
end

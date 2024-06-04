"""
Abstract type to type the functions that should work with all the type of models and features encoding
"""
abstract type learningType end

"""
Abstract type to type the functions that should work with Graph-Neural-Networks.
"""
abstract type learningGNN <: learningType end

"""
Abstract type to type the functions that should work with Graph-Neural-Networks.
This abstract type was originally tought to be used for the models that use the Block Architecture here implemented.
In the current implementation it coincides more or less to `learningGNN`.
"""
abstract type learningBlockGNN <: learningGNN end

"""
Abstract type for the Sampling functions.
"""
abstract type AbstractSampler end

"""
Abstract type for the Neural Networks models.
"""
abstract type AbstractModel end

"""
# Arguments:
- `nn`: a model, sub-type of `learningType`.

implementation for the function that.
The general rule is that `nn` is a model and we can directly call the function `Flux.params`
This abstract implementation will be soon removed.
"""
function get_parameters(nn::learningType)
	return Flux.params(nn)
end

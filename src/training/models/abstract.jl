"""
abstract learning type to type the functions that should work with all the type of models and features encoding
"""
abstract type learningType end

abstract type learningGNN <: learningType end

abstract type learningBlockGNN <: learningGNN end

abstract type AbstractSampler end

abstract type AbstractModel end

function get_parameters(nn::learningType)
        return Flux.params(nn)
end


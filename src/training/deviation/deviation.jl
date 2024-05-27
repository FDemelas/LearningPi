"""
abstract type abstract_deviation end

Abstract type to handle the deviation vector, i.e. the starting point from which our model produce an additive activation. 
"""
abstract type abstract_deviation end

"""
struct cr_deviation end

Type to use as deviation vector (i.e. the starting point from which our model produce an additive activation) the dual variables associated to the relaxed
constraints in the optimal solution of the Continuous Relaxation. 
"""
struct cr_deviation <: abstract_deviation end

"""
struct zero_deviation end

Type to use as deviation vector (i.e. the starting point from which our model produce an additive activation) the all zeros vector. 
"""
struct zero_deviation <: abstract_deviation end

"""
function get_λ(x)

#Arguments:
- `x`: the bipartite-graph representation of the instance

Returns the dual variables associated to the dualized constraints in the optimal solution of the continuous relaxation,
taking the good components from the nodes features matrix in the bipartite-graph representation.
"""
function get_λ(x)
	@views y = (x.ndata.x)[1,1:prod(x.gdata.u)] 
	return model_device(reshape(y, cpu(x.gdata.u)...))
end


"""
function deviationFrom(x,_::cr_deviation)

#Arguments:
- `x`: the bipartite-graph representation of the instance

For the `cr_deviation` it returns the dual variables associated to the dualized constraints in the optimal solution of the continuous relaxation,
taking the good components from the nodes features matrix in the bipartite-graph representation.
"""
function deviationFrom(x,_::cr_deviation)
	return (get_λ(x))
end	


"""
function deviationFrom(x,_::zero_deviation)

#Arguments:
- `x`: the bipartite-graph representation of the instance

For the `zero_deviation` it returns an all-zeros vector with the correct size.
The size will be the same as the dual variables associated to the dualized constraints in the optimal solution of the continuous relaxation,
taking the good components from the nodes features matrix in the bipartite-graph representation.
"""
function deviationFrom(x,_::zero_deviation)
	return CUDA.zeros(Float32,size(x))
end

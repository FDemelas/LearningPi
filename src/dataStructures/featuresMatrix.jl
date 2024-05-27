"""
abstract type `abstract_features_matrix`

Abstract type for the construction of the nodes-features matrix associated to the bipartite graph representation of the instance.
"""
abstract type abstract_features_matrix end

"""
struct `cr_features_matrix`

with this choice the features matrix considers the informations related to the continuous relaxation of the instance.
"""
struct cr_features_matrix <: abstract_features_matrix
	edge_features::Bool
	β::Int
	cr_features_matrix(edge_features=false,β=-1) = (
		new(edge_features,β)
	)
end


"""
struct `lr_features_matrix`

with this choice the features matrix considers the informations related to the continuous relaxation of the instance.
"""
struct lr_features_matrix <: abstract_features_matrix
	edge_features::Bool
	β::Int
	seed::Int
	lr_features_matrix(edge_features=false,β=-1,seed=-1) = (
		new(edge_features,β,seed)
	)
end

"""
struct `withou_cr_features_matrix`

with this choice the features matrix does not considers the informations related to the continuous relaxation of the instance.
"""
struct without_cr_features_matrix <: abstract_features_matrix 
	edge_features::Bool
	β::Int
	without_cr_features_matrix(edge_features=false,β=-1) = (
		new(edge_features,β)
	)
end

"""
function size_features_constraint(fmt::cr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the constraints. In this case 6.
"""
function size_features_constraint(fmt::cr_features_matrix)
	return 6
end


"""
function size_features_constraint(fmt::lr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the constraints. In this case 6.
"""
function size_features_constraint(fmt::lr_features_matrix)
	return 5
end

"""
function size_features_constraint(fmt::without_cr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the constraints. In this case 4.
"""
function size_features_constraint(fmt::without_cr_features_matrix)
	return 4
end

"""
function size_features_variable(fmt::cr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the variables. In this case 4.
"""
function size_features_variable(fmt::cr_features_matrix)
	return 4
end

"""
function size_features_variable(fmt::lr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the variables. In this case 4.
"""
function size_features_variable(fmt::lr_features_matrix)
	return 3
end

"""
function size_features_variable(fmt::without_cr_features_matrix)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 

returns the size of the features associated to the variables. In this case 2.
"""
function size_features_variable(fmt::without_cr_features_matrix)
	return 2
end

"""
function cr_features(fmt::cr_features_matrix, x, y)

#Arguments:
-`fmt`: feature matrix type (it shoul be cr_features_matrix) 
"""
function get_cr_features(_::cr_features_matrix, x, y, _)
	return Float32[x, y]
end

"""
function cr_features(fmt::lr_features_matrix, x, y)

#Arguments:
-`fmt`: feature matrix type (it shoul be lr_features_matrix) 
"""
function get_cr_features(_::lr_features_matrix, _,_, z)
	return Float32[ z ]
end

"""
function cr_features(fmt::without_cr_features_matrix, _, _)

#Arguments:
-`fmt`: feature matrix type (it shoul be without_cr_features_matrix) 
"""
function get_cr_features(_::without_cr_features_matrix, _, _, _)
	return Float32[]
end

function preprocess_weight(fmt::abstract_features_matrix,value::Real)
	if fmt.edge_features
		if fmt.β <= 0
			return value
		else
			return exp( fmt.β * value )
		end
	else 
		return 1
	end
end

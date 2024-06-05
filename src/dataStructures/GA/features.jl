"""
Features structure for the Generalized Assignment instance.

# Fields:
-`xCR`: primal solution of the Linear Relaxation associated to the variables that associate one items to a pack,
-`λ`: dual solution of the Linear Relaxation associated to the packing constraints,
-`μ`: dual solution of the Linear Relaxation associated to the packing constraints,
-`objCR`: objective value of the Linear Relaxation,
-`xLR`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables that associate one items to a pack (using the dual variables λ of the linear relaxation),
-`objLR`: objective value of the Knapsack Lagrangian Relaxation (using the dual variables λ of the linear relaxation). 
"""
struct featuresGA
	xCR::Matrix{Float32}
	λ::Matrix{Float32}
	μ::Vector{Float32}
	xRC::Matrix{Float32}
	λSP::Matrix{Float32}
	μSP::Vector{Float32}
	objCR::Float32
	xLR::Matrix{Float32}
	objLR::Float32
end

"""
# Arguments:
		- `ins`: instance object, it should be of type instanceGA.

	read the features and returns a features structure.
"""
function create_features(ins::instanceGA)
	objCR, λ, μ, xCR,  λSP, μSP, xRC = CR(ins)

	objLR, xLR = LR(ins, -λ')

	return featuresGA(xCR, -λ', μ, xRC, λSP, μSP, objCR, xLR, objLR)
end

"""
# Arguments:
- `ins`: instance structure, it should be a sub-type of instanceGA,
- `featObj`: features object containing all the characteristics,
- `fmt`: features matrix type.

Construct the matrix of the features for a bipartite-graph representation of the instance.
"""
function features_matrix(ins::instanceGA,featObj, fmt::abstract_features_matrix)
	sizeFC = size_features_constraint(fmt)
	sizeFV = size_features_variable(fmt)
	sizeF = 2 * (sizeFC + sizeFV)
	f = zeros(Float32, (sizeF, ins.J + ins.I + ins.J * ins.I))

	isGeq = false
	isLeq = true
	isDualized = true


	λ0 = zeros(1,ins.I) # randn?
    	xLR0 = zeros(ins.I,ins.J)
	μ0 = zeros(ins.J)

	if ( typeof(fmt) == lr_features_matrix )
		obj, xLR0, _ =  LR(ins,λ0) 
    	μ0 = sum( xLR0,dims=1) .- 1
	end

	current_idx = 1
	for j in 1:ins.I
		f[1:sizeFC, current_idx] = vcat(get_cr_features(fmt,featObj.λ[j], featObj.λSP[j],λ0[j]), Float32[1, isGeq, isLeq, isDualized])
		current_idx += 1
	end

	isInteger = true
	for j in 1:ins.J
		for i in 1:ins.I
			f[sizeFC+1:(sizeFV+sizeFC), current_idx] = vcat(get_cr_features(fmt,featObj.xCR[i,j], featObj.xRC[i,j],xLR0[i,j]), Float32[ ins.p[i,j], isInteger] )
			current_idx += 1
		end
	end

	isGeq = false
	isLeq = true
	isDualized = false
	for i in 1:ins.J
		f[(sizeF-sizeFC+1):end, current_idx] = vcat(get_cr_features(fmt,featObj.μ[i], featObj.μSP[i],μ0[i]),Float32[ins.c[i],  isGeq, isLeq, isDualized])
		current_idx += 1
	end
	return f
end

"""
# Arguments:
- `lt`: learnign type, it should be a sub-type of learningType,
- `featObj`: features object containing all the characteristics, 
- `ins`: instance structure, it should instanceGA,
- `fmt`: features matrix type.

Returns the bipartite graph representation with the associated nodes-features matrix.
"""
function featuresExtraction(lt::learningType, featObj, ins::instanceGA, fmt::abstract_features_matrix)

	nodesPC = collect(1:ins.I)

	nodesx = [length(nodesPC) + i for i in 1:ins.I*ins.J]
	
	nodesCC = [length(nodesPC) + length(nodesx) + i for i in 1:ins.J]

	tails = Int64[]
	heads = Int64[]
	weightsE = Float32[]

	β=0.5

	for i in 1:ins.I
		for j in 1:ins.J
			push!(tails, nodesx[i+ins.I*(j-1)])
			push!(heads, nodesCC[j])
			push!(weightsE, preprocess_weight(fmt,ins.w[i,j]))


			push!(tails, nodesCC[j])
			push!(heads, nodesx[i+ins.I*(j-1)])
			push!(weightsE, preprocess_weight(fmt,ins.w[i,j]))


			push!(tails, nodesx[i+ins.I*(j-1)])
			push!(heads, nodesPC[i])
			push!(weightsE, preprocess_weight(fmt,1))


			push!(tails, nodesPC[i])
			push!(heads, nodesx[i+ins.I*(j-1)])
            push!(weightsE, preprocess_weight(fmt,1))
		end
	end

	f = features_matrix(ins,featObj,fmt)

	g = GNNGraph(tails, heads, weightsE, ndata = f, gdata = [1, lengthLM(ins)]) #|> gpu

	return add_self_loops(g)
end

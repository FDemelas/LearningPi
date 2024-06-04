"""
Features structure for the Capacitated Warehouse Location instance.

# Fields:
-`xCR`: primal solution of the Linear Relaxation associated to the variables that associate one items to a pack.
-`yCR`: primal solution of the Linear Relaxation associated to the variables say if we use or not a pack.
-`λ`: dual solution of the Linear Relaxation associated to the packing constraints.
-`μ`: dual solution of the Linear Relaxation associated to the packing constraints.
-`objCR`: objective value of the Linear Relaxation.
-`xLR`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables that associate one items to a pack (using the dual variables λ of the linear relaxation).
-`yLR`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables say if we use or not a pack (using the dual variables λ of the linear relaxation).
-`objLR`: objective value of the Knapsack Lagrangian Relaxation (using the dual variables λ of the linear relaxation). 
"""
struct featuresCWL
	xCR::Matrix{Float32}
	yCR::Vector{Float32}
	λ::Vector{Float32}
	μ::Vector{Float32}
	xRC::Matrix{Float32}
	yRC::Vector{Float32}
	λSP::Vector{Float32}
	μSP::Vector{Float32}
	objCR::Float32
	xLR::Matrix{Float32}
	yLR::Vector{Int64}
	objLR::Float32
end

"""
# Arguments:
	- `ins`: instance object, it should be of type instanceCWL 

	solve the Continuous Relaxation and the Lagrangian Sub-Problem considering as Lagrangian Multipliers
	the dual variables associated to the relaxed constraints and then returns a features structure.
"""
function create_features(ins::instanceCWL)
	objCR, λ, μ, xCR, yCR, λSP, μSP, xRC, yRC = CR(ins)

	objLR, xLR, yLR = LR(ins, λ)

	return featuresCWL(xCR, yCR, λ, μ, xRC, yRC, λSP, μSP, objCR, xLR, yLR, objLR)
end

"""
# Arguments:
- `ins`: instance structure, it should be a sub-type of instanceCWL
- `featObj`: features object containing all the characteristics 
- `fmt`: features matrix type.

Construct the matrix of the features for a bipartite-graph representation of the instance.
"""
function features_matrix(ins::instanceCWL, featObj, fmt::abstract_features_matrix)
	sizeFC = size_features_constraint(fmt)
	sizeFV = size_features_variable(fmt)
	sizeF = 2 * (sizeFC + sizeFV)
	f = zeros(Float32, (sizeF, ins.J + 2 * ins.I + ins.J * ins.I))

	isGeq = true
	isLeq = true
	isDualized = true

	λ0 = zeros(1,ins.J) # randn?
    	xLR0 = zeros(ins.I,ins.J)
	yLR0 = zeros(ins.I)
	μ0 = zeros(ins.I)
	
	if ( typeof(fmt) == lr_features_matrix )
		obj, xLR0, yLR0 =  LR(ins,λ0) 
    	μ0 = sum( xLR0,dims=1) .- 1
	end

	current_idx = 1
	for j in 1:ins.J
		f[1:sizeFC, current_idx] = vcat(get_cr_features(fmt, featObj.λ[j], featObj.λSP[j],λ0[j]), Float32[1, isGeq, isLeq, isDualized])
		current_idx += 1
	end

	isInteger = false
	for j in 1:ins.J
		for i in 1:ins.I
			f[sizeFC+1:(sizeFV+sizeFC), current_idx] = vcat(get_cr_features(fmt, featObj.xCR[i, j], featObj.xRC[i, j],xLR0[i,j]), Float32[ins.c[i, j], isInteger])
			current_idx += 1
		end
	end

	isInteger = true
	for i in 1:ins.I
		f[(sizeFV+sizeFC+1):(2*sizeFV+sizeFC), current_idx] = vcat(get_cr_features(fmt, featObj.yCR[i], featObj.yRC[i], yLR0[i]), Float32[ins.f[i], isInteger])
		current_idx += 1
	end

	isGeq = false
	isLeq = true
	isDualized = false
	for i in 1:ins.I
		f[(sizeF-sizeFC+1):end, current_idx] = vcat(get_cr_features(fmt, featObj.μ[i], featObj.μSP[i],μ0[i]), Float32[ins.q[i], isGeq, isLeq, isDualized])
		current_idx += 1
	end
	return f
end

"""
# Arguments:
- `lt`: learnign type, it should be a sub-type of learningType
- `featObj`: features object containing all the characteristics 
- `ins`: instance structure, it should instanceCWL
- `fmt`: features matrix type.

Returns the bipartite graph representation with the associated nodes-features matrix.
"""
function featuresExtraction(_::learningType, featObj, ins::instanceCWL, fmt::abstract_features_matrix)
	nodesPC = collect(1:ins.J)

	nodesx = [length(nodesPC) + i for i in 1:ins.I*ins.J]
	nodesy = [length(nodesPC) + length(nodesx) + i for i in 1:ins.I]

	nodesCC = [length(nodesPC) + length(nodesx) + length(nodesy) + i for i in 1:ins.I]

	tails = Int64[]
	heads = Int64[]
	weightsE = Float32[]

	for e in 1:ins.I
		push!(tails, nodesy[e])
		push!(heads, nodesCC[e])
		push!(weightsE, preprocess_weight(fmt,-ins.q[e]))
		

		push!(tails, nodesCC[e])
		push!(heads, nodesy[e])
		push!(weightsE, preprocess_weight(fmt,-ins.q[e]))
	end

	for e in 1:ins.I
		for k in 1:ins.J
			push!(tails, nodesx[k+ins.I*(e-1)])
			push!(heads, nodesCC[e])
			push!(weightsE, preprocess_weight(fmt,ins.d[e]))

			push!(tails, nodesCC[e])
			push!(heads, nodesx[k+ins.I*(e-1)])
			push!(weightsE, preprocess_weight(fmt,ins.d[e]))
		end
	end

	for e in 1:ins.I
		for k in 1:ins.J
			push!(tails, nodesx[k+ins.I*(e-1)])
			push!(heads, nodesPC[k])
			push!(weightsE, preprocess_weight(fmt,1))

			push!(tails, nodesPC[k])
			push!(heads, nodesx[k+ins.I*(e-1)])
			push!(weightsE, preprocess_weight(fmt,1))
		end
	end

	f = features_matrix(ins, featObj, fmt)

	g = GNNGraph(tails, heads, weightsE, ndata = f, gdata = [1, lengthLM(ins)]) #|> gpu

	return add_self_loops(g)
end


"""
Struct containing the information of the features for an instance.

# Fields:
- `xCR`: the value of the flow variables for the optimal solution of the linear relaxation,
- `yCR`: the value of the decision variables for the optimal solution of the linear relaxation,
- `λ`: the value of the dual variables associated to the flow constraints, for the optimal solution of the linear relaxation,
- `μ`: the value of the dual variables associated to the capacity constraints for the optimal solution of the linear relaxation,
- `objCR`: the objective value of the linear relaxation,
- `xLR`: the value of the flow variables for the optimal solution of the sub-problem for the knapsack relaxation, considering as lagrangian multiers the vector λ,
- `yLR`: the value of the design variables for the optimal solution of the sub-problem for the knapsack relaxation, considering as lagrangian multiers the vector λ,
- `LRarcs`: the objective values, for each edge, of the optimal solution of the sub-problem for the knapsack relaxation, considering as lagrangian multiers the vector λ,
- `objLR`: the objective value of the sub-problem for the knapsack relaxation, considering as lagrangian multiers the vector λ,
- `origins`: a matrix of size K×V the cost of the shortest path from the origin to the current node with costs in an edge e:  ins.r[k,e]+ins.f[e]/ins.c[e],
- `destinations`: a matrix of size K×V the cost of the shortest path from the current node to the destination with costs in an edge e:  ins.r[k,e]+ins.f[e]/ins.c[e],
- `distance`: a matrix of size V×V with the distance in terms of number of edges for the shortest path from each two nodes.
"""
struct featuresMCND
	xCR::Matrix{Float32}
	yCR::Vector{Float32}
	λ::Matrix{Float32}
	μ::Vector{Float32}
	xRC::Matrix{Float32}
	yRC::Vector{Float32}
	λSP::Matrix{Float32}
	μSP::Vector{Float32}
	objCR::Float32
	xLR::Matrix{Float32}
	yLR::Vector{Int64}
	LRarcs::Vector{Float32}
	objLR::Float32
end

"""
# Arguments:
  - `ins`: instance structure, should be of type cpuInstanceMCND.

Create and return as output a features structure for the MCND instance `ins`.  
"""
function create_features(ins::cpuInstanceMCND)

	objCR, λ, μ, xCR, yCR, λSP, μSP, xRC, yRC = CR(ins)

	xLR = zeros(sizeK(ins), sizeE(ins))
	yLR = zeros(sizeE(ins))
	LRarcs = zeros(sizeE(ins))

	demands = collect(1:sizeK(ins))

	for e in 1:sizeE(ins)
		LRarcs[e] = LearningPi.LR(ins, λ, xLR, yLR, demands, e)
	end

	objLR = sum(LRarcs) - constantLagrangianBound(ins, λ)

	return featuresMCND(xCR, yCR, λ, μ, xRC, yRC, λSP, μSP, objCR, xLR, yLR, LRarcs, objLR)
end

"""
# Arguments:
- `ins`: instance structure, it should be a sub-type of abstractInstanceMCND,
- `featObj`: features object containing all the characteristics, 
- `fmt`: features matrix type.

Construct the matrix of the features for a bipartite-graph representation of the instance.
"""
function features_matrix(ins::abstractInstanceMCND, featObj::featuresMCND, fmt::abstract_features_matrix)
	V, K, E = sizeV(ins), sizeK(ins), sizeE(ins)
	sizeFC = size_features_constraint(fmt)
	sizeFV = size_features_variable(fmt)
	sizeF = 2 * (sizeFC + sizeFV)

	f = zeros(Float32, (sizeF, (V + E) * K + 2 * E))

	isGeq = true
	isLeq = true
	isDualized = true

	λ0 = zeros(K, V) # randn?

	obj, xLR0, yLR0 = (typeof(fmt) == lr_features_matrix) ? LR(ins, λ0) : (0, zeros(size(featObj.xCR)), zeros(size(featObj.yCR)))
	μ0 = (typeof(fmt) == lr_features_matrix) ? sum(xLR0, dims = 1) - ins.c' : zeros(size(featObj.yCR))



	current_idx = 1
	for i in 1:V
		for k in 1:K
			f[1:sizeFC, current_idx] = vcat(get_cr_features(fmt, featObj.λ[k, i], featObj.λSP[k, i], λ0[k, i]), Float32[b(ins, i, k), isGeq, isLeq, isDualized])
			current_idx += 1
		end
	end

	isInteger = false

	for e in 1:E
		for k in 1:K
			f[sizeFC+1:(sizeFV+sizeFC), current_idx] = vcat(get_cr_features(fmt, featObj.xCR[k, e], featObj.xRC[k, e], xLR0[k, e]), Float32[routing_cost(ins, e, k), isInteger])
			current_idx += 1
		end
	end

	isInteger = true
	for e in 1:E
		f[(sizeFV+sizeFC+1):(2*sizeFV+sizeFC), current_idx] = vcat(get_cr_features(fmt, featObj.yCR[e], featObj.yRC[e], yLR0[e]), Float32[fixed_cost(ins, e), isInteger])
		current_idx += 1
	end

	isGeq = false
	isLeq = true
	isDualized = false
	for e in 1:E
		f[(sizeF-sizeFC+1):end, current_idx] = vcat(get_cr_features(fmt, featObj.μ[e], featObj.μSP[e], μ0[e]), Float32[capacity(ins, e), isGeq, isLeq, isDualized])
		current_idx += 1
	end

	return f
end

"""
# Arguments:
- `lt`: learnign type, it should be a sub-type of learningGNN,
- `featObj`: features object containing all the characteristics,
- `ins`: instance structure, it should be a sub-type of abstractInstanceMCND.

Returns the bipartite graph representation with the associated nodes-features matrix.
"""
function featuresExtraction(lt::learningType, featObj, ins::abstractInstanceMCND, fmt::abstract_features_matrix)
	V, K, E = sizeV(ins), sizeK(ins), sizeE(ins)
	lnfc = (V * K) + 1
	lny = lnfc + E * K
	lncc = lny + E

	nodesFC = collect(1:V*K)
	nodesx = collect(lnfc:(lnfc+E*K-1))
	nodesy = collect(lny:(lny+E-1))
	nodesCC = collect(lncc:(lncc+E-1))

	sizeBiArcs = 2 * E * (1 + 3 * K)
	tails = zeros(Int64, sizeBiArcs)
	heads = zeros(Int64, sizeBiArcs)
	weightsE = zeros(Float32, sizeBiArcs)

	tmp = 1
	tails[tmp:E] = nodesy
	heads[tmp:E] = nodesCC
	weightsE[tmp:E] = -ins.c
	tmp += E

	tails[tmp:(tmp+E-1)] = nodesCC
	heads[tmp:(tmp+E-1)] = nodesy
	weightsE[tmp:(tmp+E-1)] = -ins.c
	tmp += E

	for e in 1:E
		for k in 1:K
			tails[tmp] = nodesx[k+K*(e-1)]
			heads[tmp] = nodesCC[e]
			weightsE[tmp] = preprocess_weight(fmt, 1)
			tmp += 1
			tails[tmp] = nodesCC[e]
			heads[tmp] = nodesx[k+K*(e-1)]
			weightsE[tmp] = preprocess_weight(fmt, 1)
			tmp += 1
		end
	end

	for e in 1:E
		i = tail(ins, e)
		j = head(ins, e)
		for k in 1:K
			tails[tmp] = nodesx[k+K*(e-1)]
			heads[tmp] = nodesFC[k+K*(i-1)]
			weightsE[tmp] = preprocess_weight(fmt, 1)
			tmp += 1

			tails[tmp] = nodesFC[k+K*(i-1)]
			heads[tmp] = nodesx[k+K*(e-1)]
			weightsE[tmp] = preprocess_weight(fmt, 1)
			tmp += 1

			tails[tmp] = nodesx[k+K*(e-1)]
			heads[tmp] = nodesFC[k+K*(j-1)]
			weightsE[tmp] = preprocess_weight(fmt, -1)
			tmp += 1

			tails[tmp] = nodesFC[k+K*(j-1)]
			heads[tmp] = nodesx[k+K*(e-1)]
			weightsE[tmp] = preprocess_weight(fmt, -1)
			tmp += 1
		end
	end

	f = features_matrix(ins, featObj, fmt)

	g = GNNGraph(tails, heads, weightsE, ndata = f, gdata = [sizeK(ins), sizeV(ins)])

	return add_self_loops(g)
end

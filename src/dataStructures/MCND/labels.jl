"""
Struct containing the information relative to the labels

# Fields
- `π::Matrix{Float32}`: matrix containing the gold Lagrangian multipliers. π[k, i] gives the values of the Lagrangian multiplier associated with demand k and node i.
- `x::Matrix{Int16}`: solution x of the Lagrangian problem. x[k, a] gives the value of the solution x_a^k of the Lagrangian problem L(π) for demand k and arc a.
- `y::Vector{Int16}`: solution y of the Lagrangian problem. y[a] gives the value of the solution y_a of the Lagrangian problem L(π) for arc a.
- `LRarcs::Vector{Float32}`: Values of the Lagrangian problem for the arcs. LRarcs[a] gives the value of subproblem L_a associated with arc a.
- `objLR::Float32`: Value of the Lagrangian dual problem
"""
struct labels
	π::Matrix{Float32}
	xLR::Matrix{Int16}
	yLR::Vector{Int16}
	LRarcs::Vector{Float32}
	objLR::Float32
end

"""
createLabels(π, x, y, LRarcs, objLR, ins::abstractInstanceMCND)

# Arguments:
- `π`: a (optimal) Lagrangian multipliers vector
- `x`: the flow variables in the Lagrangian sub-problem, obtained afer the resolution of the sub-problem with multipliers π
- `y`: the design variables in the Lagrangian sub-problem, obtained afer the resolution of the sub-problem with multipliers π
- `LRarcs`: a vector containign the bounds for each edge of the Lagrangian sub-problem considering π as Lagrangian multipleirs vector
- `objLR`: the bound of the Lagrangian sub-problem considering π as Lagrangian multipleirs vector
- `ins`: the instance structure (standard instance formulation, without regularization).

	For instance (not normalized), nothing to do, just return a proper label structure.
"""
createLabels(π, xLR, yLR, LRarcs, objLR, ins::abstractInstanceMCND) = labels(π, xLR, yLR, LRarcs, objLR)


"""
function read_labels(fileLabel::String, ins::abstractInstanceMCND)

	#Arguments:
		- `fileLabel`: the path to the file where to find labels informations
		- `ins`: instance object, it should be of type abstractInstanceMCND 

	read the labels and returns a labels structure.
"""
function read_labels(fileLabel::String, ins::abstractInstanceMCND)
	π = zeros(sizeK(ins), sizeV(ins))
	k = 1
	f = open(fileLabel)
	for k in 1:sizeK(ins)
		line = split(readline(f), " "; keepempty = false)
		for e in 1:sizeV(ins)
			π[k, e] = parse(Float32, line[e])
		end
	end
	close(f)
	x = zeros(sizeK(ins), sizeE(ins))
	y = zeros(sizeE(ins))
	LRarcs = zeros(sizeE(ins))

	demands = collect(1:sizeK(ins))

	for e in 1:sizeE(ins)
		LRarcs[e] = LearningPi.LR(ins, π, x, y, demands, e)
	end
	objLR = sum(LRarcs) - constantLagrangianBound(ins, π)

	return LearningPi.labels(π, x, y, LRarcs, objLR)
end

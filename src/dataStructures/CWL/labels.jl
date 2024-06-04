"""
Label structure for the Capacitated Warehouse Location Problem.
# Fields:
	-`π`: optimal lagrangian multipliers vector.
	-`xLR`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables that associate one items to a pack (using the optimal Lagrangian multipliers).
	-`yLR`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables say if we use or not a pack (using the optimal Lagrangian multipliers).
	-`objLR`: optimal value of the Lagrangian Dual
"""
struct labelsCWL
	π::Vector{Float32}
	xLR::Matrix{Float32}
	yLR::Vector{Float32}
	objLR::Float32
end

"""
# Arguments:

-`π`: optimal lagrangian multipliers vector.
-`x`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables that associate one items to a pack (using the optimal Lagrangian multipliers).
-`y`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables say if we use or not a pack (using the optimal Lagrangian multipliers).
-`objLR`: optimal value of the Lagrangian Dual
- `ins`: instance object, it should be of type sub-type of instanceCWL 

Given all the fields construct a label structure for the Bin Packing Problem.
"""
createLabels(π, x, y, objLR, ins::instanceCWL) = labelsCWL(π, x, y, objLR)

"""
# Arguments:
		- `fileLabel`: the path to the file where to find labels informations
		- `ins`: instance object, it should be of type sub-type of instanceCWL 

	read the labels and returns a labels structure.
"""
function read_labels(fileLabel::String, ins::instanceCWL)
	π = zeros(Float32, ins.J)
	f = open(fileLabel)
	line = split(readline(f), " "; keepempty = false)
	for i in 1:ins.J
		π[i] = parse(Float32, line[i])
	end
	close(f)
	obj, x, y = LR(ins, π)
	return LearningPi.labelsCWL(π, x, y, obj)
end
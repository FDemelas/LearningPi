"""
Label structure for the Generalized Assignment Problem.
	#Fields:
	-`π`: optimal lagrangian multipliers vector.
	-`xLR`: primal solution of the Lagrangian Subproblem with optimal Lagrangian multipliers.
	-`objLR`: optimal value of the Lagrangian Dual.
"""
struct labelsGA
	π::Matrix{Float32}
	xLR::Matrix{Float32}
	objLR::Float32
end

"""
function createLabels(π, x, objLR, ins::instanceGA)

#Arguments:
	-`π`: optimal lagrangian multipliers vector.
	-`x`: primal solution of the Knapsack Lagrangian Relaxation associated to the variables that associate one items to a pack (using the optimal Lagrangian multipliers).
	-`objLR`: optimal value of the Lagrangian Dual

Given all the fields construct a label structure for the Bin Packing Problem.
"""
createLabels(π, x, objLR, ins::instanceGA) = labelsGA(π, x, objLR)

"""
function read_labels(fileLabel::String, ins::instanceGA)

	#Arguments:
		- `fileLabel`: the path to the file where to find labels informations
		- `ins`: instance object, it should be sub-type of instanceGA 

	read the labels and returns a labels structure.
"""
function read_labels(fileLabel::String, ins::instanceGA)
	π = zeros(Float32, ins.I)
	f = open(fileLabel)
	line = split(readline(f), " "; keepempty = false)
	for i in 1:ins.I
		π[i] = parse(Float32, line[i])
	end
	close(f)
	obj, x = LR(ins, π)
	return LearningPi.labelsGA(π, x, obj)
end

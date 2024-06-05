abstract type abstract_loss end

abstract type abstract_loss_factory end

"""
# Arguments:
	- `_`: the loss parameters.

	returns the device (cpu/gpu) used to compute the loss.
	For a general loss will be CPU.
"""
function get_device(_::abstract_loss)
	return cpu
end


"""
# Arguments:
- `_`: lagrangian multipliers vector candidate, 
- `v`: the value of the loss function,
- `example`: dataset sample object,
- `_`: loss parameters.

Compute the value of the sub-problem for the loss for which it cannot be obtained in a smarter way.
"""
function sub_problem_value(π, _, example, _::abstract_loss)
	return LR(example.instance, cpu(π))[1]
end



#################################
#          Shared Functions     #
#################################


"""
# Arguments:
	- `x`: the solution of the Lagrangian Sub-problem,
	- `ins`: a cpuInstanceMCND structure.

This function compute and returns the gradient of the sub-problem objective function w.r.t. the Lagrangian Multipliers.
"""
function gradient_lsp(x::AbstractVecOrMat, ins::cpuInstanceMCND)
	grad = zeros(Float32, sizeLM(ins))
	for k in 1:sizeK(ins)
		for i in 1:sizeV(ins)
			grad[k, i] = sum([x[k, ij] for ij in 1:sizeE(ins) if tail(ins, ij) == i]) - sum([x[k, ij] for ij in 1:sizeE(ins) if head(ins, ij) == i]) - b(ins, i, k)
		end
	end
	return grad
end


"""
# Arguments:
	- `x`: the solution of the Lagrangian Sub-problem,
	- `ins`: a cpuInstanceCWL structure.

This function compute and returns the gradient of the sub-problem objective function w.r.t. the Lagrangian Multipliers.
"""
function gradient_lsp(x::AbstractVecOrMat,ins::cpuInstanceGA)
    grad = zeros(Float32, 1,ins.I)
    for i in 1:ins.I
        grad[1,i] = sum([x[i,j] for j in 1:ins.J]) - 1
    end
    return -grad
end

"""
# Arguments:
	- `x`: the solution of the Lagrangian Sub-problem,
	- `ins`: a cpuInstanceCWL structure.

This function compute and returns the gradient of the sub-problem objective function w.r.t. the Lagrangian Multipliers.
"""
function gradient_lsp(x::AbstractVecOrMat, ins::gpuMCNDinstance)
	return x * ins.E - ins.B
end


"""
# Arguments:
	- `x`: the solution of the Lagrangian Sub-problem,
	- `ins`: a cpuInstanceCWL structure.

This function compute and returns the gradient of the sub-problem objective function w.r.t. the Lagrangian Multipliers.
"""
function gradient_lsp(x::AbstractVecOrMat, ins::cpuInstanceCWL)
	return sum(x, dims = 1) .- 1
end
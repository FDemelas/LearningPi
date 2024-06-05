###############################
#         LR - GPU  loss      #
###############################

"""
Structure that realize a LR gpu loss.
This structure can be used as function.
The constructor need no paramameters.
"""
struct loss_LR_gpu <: abstract_loss end

"""
Structure that should be used to construct a LR GPU loss function.
"""
struct loss_LR_gpu_factory <: abstract_loss_factory end

"""
# Arguments:
- `_`: a factory, for this implentation it should be of type ~`loss_LR_gpu_factory`.

Return a loss corresponding to the factory.
"""
create_loss(_::loss_LR_gpu_factory) = loss_LR_gpu()

"""
# Arguments:
- `π`: a Lagrangian Multipliers Vector,
- `example`: an abstract example.

Computes the value of the LR GPU loss.
"""
function (l::loss_LR_gpu)(π; example::abstract_example)
	loss_device = get_device(l)
	return -LR(example.instance, loss_device(π))[1]
end


"""
# Arguments:
- `_`: loss function.

returns the device to use with this loss.
implementationn this case GPU.
"""
function get_device(::loss_LR_gpu)
	@assert has_gpu
	return gpu
end

"""
# Arguments:
- `_`: lagrangian multipliers (are not used in this implementation),
- `v`: loss function value,
- `example`: an abstract_example,
- `_`: loss function .

Compute the sub-problem value without solving the Lagrangian Sub-Problem, if it is already solved during the computation of the loss.
"""
function sub_problem_value(_, v, _, _::loss_LR_gpu)
	return -v
end

"""
Compute the value of the Learning by Experience loss (usining the inverse of value of the sub-problem) and its pullback function.
"""
function ChainRulesCore.rrule(::loss_LR_gpu, π::AbstractArray; example::abstract_example)
	obj, x, _ = LR(example.instance, π)
	grad = -gradient_lsp(x, example.instance)
	loss_pullback(dl) = (NoTangent(), grad * dl, NoTangent())
	return -obj, loss_pullback
end

# declare the LR-gpu Loss as Flux functor
Flux.@functor loss_LR_gpu

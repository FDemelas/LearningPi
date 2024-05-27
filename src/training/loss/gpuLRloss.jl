###############################
#         LR - GPU  loss      #
###############################


struct loss_LR_gpu <: abstract_loss end

struct loss_LR_gpu_factory  <: abstract_loss_factory end

create_loss(_::loss_LR_gpu_factory) = loss_LR_gpu()

function (l::loss_LR_gpu)(π; example::abstract_example)
	loss_device = get_device(l)
	return - LR(example.instance, loss_device(π))[1]
end

function get_gevice(::loss_LR_gpu) 
    @assert has_gpu
    return gpu
end


function sub_problem_value(_, v, _, _::loss_LR_gpu)
	return -v
end

"""
Compute the value of the Learning by Experience loss (usining the inverse of value of the sub-problem) and its pullback function.
"""
function ChainRulesCore.rrule(::loss_LR_gpu, π::AbstractArray; example::abstract_example)
	obj, x, _ = LR(example.instance, π)
	grad = - gradient_lsp(x, example.instance)
    loss_pullback(dl) = (NoTangent(), grad * dl, NoTangent())
	return -obj , loss_pullback
end


Flux.@functor loss_LR_gpu

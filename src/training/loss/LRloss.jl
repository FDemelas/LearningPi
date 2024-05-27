##############################
#           LR   loss         #
###############################


struct loss_LR <: abstract_loss end

struct loss_LR_factory  <: abstract_loss_factory end

create_loss(_::loss_LR_factory) = loss_LR()

function (l::loss_LR)(π; example::abstract_example)
	loss_device = get_device(l)
	return - LR(example.instance, loss_device(π))[1]
end

function sub_problem_value(_, v, _, _::loss_LR)
	return -v
end

"""
Compute the value of the Learning by Experience loss (usining the inverse of value of the sub-problem) and its pullback function.
"""
function ChainRulesCore.rrule(::loss_LR, π::AbstractArray; example::abstract_example)
	value, x, _ = LR(example.instance, cpu(π))
	grad = - gradient_lsp(x, example.instance)
	loss_pullback(dl) = (NoTangent(), grad * dl, NoTangent())
	return - value, loss_pullback
end

Flux.@functor loss_LR
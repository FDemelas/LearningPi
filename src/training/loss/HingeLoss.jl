###############################
#           Hinge   loss      #
###############################

"""
Structure of parameters for loss obtained as the inverse of the sub-problem obj value.

# Fields:
-`α`: regularization term. Warning: for the moment this parameter not used!
"""
struct loss_hinge <: abstract_loss
	α::Float32
end

struct loss_hinge_factory <: abstract_loss_factory end

"""
# Arguments:
-`:`: loss parameters, it should be a structure of type HingeLoss.   

return the loss correspondent to loss paramameters of type HingeLoss.
"""
function create_loss(lf::loss_hinge_factory, α = 0.1)
	return loss_hinge(α)
end

"""
# Arguments:
- `π`: lagrangian multipliers vector candidate,
- `example`: dataset sample object.
  
Computes the value of the Hinge loss.
"""
function (l::loss_hinge)(π; example)
	return value_LR(example.instance, π, example.gold.xLR', example.gold.yLR) - LR(example.instance, cpu(π))[1]
end

"""
Compute the value of the Hinge Loss and its pullback function.
"""
function ChainRulesCore.rrule(::loss_hinge, π::AbstractArray; example)
	obj, x, _ = LR(example.instance, cpu(π))
	grad = gradientX(x, example.instance)
	gradG = gradientX(example.gold.xLR', example.instance)
	value = value_LR(example.instance, π, example.gold.xLR', example.gold.yLR) - obj
	loss_pullback(dl) = (NoTangent(), (gradG - grad) * dl, NoTangent())
	return value, loss_pullback
end

"""
# Arguments:
- `_`: lagrangian multipliers vector candidate, 
- `v`: the value of the loss function,
- `example`: dataset sample object,
- `_`: loss parameters, it should be a structure of type HingeLoss.   

Compute the value of the sub-problem without recomputing it, but using the value of the loss function (for the HingeLoss) 
and other informations contained in the sample.
"""
function sub_problem_value(π, v, example, _::loss_hinge)
	return -v + value_LR(example.instance, cpu(π), example.gold.xLR', example.gold.yLR)
end

Flux.@functor loss_hinge

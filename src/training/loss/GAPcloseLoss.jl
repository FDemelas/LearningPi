###############################
#     GAP  closure  loss      #
###############################


"""
Structure that realize a GAP closure loss.
This structure can be used as function.

# Fields:
- `lr`: a lagranian sub-problem loss of type `loss_LR`.

The constructor need no paramameters.
"""
struct loss_GAP_closure <: abstract_loss
	lr::loss_LR
	loss_GAP_closure() = new(loss_LR())
end

"""
Structure that should be used to construct a GAP closure loss function.
"""
struct loss_GAP_closure_factory <: abstract_loss_factory end

"""
# Arguments:
- `_`: a factory, for this implentation it should be of type ~`loss_GAP_closure_factory`.

Return a loss corresponding to the factory.
"""
create_loss(_::loss_GAP_closure_factory) = loss_GAP_closure()

"""
# Arguments:
- `π`: a Laagrangian Multipliers Vector,
- `example`: an abstract example.

Computes the value of the GAP closure loss.
"""
function (l::loss_GAP_closure)(π; example::abstract_example)
	return -l.lr(π; example) / (example.gold.objLR - example.linear_relaxation) * 100
end

# declare the Gap Closure Loss as Flux functor
Flux.@functor loss_GAP_closure

"""
# Arguments:
- `_`: lagrangian multipliers (are not used in this implementation),
- `v`: loss function value,
- `example`: an abstract_example,
- `_`: loss function. 

Compute the sub-problem value without solving the Lagrangian Sub-Problem, if it is already solved during the computation of the loss.
"""
function sub_problem_value(_, v, example::abstract_example, _::loss_GAP_closure)
	return -v * (example.gold.objLR - example.linear_relaxation) / 100
end

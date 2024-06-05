###############################
#           MSE   loss        #
###############################

"""
Structure that realize a MSE loss.
This structure can be used as function.
The constructor need no paramameters.
"""
struct loss_mse <: abstract_loss end

"""
Structure that should be used to construct a MSE loss function.
"""
struct loss_mse_factory <: abstract_loss_factory end

"""
# Arguments:
- `_`: a factory, for this implentation it should be of type ~`loss_mse_factory`.

Return a loss corresponding to the factory.
"""
create_loss(_::loss_mse_factory) = loss_mse()

"""
# Arguments:
- `π`: lagrangian multipliers vector candidate,
- `example`: dataset sample object,
-`_`: loss parameters, it should be a structure of type MSELoss.   

Returns the loss function value obtained taking the MSE beteern the predicted Lagrangian multipliers `π` and the optimal ones in `example`.
"""
function (l::loss_mse)(π; example)
	return Flux.mse(π, example.gold.π)
end

# declare the LR (cpu) Loss as Flux functor
Flux.@functor loss_mse
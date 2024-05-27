###############################
#           MSE   loss        #
###############################



struct loss_mse <: abstract_loss end

struct loss_mse_factory  <: abstract_loss_factory end

create_loss(_::loss_mse_factory) = loss_mse()

"""
loss_mse(π; example)

# Arguments:
- `π`: lagrangian multipliers vector candidate.
- `example`: dataset sample object.
-`_`: loss parameters, it should be a structure of type MSELoss.   

returns the loss function value obtained taking the MSE beteern the predicted Lagrangian multipliers `π` and the optimal ones in `example`.
"""
function (l::loss_mse)(π; example)
	return Flux.mse(π,example.gold.π)
end

Flux.@functor loss_mse
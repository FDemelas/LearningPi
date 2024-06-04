###############################
#  LR ( CPU Multi-Pred ) loss #
###############################

"""
Structure that realize a multi-prediction LR (CPU) loss.
This structure can be used as function.
# Fields:
- `α`: a penalization parameter to weight the different predictions, by default is 0.5.
- `lr`: a loss of type `loss_LR`, automatically constructed
"""
struct loss_multi_LR <: abstract_loss
	α::Float32
	lr::loss_LR
	loss_multi_LR(α = 0.5) = new(α, loss_LR())
end

"""
Structure that should be used to construct a multi-prediction LR (CPU) loss function.
"""
struct loss_multi_LR_factory <: abstract_loss_factory end

"""
# Arguments:
- `π`: a Lagrangian Multipliers Vector,
- `example`: an abstract example.

Computes the value of the multi-prediction LR (CPU) loss.
"""
function (l::loss_multi_LR)(πs::AbstractVector; example::abstract_example)
	πs = get_device(l)(πs)
	lπs = Int64(size(πs)[1] / example.features.gdata.u[1])
	for i in 1:lπs
		sIdx = ((i - 1) * example.features.gdata.u[1] + 1)
		eIdx = i * example.features.gdata.u[1]
		value += (l.lr(πs[sIdx:eIdx, :]; example)) * ((l.α)^(lπs - i))
	end
	return values
end


# declare the multi-prediction LR (cpu) Loss as Flux functor
Flux.@functor loss_multi_LR
###############################
#   LR ( 4 Multi Pred ) loss  #
###############################

struct loss_multi_LR_factory <: abstract_loss_factory end

struct loss_multi_LR <: abstract_loss
α::Float32
lr::loss_LR
loss_multi_LR(α=0.5) = new(α,loss_LR())
end

function (l::loss_multi_LR)(πs::AbstractVector; example::abstract_example)
    πs = get_device(l)(πs)
    lπs = Int64(size(πs)[1] / example.features.gdata.u[1])
    for i in 1:lπs
		sIdx = ((i - 1) * example.features.gdata.u[1] + 1)
		eIdx = i * example.features.gdata.u[1]
		value += ( l.lr(πs[sIdx:eIdx, :];example)) * ((l.α)^(lπs - i))
	end
    return values
end
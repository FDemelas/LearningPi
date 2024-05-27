###############################
#           GAP  loss         #
###############################


struct loss_GAP <: abstract_loss 
	lr::loss_LR
	loss_GAP() = new(loss_LR())
end
	
struct loss_GAP_factory <: abstract_loss_factory end
	
create_loss(_::loss_GAP_factory) = loss_GAP()
	
function (l::loss_GAP)(π; example::abstract_example)
	return (1 - l.lr(π; example) / (example.gold.objLR)) * 100
end
	
Flux.@functor loss_GAP
	
function sub_problem_value(_, v, example, _::loss_GAP)
	return (1 - v / 100) * (example.gold.objLR)
end
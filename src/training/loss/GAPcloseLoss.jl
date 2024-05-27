###############################
#     GAP  closure  loss      #
###############################


struct loss_GAP_closure <: abstract_loss 
    lr::loss_LR
    loss_GAP_closure() = new(loss_LR())
    end
    
    struct loss_GAP_closure_factory <: abstract_loss_factory end
    
    create_loss(_::loss_GAP_closure_factory) = loss_GAP_closure()
    
    function (l::loss_GAP_closure)(π; example::abstract_example)
        return - l.lr(π; example) / (example.gold.objLR - example.linear_relaxation) * 100
    end
    
    Flux.@functor loss_GAP_closure
    

    function sub_problem_value(_, v, example, _::loss_GAP_closure)
        return -v * (example.gold.objLR - example.linear_relaxation) / 100
    end
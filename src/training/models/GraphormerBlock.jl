struct GraphormerBlock <:AbstractModel
    Convolution
    MLP
    GraphormerBlock(hidden_sample,inpOut::Int, init,rng,pDrop,h_MLP::Vector{Int},ConvLayer=GraphConv,act::Any=relu,act_conv=identity,aggr=mean) =
    (
        r1 = LayerNorm(1.0f-5, inpOut);
        r2 = LayerNorm(1.0f-5, inpOut);
        
        hMLP = cat([inpOut], h_MLP, dims = 1);
        layersMLP = [Dense(hMLP[i] => hMLP[i+1], act; init = init) for i in 1:(length(hMLP)-1)];
        Convolution = GNNChain(Parallel(+, GNNChain( r1, ConvLayer(inpOut => inpOut,act_conv,; aggr, init), Dropout(pDrop;rng)), identity));
        
        MLP=Parallel(+, Chain( r2, layersMLP..., Dense(hMLP[end] => inpOut, act; init = init), Dropout(pDrop;rng)), identity);
        MLP = hidden_sample ? Chain(MLP,Dense(inpOut=>2*inpOut;init)) : MLP;
        new(Convolution,MLP)
    )
    GraphormerBlock(Convolution,MLP)=(new(Convolution,MLP))
end    

function (m::GraphormerBlock)(x,h)
    h = m.Convolution(x,h)
    return m.MLP(h)
end

Flux.@functor GraphormerBlock    

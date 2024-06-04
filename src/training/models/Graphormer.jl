"""
This structure provide an implementation of the main neural-network model of this project.

# Fields:
- `HiddenMap`: a `Chain` to map the features into the Hidden Space,
- `Graphormers`: a `GNNChain` to apply several Graphormer Blocks to the hidden-features representation, each component can be seen as main-block of the model,
- `Decoders`: a `Chain` of decoders, it should have the same size as the desired predictions,
- `Sampling`: sampling function,
- `n_gr`: number of main-blocks,
- `train_mode`: if the model is in training mode or not, the main change is that if off we does not sample, but just take the mean,
- `prediction_layers`: indexes of the main-blocks after which we want insert a Decoder to provide a Lagrangian Multipliers Prediction,
- `where_sample`: a `SamplingPosition` to handle different possibilities of Sampling,
- `only_last`: a boolean that says if we want only a single Lagragian Multipliers prediction associated to the last main-block
- `dt`: deviation type.

The constructor of this structure have the following 
# Arguments:
- `HiddenMap`: as in the Fields,
- `Graphormers`: as in the Fields,
- `Decoders`: as in the Fields,
- `Sampling`:as in the Fields,
- `where_sample`:as in the Fields,
-  `prediction_layers`: as in the Fields, by default empty, in this case we predict only in the last Graphormers layers,
- `dt`: as in the Fields, by default `cr_deviation`.

This structure is declared as `Flux.functor` in order to efficiently and automatically implement the back-propagation.
It can be called providing as input simply the graph-neural-network   (a `GNNGraph`). 
"""
mutable struct Graphormer <: AbstractModel
    HiddenMap::Chain
    Graphormers::GNNChain
    Decoders::Chain
    Sampling::AbstractSampler # sampling functor
    n_gr::Int
    train_mode::Bool
    prediction_layers::Vector{Int}
    where_sample::SamplingPosition
    only_last::Bool
    dt::abstract_deviation
    Graphormer(HiddenMap,Graphormers,Decoders,Sampling, where_sample, prediction_layers=[], dt=cr_deviation()) = (
        only_last = false;
        if (prediction_layers == [])
            only_last=true;
        end;
        n_gr = length(Graphormers);
        train_mode=true;
        new(HiddenMap,Graphormers,Decoders,Sampling,n_gr, train_mode, prediction_layers,where_sample, only_last, dt) 
    )
end

"""
# Arguments:
   - `x`: input of the NN model of type Graphormer  (a `GNNGraph`).  

Forward computation of a Graphormer m, the output is the concatenation of all the multipliers predicted by the model   
"""
function (m::Graphormer)(x)
        λ = model_device(deviationFrom(x, m.dt))
        let o,y
        idx = 1
        sizes=cpu(x.gdata.u)
        h = m.HiddenMap(x.ndata.x)
        for i in 1:m.n_gr
            h = m.Graphormers[i](x,h)
            if m.where_sample.hidden_state 
                if m.train_mode
                    h = m.Sampling(h)
                else
                    h,_ = MLUtils.chunk(h, 2; dims = 1)
                end
            end
            if ( i in m.prediction_layers )# && !( m.only_last )
                if m.where_sample.before_decoding 
                    if m.train_mode
                        y = m.Sampling(h)
                    else
                        y,_ = MLUtils.chunk(h, 2; dims = 1)
                    end
                else
                    y = copy(h)
                end
                h2 = m.Decoders[idx](y[:,1:prod(sizes)])
                if m.where_sample.outside 
                    if m.train_mode
                        dev = m.Sampling(h2)
                    else
                        dev,_ = MLUtils.chunk(h2, 2; dims = 1)
                    end
                else
                    dev = copy(h2)
                end
                λ += reshape(dev,sizes...)
                if idx == 1
                    o = copy(λ)                   
                else
                    o = vcat(o,λ) 
                end
                idx += 1
            end
        end
    return o
    end
end

# Call @functor to allow for training.
Flux.@functor Graphormer

import Flux: gpu

"""
# Arguments:
- `m`: a `Graphormer` model.

Extends the `gpu` function of `Flux` to be applied to `Graphormer` model.
"""
function gpu(m::Graphormer)
    m1 = deepcopy(m)
    m1.Decoders =  gpu(m.Decoders)
    m1.Graphormers = gpu(m.Graphormers)
    m1.HiddenMap = gpu(m.HiddenMap)
    return m1
end

import Flux: cpu

"""
# Arguments:
- `m`: a `Graphormer` model.

Extends the `cpu` function of `Flux` to be applied to `Graphormer` model.
"""
function cpu(m::Graphormer)
    m1 = deepcopy(m)
    m1.Decoders =  cpu(m.Decoders)
    m1.Graphormers = cpu(m.Graphormers)
    m1.HiddenMap = cpu(m.HiddenMap)
    return m1
end

"""
# Arguments:
- `trainSet`: the (training) set,
- `nn`: a model of type `Graphormer`,
- `currentMetrics`: a dictionary that contains the metrix of the current iteration,
- `opt`: an `Optimiser`,
- `loss`: loss function,
- `epoch`: the epcoh counter (this patameter is unsued in the current implementation and it will be soon removed), 
- `lt`: learning type (this patameter is unsued in the current implementation and it will be soon removed),
- `dt`: deviation type  (this patameter is unsued in the current implementation and it will be soon removed).

This function performs the forward and backward pass for the model `nn` over all the (training) set `trainSet`.
"""
function forwardBackward(trainSet, nn::Graphormer, currentMetrics, opt,loss,  epoch::Int, lt::learningType, dt::abstract_deviation)
	nInst = length(trainSet)
	currentMetrics["loss training"] = 0
	currentMetrics["GAP training"] = 0
	par = Flux.params(nn)
	lossValue = 0.0f0
	GAPValue = 0.0f0
	GAP_CR_Value = 0.0f0
	device = get_device(loss)
	pm = is_min_SP(trainSet[1].instance) ? 1 : -1
	act_o = is_min_SP(trainSet[1].instance) ? identity : (f(x) =  -softplus(-x))
	for (idx,example) in enumerate(trainSet)
		#CUDA.seed!(idx)
          let pred,v
    		feat = model_device(example.features)
            #t0=time()
            grad = gradient(par) do
                pred = act_o(nn(feat))
                v = loss( device(pred) ; example )
            end
            #t1=time()-t0
            #v, grad = Flux.withgradient(() -> loss(device(nn(feat)); example), par)
            π = device(pred)
    		#π = prediction(gnn, feat, example, lt,dt)
    		lossValue += v / nInst
		    objPred = sub_problem_value(π, v, example,loss)
		    objGold = example.gold.objLR
		    GAPValue += pm * (objGold - objPred) / (objGold * nInst)
		    GAP_CR_Value += (1 - pm*(objGold - objPred) / (objGold - example.linear_relaxation)) / nInst
		    t0=time()
            Flux.update!(opt, par, grad)
            #t2=time()-t0
            		
        end
	end
	currentMetrics["loss training"] = pm * lossValue |> device
	currentMetrics["GAP training"] = GAPValue * 100 |> device
	currentMetrics["GAP CR training"] = GAP_CR_Value * 100 |> device
end

"""
# Arguments:
	-`nn`: neural network model of type `Graphormer`.

Returns a cpu version of the model that can be saved using a bson file.    
"""
function get_model(nn::Graphormer)
	return cpu(nn)
end

"""
# Arguments:
	-`nn`: neural network model of type `Graphormer`,
    -`lt`: learning type (of type `learningGNN`).

In this case only returns the model `nn`.
"""
function load_model(nn::Graphormer,lt::learningGNN)
	return nn
end

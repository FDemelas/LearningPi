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
function (m::Graphormer)(x)

# Arguments:
   - `x`: input of the NN model of type Graphormer    

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


# Call @functor to allow for training. Described below in more detail.
Flux.@functor Graphormer

import Flux: gpu

function gpu(m::Graphormer)
    m1 = deepcopy(m)
    m1.Decoders =  gpu(m.Decoders)
    m1.Graphormers = gpu(m.Graphormers)
    m1.HiddenMap = gpu(m.HiddenMap)
    return m1
end

import Flux: cpu
function cpu(m::Graphormer)
    m1 = deepcopy(m)
    m1.Decoders =  cpu(m.Decoders)
    m1.Graphormers = cpu(m.Graphormers)
    m1.HiddenMap = cpu(m.HiddenMap)
    return m1
end

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
get_model(nn::GNNChain)

#Arguments:
	-`nn`: neural network model

returns a cpu version of the model that can be saved using a bson file.    
"""
function get_model(nn::Graphormer)
	return cpu(nn)
end

function load_model(nn::Graphormer,lt::learningGNN)
	return nn
end

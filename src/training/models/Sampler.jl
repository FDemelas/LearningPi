"""
Structure that implement the Sampling mechanism from a Gaussian distribution.

# Fields:
- `rng`: random number generator.

An instantiation of this structure can be used as function. 
"""
struct Sampler <: AbstractSampler
    rng
end

"""
# Arguments:
- `x` a vector (of length even), the first half components are the mean `μ` and the last hals the standard deviation σ.

The standard deviation is bounded in [-6,2] ... magic numbers.
The output is a vector of size half the size of `x` sampled from a gaussian of mean `μ` and standard deviation `σ`. 
"""
function (f::Sampler)(x)
	μ, σ2 = MLUtils.chunk(x, 2; dims = 1)
	σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
	σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
	σ2 = exp.(σ2)
	sigma = sqrt.(σ2)
	ϵ = randn(f.rng, Float32, size(σ2)) |> gpu
	z = μ + sigma .* ϵ
	return z
end

# declare the Sampler as Flux functor
Flux.@functor Sampler

"""
Structure that handle the position in the model where is performed the sample.
For the moment only three alternative are available and are encoded in boolean fields.

# Fields:
- `outside`: if true the sampling is performed in the output space.
- `hidden_state`: in all the hidden states between two main blocks.
- `before_decoding`: in the hidden space, but only before call the decoder. 
"""
struct SamplingPosition
    outside::Bool
    hidden_state::Bool
    before_decoding::Bool
end
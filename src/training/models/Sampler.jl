
struct Sampler <: AbstractSampler
    rng
end

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

Flux.@functor Sampler

struct SamplingPosition
    outside::Bool
    hidden_state::Bool
    before_decoding::Bool
end
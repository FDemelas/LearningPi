"""
struct LayerNorm

#Fields:
- `eps`: regularization parameter
- `d`: size of the input of the normalization layer 

Describe the Layer Normalization for the provided parameters
"""
struct LayerNorm
	eps::Float32
	d::Int64
end

"""
function (m::LayerNorm)(x)

Perform a Layer Normalization using `x` as input .
"""
function (m::LayerNorm)(x)
	mean_x = Flux.mean(x, dims = 1)
	sigma_x = sqrt.(Flux.mean((x .- mean_x) .^ 2; dims = 1))

	x_normed = (x .- mean_x) ./ (sigma_x .+ m.eps)

	return x_normed
end

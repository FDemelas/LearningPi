"""
struct RMSNorm

#Fields:
- `eps`: additive regularization parameter
- `sqrtd`: multiplicative regularization parameter

Describe the RMS Normalization for the provided parameters
"""
struct RMSNorm
	eps::Float32
	sqrtd::Float32
end

"""
function create_rms(d)

#Arguments:
- `d`: size of input in the input layers

return a RMSNorm function.
"""
function create_rms(d)
	return RMSNorm(1.0f-8, sqrt(d))
end

"""
function (m::RMSNorm)(x)

Perform a RMS Normalization using `x` as input .
"""

function (m::RMSNorm)(x)
	norm_x = sqrt.(sum(x .^ 2, dims = 1))

	rms_x = norm_x .* m.sqrtd .+ m.eps
	x_normed = x ./ rms_x

	return x_normed
end
"""
# Arguments
	- `fileName` : a path to a json file that contains the data for the instance, features and labels
	- `factory` : an instance factory for the Generalized Assignment problem

This function reads the instance, the features and the labels from the json and returns three structures that contains all the informations.
"""
function dataLoader(fileName::String, factory::cpuGAinstanceFactory)
	ds = JSON.parsefile(fileName)

	I = ds["instance"]["I"]
	J = ds["instance"]["J"]

	p = zeros(Float32, I, J)
	w = zeros(Float32, I, J)
	c = zeros(Int64, I)

	for j in 1:J
		for i in 1:I
			p[i, j] = ds["instance"]["p"][j][i]
			w[i, j] = ds["instance"]["w"][j][i]
		end
		c[j] = ds["instance"]["c"][j]
	end

	ins = cpuInstanceGA(I, J, p, w, c)

	xCR = mapreduce(permutedims, vcat, ds["features"]["linear relaxation"]["x"])
	λ = ds["features"]["linear relaxation"]["lambda"]
	μ = ds["features"]["linear relaxation"]["mu"]
	xRC = mapreduce(permutedims, vcat, ds["features"]["linear relaxation"]["x_rc"])
	λSP = ds["features"]["linear relaxation"]["lambda_sp"]
	μSP = ds["features"]["linear relaxation"]["mu_sp"]

	objCR = ds["features"]["linear relaxation"]["value"]
	xLR = mapreduce(permutedims, vcat, ds["features"]["knapsack relaxation"]["x"])
	objLR = ds["features"]["knapsack relaxation"]["value"]

	λm = Matrix([λ[i][1] for i in eachindex(λ)]')
	λSPm = Matrix([λSP[i][1] for i in eachindex(λSP)]')

	feat = featuresGA(xCR, λm, μ, xRC, λSPm, μSP, objCR, xLR, objLR)

	π = ds["labels"]["pi"]
	πm = Matrix([π[i][1] for i in eachindex(π)]')

	xLR = mapreduce(permutedims, vcat, ds["labels"]["x"])
	Ld = ds["labels"]["Ld"]

	lab = createLabels(πm, xLR, Ld, ins)

	return ins, lab, feat
end

"""
# Arguments:    
- `ins`: instance structure, it should be of type <: instanceGA
- `lab`: labels structure, it should be of type labelsGA
- `feat`: features structure, it should be of type featuresGA
- `fileName`: the path to the file json where print the data

"""
function print_json(ins::instanceGA, lab::labelsGA, feat::featuresGA, fileName::String)
	ds = Dict()

	ds["instance"] = Dict()
	ds["features"] = Dict()
	ds["labels"] = Dict()

	ds["instance"]["J"] = ins.J
	ds["instance"]["I"] = ins.I
	ds["instance"]["p"] = ins.p
	ds["instance"]["w"] = ins.w
	ds["instance"]["c"] = ins.c

	ds["features"]["knapsack relaxation"] = Dict()
	ds["features"]["linear relaxation"] = Dict()

	ds["features"]["knapsack relaxation"]["value"] = feat.objLR
	ds["features"]["knapsack relaxation"]["x"] = [feat.xLR[i, :] for i in 1:ins.I]

	ds["features"]["linear relaxation"]["value"] = feat.objCR
	ds["features"]["linear relaxation"]["x"] = [feat.xCR[i, :] for i in 1:ins.I]

	ds["features"]["linear relaxation"]["lambda"] = feat.λ
	ds["features"]["linear relaxation"]["mu"] = feat.μ
	ds["features"]["linear relaxation"]["x_rc"] = [feat.xRC[i, :] for i in 1:ins.I]

	ds["features"]["linear relaxation"]["lambda_sp"] = feat.λSP
	ds["features"]["linear relaxation"]["mu_sp"] = feat.μSP

	ds["labels"]["pi"] = lab.π
	ds["labels"]["x"] = lab.xLR
	ds["labels"]["Ld"] = lab.objLR

	f = open(fileName, "w")
	JSON.print(f, ds)
	close(f)
end
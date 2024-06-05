"""
# Arguments
	- `fileName` : a path to a json file that contains the data for the instance, features and labels,
	- `factory` : an instance factory for the Capacitated Warehouse Location problem.

It reads the instance, the features and the labels from the json located in `fileName` and returns three structures that contains all the informations.
"""
function dataLoader(fileName::String, factory::cpuCWLinstanceFactory)
	ds = JSON.parsefile(fileName)

	I = ds["instance"]["I"]
	J = ds["instance"]["J"]

	f = zeros(Float32, I)
	d = zeros(Int64, J)
	r = zeros(Float32, I, J)
	q = zeros(Int64, I)

	for i in 1:I
		for j in 1:J
			r[i, j] = ds["instance"]["c"][j][i]
		end
		q[i] = ds["instance"]["demands"][i]
		f[i] = ds["instance"]["f"][i]
	end

	for j in 1:J
		d[j] = ds["instance"]["volumes"][j]
	end

	ins = create_data_object(factory, I, J, r, f, q, d)

	xCR = mapreduce(permutedims, vcat, ds["features"]["linear relaxation"]["x"])
	yCR = ds["features"]["linear relaxation"]["y"]
	λ = ds["features"]["linear relaxation"]["lambda"]
	μ = ds["features"]["linear relaxation"]["mu"]
	xRC = mapreduce(permutedims, vcat, ds["features"]["linear relaxation"]["x_rc"])
	yRC = ds["features"]["linear relaxation"]["y_rc"]
	λSP = ds["features"]["linear relaxation"]["lambda_sp"]
	μSP = ds["features"]["linear relaxation"]["mu_sp"]

	objCR = ds["features"]["linear relaxation"]["value"]
	xLR = mapreduce(permutedims, vcat, ds["features"]["knapsack relaxation"]["x"])
	yLR = ds["features"]["knapsack relaxation"]["y"]
	objLR = ds["features"]["knapsack relaxation"]["value"]
	feat = featuresCWL(xCR, yCR, λ, μ, xRC, yRC, λSP, μSP, objCR, xLR, yLR, objLR)

	π = ds["labels"]["pi"]
	xLR = mapreduce(permutedims, vcat, ds["labels"]["x"])
	yLR = ds["labels"]["y"]
	Ld = ds["labels"]["Ld"]

	lab = createLabels(π, xLR, yLR, Ld, ins)

	return ins, lab, feat
end

"""
# Arguments:    
-`ins`: instance structure, it should be of type  of `instanceCWL`,
- `lab`: labels structure, it should be of type `labelsCWL`,
- `feat`: features structure, it should be of type `featuresCWL`,
- `fileName`: the path to the file json where print the data.

Print the information provided in the instance `ins`, the labels `lab` and the features `feat` in a JSON file located in the path `fileName`.
"""
function print_json(ins::instanceCWL, lab::labelsCWL, feat::featuresCWL, fileName::String)
	ds = Dict()

	ds["instance"] = Dict()
	ds["features"] = Dict()
	ds["labels"] = Dict()

	ds["instance"]["J"] = ins.J
	ds["instance"]["I"] = ins.I
	ds["instance"]["demands"] = ins.q
	ds["instance"]["volumes"] = ins.d
	ds["instance"]["c"] = ins.c
	ds["instance"]["f"] = ins.f

	ds["features"]["knapsack relaxation"] = Dict()
	ds["features"]["linear relaxation"] = Dict()

	ds["features"]["knapsack relaxation"]["value"] = feat.objLR
	ds["features"]["knapsack relaxation"]["x"] = [feat.xLR[k, :] for k in 1:ins.I]
	ds["features"]["knapsack relaxation"]["y"] = feat.yLR

	ds["features"]["linear relaxation"]["value"] = feat.objCR
	ds["features"]["linear relaxation"]["x"] = [feat.xCR[k, :] for k in 1:ins.I]
	ds["features"]["linear relaxation"]["y"] = feat.yCR
	ds["features"]["linear relaxation"]["lambda"] = feat.λ
	ds["features"]["linear relaxation"]["mu"] = feat.μ
	ds["features"]["linear relaxation"]["x_rc"] = [feat.xRC[k, :] for k in 1:ins.I]
	ds["features"]["linear relaxation"]["y_rc"] = feat.yRC
	ds["features"]["linear relaxation"]["lambda_sp"] = feat.λSP
	ds["features"]["linear relaxation"]["mu_sp"] = feat.μSP

	ds["labels"]["pi"] = lab.π
	ds["labels"]["x"] = lab.xLR
	ds["labels"]["y"] = lab.yLR
	ds["labels"]["Ld"] = lab.LR

	f = open(fileName, "w")
	JSON.print(f, ds, 5)
	close(f)
end
"""
# Arguments
	- `fileName` : a path to a json file that contains the data for the instance, features and labels,
	- `factory` : an instance object (for the same instance as the features file).

It reads the instance, the features and the labels from the json and returns three structures that contains all the informations.
"""
function dataLoader(fileName::String, factory::MCNDinstanceFactory)
	ds = JSON.parsefile(fileName)

	n = ds["instance"]["N"]
	edges = [(Integer(ds["instance"]["tail"][ij] + 1), Integer(ds["instance"]["head"][ij]) + 1) for ij ∈ 1:length(ds["instance"]["head"])]
	K = [(Integer(ds["instance"]["o"][k] + 1), Integer(ds["instance"]["d"][k] + 1), Integer(ds["instance"]["q"][k])) for k in 1:length(ds["instance"]["q"])]

	m, p = length(edges), length(K)

	f = ds["instance"]["f"]
	r = hcat(ds["instance"]["r"]...)'

	c = ds["instance"]["c"]

	ins = create_data_object(factory, n, edges, K, f, r, c)

	xCR = hcat(ds["features"]["linear relaxation"]["x"]...)'
	yCR = ds["features"]["linear relaxation"]["y"]
	λ = hcat(ds["features"]["linear relaxation"]["lambda"]...)'
	μ = ds["features"]["linear relaxation"]["mu"]
	objCR = ds["features"]["linear relaxation"]["value"]
	xLR = hcat(ds["features"]["knapsack relaxation"]["x"]...)'
	yLR = ds["features"]["knapsack relaxation"]["y"]
	La = ds["features"]["knapsack relaxation"]["L_a"]
	xRC = hcat(ds["features"]["linear relaxation"]["x_rc"]...)'
	yRC = ds["features"]["linear relaxation"]["y_rc"]
	λSP = hcat(ds["features"]["linear relaxation"]["floow_sp"]...)'
	μSP = ds["features"]["linear relaxation"]["cap_sp"]
	objLR = ds["features"]["knapsack relaxation"]["value"]

	feat = featuresMCND(xCR, yCR, λ, μ, xRC, yRC, λSP, μSP, objCR, xLR, yLR, La, objLR)

	π = hcat(ds["labels"]["pi"]...)
	xLR = hcat(ds["labels"]["x"]...)
	yLR = ds["labels"]["y"]
	La = ds["labels"]["La"]
	Ld = ds["labels"]["Ld"]

	lab = createLabels(π, xLR, yLR, La, Ld, ins)

	return ins, lab, feat
end

"""
# Arguments:    
- `ins`: instance structure, it should be of type `cpuMCNDinstance`,
- `lab`: labels structure, it should be of type `labelsMCND`,
- `feat`: features structure, it should be of type `featuresMCND`,
- `fileName`: the path to the file json where print the data,
- `factory`: instance factory should be of type `cpuMCNDinstanceFactory`.

"""
function print_json(ins, lab, feat, fileName, factory::cpuMCNDinstanceFactory)
	ds = Dict()
	#ds = JSON.parsefile(fileName)

	ds["instance"] = Dict()
	ds["features"] = Dict()
	ds["labels"] = Dict()

	ds["instance"]["N"] = sizeV(ins)
	ds["instance"]["head"] = [head(ins, e) - 1 for e in 1:sizeE(ins)]
	ds["instance"]["tail"] = [tail(ins, e) - 1 for e in 1:sizeE(ins)]
	ds["instance"]["o"] = [origin(ins, k) - 1 for k in 1:sizeK(ins)]
	ds["instance"]["d"] = [destination(ins, k) - 1 for k in 1:sizeK(ins)]
	ds["instance"]["q"] = [volume(ins, k) for k in 1:sizeK(ins)]

	ds["instance"]["f"] = ins.f
	ds["instance"]["r"] = [ins.r[k, :] for k in 1:sizeK(ins)]
	ds["instance"]["c"] = ins.c

	ds["features"]["knapsack relaxation"] = Dict()
	ds["features"]["linear relaxation"] = Dict()
    #	ds["features"]["shortest paths"] = Dict()

	ds["features"]["knapsack relaxation"]["value"] = feat.objLR
	#	ds["features"]["shortest paths"]["distance"] = [feat.distance[i, :] for i in 1:sizeV(ins)]

	ds["features"]["knapsack relaxation"]["x"] = [feat.xLR[k, :] for k in 1:sizeK(ins)]
	ds["features"]["linear relaxation"]["y"] = feat.yCR
	ds["features"]["knapsack relaxation"]["L_a"] = feat.LRarcs
	ds["features"]["linear relaxation"]["value"] = feat.objCR
	ds["features"]["linear relaxation"]["x"] = [feat.xCR[k, :] for k in 1:sizeK(ins)]
	ds["features"]["linear relaxation"]["lambda"] = [feat.λ[k, :] for k in 1:sizeK(ins)]
	#	ds["features"]["shortest paths"]["destination"] = [feat.destinations[k, :] for k in 1:sizeK(ins)]
	#	ds["features"]["shortest paths"]["origin"] = [feat.origins[k, :] for k in 1:sizeK(ins)]
	ds["features"]["knapsack relaxation"]["y"] = feat.yLR
	ds["features"]["linear relaxation"]["mu"] = feat.μ
	ds["features"]["linear relaxation"]["x_rc"] = [feat.xRC[k, :] for k in 1:sizeK(ins)]
	ds["features"]["linear relaxation"]["y_rc"] = feat.yRC
	ds["features"]["linear relaxation"]["floow_sp"] = [feat.λSP[k, :] for k in 1:sizeK(ins)]
	ds["features"]["linear relaxation"]["cap_sp"] = feat.μSP

	# z = SolveDualHard(ins1,1e-3)
	ds["labels"]["pi"] = lab.π
	ds["labels"]["x"] = lab.xLR
	ds["labels"]["y"] = lab.yLR
	ds["labels"]["La"] = lab.LRarcs
	ds["labels"]["Ld"] = lab.objLR

	f = open(fileName, "w")
	JSON.print(f, ds)
	close(f)
end

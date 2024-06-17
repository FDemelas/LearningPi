using LearningPi
using Statistics
using NearestNeighbors
using Distances
using ArgParse

global mean_GAPS_VAL = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global min_GAPS_VAL = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global max_GAPS_VAL = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global std_GAPS_VAL = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global t_VAL = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global mean_GAPS_TEST = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global min_GAPS_TEST = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global max_GAPS_TEST = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global std_GAPS_TEST = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)
global t_TEST = Dict(1 => 0.0, 3 => 0.0, 5 => 0.0, 10 => 0.0, 20 => 0.0)





function main(args)
	s = ArgParseSettings("Training a model with k-fold: " *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option

	@add_arg_table! s begin
		"--data"
		arg_type = String
		required = true
		help = "dataset folder for the training"
		"--factory"
		arg_type = String
		required = true
		help = "factory type"
	end

	parsed_args = parse_args(args, s)
	data = parsed_args["data"]
	factory = contains(parsed_args["factory"], "MCND") ? LearningPi.cpuMCNDinstanceFactory() : LearningPi.cpuGAinstanceFactory()

	for seed in [1, 2, 3]
		l, dS = LearningPi.createKfold(LearningPi.learningMLP(), LearningPi.cr_features_matrix(), data, [-1, -1, -1], seed; factory)

		ins = dS.train.examples_list[1].instance
		f = dS.train.examples_list[1]

		nf = []
		zs = []

		for (i, f) in enumerate(dS.train.examples_list)
			for k in 1:size(f.features, 2)
				for i in 1:size(f.features, 3)
					push!(nf, f.features[:, k, i])
					append!(zs, f.gold.π[k, i])
				end
			end
		end

		nf = reduce(hcat, nf)


		kdtree = KDTree(nf)# BallTree(nf,CosineDist())


		for k in [1, 3, 5, 10, 20]
			GAPS_VAL = []
			GAPS_TEST = []
			times_VAL = []
			times_TEST = []
			for (r, f) in enumerate(dS.val.examples_list)
				ins = f.instance
				nf_val = f.features
				zs_val = f.gold.π
				t0 = time()
				pred = zeros(LearningPi.sizeLM(ins))
				for j in 1:LearningPi.sizeLM(ins)[1]
					for i in 1:LearningPi.sizeLM(ins)[2]
						idxs, dists = knn(kdtree, nf_val[:, j, i], k, true)
						pred[j, i] = mean(zs[idxs])
					end
				end
				objPred, _ = LR(ins, pred)
				append!(times_VAL, time() - t0)
				objGold = f.gold.objLR
				append!(GAPS_VAL, (1 - objPred / objGold) * 100)
			end
			for (r, f) in enumerate(dS.test.examples_list)
				ins = f.instance
				nf_val = f.features
				zs_val = f.gold.π

				t0 = time()
				pred = zeros(LearningPi.sizeLM(ins))
				for j in 1:LearningPi.sizeLM(ins)[1]
					for i in 1:LearningPi.sizeLM(ins)[2]
						idxs, dists = knn(kdtree, nf_val[:, j, i], k, true)
						pred[j, i] = mean(zs[idxs])
					end
				end
				objPred, _ = LR(ins, pred)
				append!(times_TEST, time()-t0)
				objGold = f.gold.objLR
				append!(GAPS_TEST, (1 - objPred / objGold) * 100)
			end

			global mean_GAPS_VAL[k] += mean(GAPS_VAL) / 3
			global min_GAPS_VAL[k] += minimum(GAPS_VAL) / 3
			global max_GAPS_VAL[k] += maximum(GAPS_VAL) / 3
			global std_GAPS_VAL[k] += std(GAPS_VAL) / 3
			global t_VAL[k] += mean(times_VAL) / 3

			global mean_GAPS_TEST[k] += mean(GAPS_TEST) / 3
			global min_GAPS_TEST[k] += minimum(GAPS_TEST) / 3
			global max_GAPS_TEST[k] += maximum(GAPS_TEST) / 3
			global std_GAPS_TEST[k] += std(GAPS_TEST) / 3
			global t_TEST[k] += mean(times_TEST) / 3
		end
	end

	for k in [1, 3, 5, 10, 20]
		println("MEAN for ", k, "-NN")

		println("VALIDATION")
		println("Mean GAP over the validation set:", mean_GAPS_VAL[k])
		println("STD GAP over the validation set:", std_GAPS_VAL[k])
		println("Max GAP over the validation set:", max_GAPS_VAL[k])
		println("Min GAP over the validation set:", min_GAPS_VAL[k])
		println("Mean time ", t_VAL[k])

		println("TEST")
		println("Mean GAP over the test set:", mean_GAPS_TEST[k])
		println("STD GAP over the test set:", std_GAPS_TEST[k])
		println("Max GAP over the test  set:", max_GAPS_TEST[k])
		println("Min GAP over the test set:", min_GAPS_TEST[k])
		println("Mean time ", t_TEST[k])
	end

	println("model & GAP val & GAP test & time val & time test")
	for k in [1, 3, 5, 10, 20]
		println(k, "-NN & ", mean_GAPS_VAL[k], " & ", mean_GAPS_TEST[k], " & ", t_VAL[k], " & ", t_TEST[k], "\\\\")
	end

end


main(ARGS)
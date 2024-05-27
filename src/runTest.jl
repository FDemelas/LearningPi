using LearningPi
using BSON: @load  
using Random: Random, shuffle!
using Statistics
using CUDA
using Flux: gpu, cpu, softplus
using JSON
using MLUtils
using ArgParse

K=5

global m_gap_CR_train, m_gap_LR_train, m_gap_pred_train, m_gap_max_pred_train = 0.0,0.0,0.0,0.0
global m_gap_CR_val,m_gap_LR_val,m_gap_pred_val,m_gap_max_pred_val=0.0,0.0,0.0,0.0
global m_gap_CR_test,m_gap_LR_test,m_gap_pred_test,m_gap_max_pred_test=0.0,0.0,0.0,0.0

global m_perc_CR_train,m_perc_LR_train,m_perc_pred_train,m_perc_max_pred_train=0.0,0.0,0.0,0.0
global m_perc_CR_val,m_perc_LR_val,m_perc_pred_val,m_perc_max_pred_val=0.0,0.0,0.0,0.0
global m_perc_CR_test,m_perc_LR_test,m_perc_pred_test,m_perc_max_pred_test=0.0,0.0,0.0,0.0

global CR_time_train=[]
global CR_time_val=[]
global CR_time_test=[]
	
global predsK_time_train,LRk_time_train = [],[]
global preds_time_train,LR_time_train,FE_time_train = [],[],[]
global predsK_time_val,LRk_time_val = [],[]
global preds_time_val,LR_time_val,FE_time_val = [],[],[]

global predsK_time_test,LRk_time_test = [],[]
global preds_time_test,LR_time_test,FE_time_test = [],[],[]
	
function print_latex_table(values,row_names::Vector{String},col_names::Vector{String},caption::String=" ")
	if size(values,1) != length(row_names) || size(values,2) != length(col_names)
			println("Error in size of row or columns labels")	
			return
	else
		sR=size(values,1)
		sC=size(values,2)	
	end 
	println("\\begin{table}")
	print("\\begin{tabular}{l|")
	for _ in 1:sC
		print("r")
	end
	println("}")
	for c in col_names
		print(" & ",c)
	end
	println(" \\\\")
	println("\\hline")
	for i in 1:sR
		print(row_names[i])
		for j in 1:sC
			print(" & ",round(values[i,j];digits=1))
		end
		println(" \\\\")
	end
	println("\\end{tabular}")
	println("\\caption{ ",caption," }")
	println("\\end{table}")
end

function p(gnn, feats,  zCR, instance,K)
        sizek, sizev = sizeK(instance),sizeV(instance)
        nbpreds = prod(LearningPi.sizeLM(instance))
        device = cpu

        @views h = (gnn.model[1](feats, feats.ndata.x))[:, 1:nbpreds]
        μ, σ2 = MLUtils.chunk(h, 2; dims = 1)

        σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
        σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)

        σ2 = exp.(σ2)
        sigma = sqrt.(σ2)
        pred=[]
        for k in 1:K
                ϵ = randn(gnn.rng, Float32, size(σ2)) |> gpu
                z = μ + sigma .* ϵ

                delta = reshape(gnn.model[2](z), sizek, sizev) |> device

                push!(pred, device(zCR) .+ device(delta))
        end
        return pred
end

function fill_tables!(ds,dataset,factory,lt,fmt,nn1,crs,golds,lrs,predsK,preds,predsK_time,preds_time,LRk_time,LR_time,CR_time,FE_time,K)
	for instPath in ds[dataset]             
		inst, lab, feat = dataLoader(instPath,factory)

	        t0=time()
		ϕ=featuresExtraction(lt,feat,inst,fmt)
		append!(FE_time,time()-t0)	

		append!(crs,feat.objCR)
		append!(golds,lab.objLR)

		t0=time()
        	pred = nn1(gpu( ϕ ))
        	append!(preds_time,time()-t0)
                append!(lrs,LR(inst,zeros(size(pred)))[1])
		


		t0=time()
		obj,_ = LR(inst,pred)
                append!(LR_time,time()-t0)

		append!(preds,obj)

                t0=time()
                objCR,_ = CR(inst)
                append!(CR_time,time()-t0)

		
		tzK=[]
		tLRK=[]
		pK=[]
		 t0=time()
                pred  = p(nn1, ϕ |> gpu,model_device(feat.λ) ,inst,K) |> cpu
                append!(predsK_time,time()-t0)
                for  k in 1:K                         
                        t0=time()
                        obj,_ = LR(inst,pred[k])
                        append!(tLRK,time()-t0)

                        append!(pK,obj)
                end

                push!(predsK,pK)
                push!(LRk_time,tLRK)     
	end
end

function print_res(dataset,crs,lrs,golds,preds,max_preds)
	println("\n","Results on ",dataset, " set","\n")	

	gap_CR=mean(1 .- crs./golds)*100
	gap_LR=mean(1 .- lrs./golds)*100
	gap_pred= mean(1 .- preds./golds)*100
	gap_max_pred=mean(1 .- max_preds./golds)*100
	
	perc_CR=mean((1 .- crs./golds)*100 .<= 0.1)*100
	perc_LR=mean((1 .- lrs./golds)*100 .<= 0.1)*100
	perc_pred= mean((1 .- preds./golds)*100 .<= 0.1)*100
	perc_max_pred=mean((1 .- max_preds./golds)*100 .<= 0.1)*100
	
	
	println("Base")
	println("Continuous Relaxation GAP ", gap_CR)
	println("Lagrangian Relaxation GAP ", gap_LR)
	println("Ex-Full Prediction GAP ",gap_pred)
	println("Max-5 Prediction GAP ", gap_max_pred)
	println(" ")
	return gap_CR,gap_LR,gap_pred,gap_max_pred,perc_CR,perc_LR,perc_pred,perc_max_pred
end


function print_times(dataset,K,predsK_time,preds_time,LRk_time,LR_time,FE_time)
	println("\n","Results on ",dataset, " set","\n")
	println("Features Extraction mean (+- var) time ", mean(FE_time), " (+- ",std(FE_time), " )")
 	println()
	println("For a single prediction we have a mean time of ")
	println("Prediction mean (+- var) time ", mean(preds_time), " (+- ",std(preds_time), " )")
        println("Resolution of the Sub-Problem mean (+- var) time ", mean(LR_time), " (+- ",std(LR_time), " )")
	println()
	println("For ", string(K), " predictions ")
	println("Prediction mean (+- var) time ", mean([sum(predsK_time[i]) for i in eachindex(predsK_time)]), " (+- ",std([sum(predsK_time[i]) for i in eachindex(predsK_time)]), " )")
        println("Resolution of the Sub-Problem mean (+- var) time ", mean([sum(LRk_time[i]) for i in eachindex(LRk_time)]), " (+- ",std([sum(LRk_time[i]) for i in eachindex(LRk_time)]), " )")
	println(" ")
end

function main(args)
	s = ArgParseSettings("Training a model with k-fold: " *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option

	@add_arg_table! s begin
		"--modelFolders"
		required = true
		nargs = '*'
		help = "models that we want consider"
		"--seeds"
		required = true
		nargs = '*'
		help = "random seeds in the modelFolders"
		"--learningType"
		required = true
		arg_type = String
		help = "Learning Type, for the selection of the model type"
		"--factory"
		required = true
		arg_type = String
		help = "Factory, for the selection of the instance type"
		"--featuresMatrixType"
        nargs = 1
        arg_type = String
	end

	CUDA.math_mode()

	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)

    lt = get_from_LearningPi(parsed_args["learningType"])
    factory = get_from_LearningPi(parsed_args["factory"])
    fmt = get_from_LearningPi(parsed_args["featuresMatrixType"][1])
    model_folders=parsed_args["modelFolders"]
    seeds=parsed_args["seeds"]
    for (seed,sub_string) in [(seeds[i],model_folders[i]) for i in eachindex(model_folders)]
        path = "/users/demelas/learning_pi_slim.jl/launchers/ICML_runs/run/HL_" * sub_string * "/"
        model_path = path * "bestValGap_" * sub_string * ".bson"
        data_path=path*"dataset.json"
    
        f = JSON.open(data_path,"r")
        ds = JSON.parse(f)
        close(f)
    
        @load model_path nn
        nn = nn |> gpu
        nn1 = nn
        
        crs_train, lrs_train, golds_train = [],[],[]
        crs_val,   lrs_val,   golds_val   = [],[],[]
        crs_test,  lrs_test,  golds_test  = [],[],[]
        preds_train, predsK_train = [],[]
        preds_val,   predsK_val   = [],[]
        preds_test,  predsK_test  = [],[]
        
        fill_tables!(ds,"train",factory,lt,fmt,nn1,crs_train,golds_train,lrs_train,predsK_train,preds_train,predsK_time_train,preds_time_train,LRk_time_train,LR_time_train,CR_time_train,FE_time_train,K)
        fill_tables!(ds,"val",factory,lt,fmt,nn1,crs_val,golds_val,lrs_val,predsK_val,preds_val,predsK_time_val,preds_time_val,LRk_time_val,LR_time_val,CR_time_val,FE_time_val,K)
        fill_tables!(ds,"test",factory,lt,fmt,nn1,crs_test,golds_test,lrs_test,predsK_test,preds_test,predsK_time_test,preds_time_test,LRk_time_test,LR_time_test,CR_time_test,FE_time_test,K)
    
        max_preds_train = [maximum(predsK_train[i]) for i in eachindex(predsK_train)]
        max_preds_val = [maximum(predsK_val[i]) for i in eachindex(predsK_val)]
        max_preds_test = [maximum(predsK_test[i]) for i in eachindex(predsK_test)]
    
        gap_CR_train,gap_LR_train,gap_pred_train,gap_max_pred_train,perc_CR_train,perc_LR_train,perc_pred_train,perc_max_pred_train = print_res("train",crs_train,lrs_train,golds_train,preds_train,max_preds_train)
        gap_CR_val,gap_LR_val,gap_pred_val,gap_max_pred_val,perc_CR_val,perc_LR_val,perc_pred_val,perc_max_pred_val = print_res("val",crs_val,lrs_val,golds_val,preds_val,max_preds_val)
        gap_CR_test, gap_LR_test, gap_pred_test, gap_max_pred_test,perc_CR_test,perc_LR_test,perc_pred_test,perc_max_pred_test = print_res("test",crs_test,lrs_test,golds_test,preds_test,max_preds_test)
    
        print_times("train",K,predsK_time_train,preds_time_train,LRk_time_train,LR_time_train,FE_time_train)
        print_times("val",K,predsK_time_val,preds_time_val,LRk_time_val,LR_time_val,FE_time_val)
        print_times("test",K,predsK_time_test,preds_time_test,LRk_time_test,LR_time_test,FE_time_test)
    
        global	m_gap_CR_train += gap_CR_train/3
        global	m_gap_LR_train += gap_LR_train/3
        global	m_gap_pred_train += gap_pred_train/3
        global	m_gap_max_pred_train += gap_max_pred_train/3
        
        global	m_gap_CR_val += gap_CR_val/3
        global	m_gap_LR_val += gap_LR_val/3
        global	m_gap_pred_val += gap_pred_val/3
        global	m_gap_max_pred_val += gap_max_pred_val/3
        
        global	m_gap_CR_test += gap_CR_test/3
        global	m_gap_LR_test += gap_LR_test/3
        global	m_gap_pred_test += gap_pred_test/3
        global	m_gap_max_pred_test += gap_max_pred_test/3
        
        global	m_perc_CR_train += perc_CR_train/3
        global	m_perc_LR_train += perc_LR_train/3
        global	m_perc_pred_train += perc_pred_train/3
        global	m_perc_max_pred_train += perc_max_pred_train/3
        
        global	m_perc_CR_val += perc_CR_val/3
        global	m_perc_LR_val += perc_LR_val/3
        global	m_perc_pred_val += perc_pred_val/3
        global	m_perc_max_pred_val += perc_max_pred_val/3
        
        global	m_perc_CR_test += perc_CR_test/3
        global	m_perc_LR_test += perc_LR_test/3
        global	m_perc_pred_test += perc_pred_test/3
        global	m_perc_max_pred_test += perc_max_pred_test/3	
    end	
    
    println("model \t & GAP train \t & GAP val \t GAP test \t & & time train \t & time val \t & & time test \t \\")
    println(" CR   \t & ",	m_gap_CR_train, " \t & ",m_gap_CR_val, " \t & ",m_gap_CR_test , " \t & & ",mean(CR_time_train), " \t & ",mean(CR_time_val), " \t & ", mean(CR_time_test)," \t & ",	m_perc_CR_train, " \t & ",m_perc_CR_val, " \t & ",m_perc_CR_test ,"\t \\" )
    println(" LR(CR)   \t & ",	m_gap_LR_train, " \t & ",m_gap_LR_val, " \t & ",m_gap_LR_test , " \t & & ",mean(LR_time_train), " \t & ",mean(LR_time_val), " \t & ", mean(LR_time_test)," \t & ",	m_perc_LR_train, " \t & ",m_perc_LR_val, " \t & ",m_perc_LR_test,"\t \\" )
    println(" PRED   \t & ",	m_gap_pred_train, " \t & ",m_gap_pred_val, " \t & ",m_gap_pred_test , " \t & & ",mean(preds_time_train), " + ",mean(FE_time_train) ," \t & ",mean(preds_time_val), " + ",mean(FE_time_val) , " \t & ", mean(preds_time_test), " + ",mean(FE_time_test) ," \t & ","\t & ",m_perc_pred_train, " \t & ",m_perc_pred_val, " \t & ",m_perc_pred_test , )
    println(" MAX-5   \t & ",	m_gap_max_pred_train, " \t & ",m_gap_max_pred_val, " \t & ",m_gap_max_pred_test , " \t & & ", sum(mean(predsK_time_train)), " \t & ", sum(mean(predsK_time_val)), " \t & ",  sum(mean(predsK_time_test))+mean(CR_time_test)+sum(mean(LRk_time_test))+mean(FE_time_test)," \t & ",	m_perc_max_pred_train, " \t & ",m_perc_max_pred_val, " \t & ",m_perc_max_pred_test ,"\t \\")

    println()
    println("Features extraction time: train",mean(FE_time_train)  , " val ",mean(FE_time_val) , " test ",mean(FE_time_test))

end

function get_from_LearningPi(symb_string::String)
    symb = Symbol(symb_string)
    return getfield(LearningPi, symb)()
end


main(ARGS)

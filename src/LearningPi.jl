module LearningPi

using ArgParse
using Flux: Adam, ADAM, RMSProp, RAdam, AdaMax, AMSGrad, NAdam, AdamW, AdaBelief, Momentum, Descent, Dense, relu, gradient, Chain, ExpDecay, Parallel, ClipNorm, Optimiser, Flux, σ, gpu, cpu, BatchNorm, Dropout, softplus
using MLUtils
using JuMP, HiGHS
using Dates
using Distances
using ChainRules, ChainRulesCore
using Random: shuffle, shuffle!, MersenneTwister, AbstractRNG, seed!
using Statistics, StatsFuns
using StatsBase: nquantile
using LinearAlgebra, SparseArrays
using CUDA
using Random: Random
using JSON: JSON
using BSON: @save, @load
using Flux: Flux
using Flux.Optimise: AbstractOptimiser
using StatsBase
using SimpleWeightedGraphs, Graphs
using GraphNeuralNetworks
using TensorBoardLogger, Logging
using ProgressMeter

#assure the good CUDA version, for the cluster
#CUDA.set_runtime_version!(v"11.4")

push!(LOAD_PATH, "../instances.jl/src/")
using Instances

global has_gpu = CUDA.functional()

if has_gpu
    to_gpu_or_not_to_gpu(x::AbstractArray) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x::AbstractArray) = x
end

global model_device = has_gpu ? gpu : cpu

include("training/models/abstract.jl")
include("training/deviation/deviation.jl")
include("dataStructures/featuresMatrix.jl")
include("dataStructures/Abstract/dataSet.jl")

include("dataStructures/GA/features.jl")
include("dataStructures/GA/labels.jl")
include("dataStructures/GA/dataset.jl")

include("dataStructures/CWL/features.jl")
include("dataStructures/CWL/labels.jl")
include("dataStructures/CWL/dataset.jl")

include("dataStructures/MCND/features.jl")
include("dataStructures/MCND/labels.jl")
include("dataStructures/MCND/dataset.jl")

include("training/UGCNConv.jl")
include("training/loss/abstractLoss.jl")
include("training/loss/LRloss.jl")
include("training/loss/gpuLRloss.jl")
include("training/loss/GAPloss.jl")
include("training/loss/GAPcloseLoss.jl")
include("training/loss/HingeLoss.jl")
include("training/loss/MSE.jl")
include("training/loss/multiPredLoss.jl")

include("training/normalization/LayerNorm.jl")
include("training/normalization/RMSNorm.jl")

include("training/models/Mlp.jl")
include("training/models/datasetGNN.jl")
include("training/models/Sampler.jl")
include("training/models/GraphormerBlock.jl")
include("training/models/Graphormer.jl")
#include("training/models/Graphormer2.jl")
include("training/models/ModelFactory.jl")

include("training/training.jl")

export abstractInstance, abstractInstanceMCND, cpuInstanceMCND, gpuInstanceMCND
export labelsExtraction, featuresExtraction
export prediction, target
export inputSize
export create_loss
export ComputeGAP, ComputeGAPset
export printOutResults
export abstractInstanceFactory, MCNDinstanceFactory,cpuMCNDinstanceFactory,gpuMCNDinstanceFactory, CWLinstanceFactory,cpuCWLinstanceFactory, cpuInstanceCWL, instanceCWL
export create_example
export createDataSet, dataLoader
export createCorpus, createKfold
export dictValuesToVector
export featuresLoader
export tail, head, capacity, fixed_cost, routing_cost, origin, destination, volume
export sizeK, sizeV, sizeE, b, isInKij
export outdegree, indegree
export load_data
export labelsLoader
export cijk
export checkDims
export LR, value_LR_a, value_LR, constantLagrangianBound
export create_model
export train
export loss_value_gradient
export hinge, hingeLoss, hingeMSE, SVR
export gradient, trainHingeLoss
export LR_value_gradient
export SolveDual
export objective_coefficient_type
export dataLoader
export create_data_object
export createLabels
export CR
end

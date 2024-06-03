This page is only devoted to describe how easily launch the training and the test.

## Training

The training the GNN models presented in the paper can be performed using the script in runKfoldTransformer.jl

For example, assuming that we are in the main directory of the project, with the command:

```shell
julia --project=. ../../src/runKfoldTransformer.jl --lr 0.0001 --seed 1 --decay 0.9 --opt rADAM --MLP_i 250 --MLP_h 1000 --MLP_f 250 --lossType LRloss --lossParam 0.9 --maxEp 300 --stepSize 1000000 --kFold 1 --data /users/demelas/MCNDsmallCom40/ --block_number 5 --nodes_number 500 --pDrop 0.25 --factory cpuMCNDinstanceFactory --learningType learningSampleTransformer --cr_deviation true --cr_features true
```

we train one `learningSampleTransformer` (`learningType`) model composed by `5` blocks (`block_number`) and an hidden state representation of size `500` (`--nodes_number`).
The first MLP that goes from the features space to the hidden representation is composed of only one hidden layer with `250` nodes (`--MLP_i`).
Also the Decoder has only one hidden layer of size `250`  (`--MLP_f`).
Instead the hidden MLP always goes from the hidden state representation to a space 2 times bigger (`--MLP_h`).
The drop-out probability is setted to `0.25` (`--pDrop`).

The training is performed on the dataset composed by `cpuMCNDinstanceFactory` (`--factory`) that can be found in the path `/users/demelas/MCNDsmallCom40/` (`--data`).
To construct the dataset we use the seed `1` (`--seed`) and it will be the same seed used to construct and initialize the model.
When the dataset is divised in folds (using the provided seed) the fold `1` is selected as test set ('--kFold`).
Notice that even if we change only the seed, but keep the same k-fold, actually the training, validation and test will not be the same.

The Optimizer used for the training is `rADAM` (`--opt`) with learning rate `0.0001` (`--lr`) and decay `0.9` (`--decay`).
The step size for the decay (i.e. number of samples before update the learning rate) is `1000000`.

The model, tensorboard logs and other information about the training/validation/test sets and the hyper-parameters choices are saved in a sub-directory of the directory `run` that will be created in the directory where we launch the training.

To launch the experiments we use slurm file of the type. 

```shell
#!/usr/bin/env bash
#SBATCH --job-name=T-1-5-s0
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=1     
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1       
#SBATCH --qos=qos_gpu_t4        
#SBATCH --output=output.txt    
#SBATCH --error=error.txt 

export JULIA_NUM_THREADS=1
julia --project=../.. ../../src/runKfoldTransformer.jl --lr 0.0001 --seed 1 --decay 0.9 --opt ADAM --MLP_i 250 --MLP_h 1000 --MLP_f 250 --lossType LRloss --lossParam 0.9 --maxEp 300 --stepSize 1000000 --kFold 1 --data /users/demelas/MCNDsmallCom40/ --block_number 5 --nodes_number 500 --pDrop 0.25 --factory cpuMCNDinstanceFactory --learningType learningSampleTransformer --cr_deviation true --cr_features true
```

## Testing

Documentation Work in Progress...
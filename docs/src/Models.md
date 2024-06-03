## General Models

### MLP

A Classic Multi Layer Perceptron that receeives as input one features vector (for each dualized constraint) and predict (in parallel) one Lagrangian Multiplier Value (for each relaxed constraint).

### GNN Models

The key models of this project belongs from this family and can be all seen as particular instantiation of the more general model implemented in `Graphormer.jl`.

The key component of this model consists on the block presented in [^1].

## Some Easy-to-Repeat Models

In this section we will present some GNN models that are already implemented as partticular instantiation of the model defined in `Graphormer.jl`.

All the constructors for this models can be found in the file `ModelFactory.jl`.

### LearningTransformer

Simply consists on sequential Chain of the block presented in [^1], without Sampling Mechanism.

### LearningSampleTransformer

As `LearningTransformer`, the only difference is that for this model we consider the Sampling mechanism, as presented in [^1]. 

### LearningSampleGasse

As `LearningSampleTransformer` this architecture consider the same Sampling Mechanism presented in [^1].
Insted of using ours architecture it use one more near to the one presented by Gasse et al. in [^2].

### LearningSampleNair

As `LearningSampleTransformer` this architecture consider the same Sampling Mechanism presented in [^1].
Insted of using ours architecture it use one more near to the one presented by Nair et al. in [^3]


### LearningSampleOutside 

As `LearningTransformer`, the only difference is that for this model we consider a sampling mechanism.
While `LearningSampleTransformer` sample in the hidden space (as presented in [^1]), in this case we sample directly in the output space.
More details on the sampling mechanism can be found in the apposite Section. 

### LearningMultiPredTransformer

This model as the same inner structure as `LearningTransformer`, but it contains several decoders and so is able to provide several Lagrangian Multipliers prediction using the same model (maximum one for block).
The model `LearningTransformer` can be seen as this model with only one decoder at the end of the Block Chain.
No sample mechanism is used in this case.

### LearningMultiPredSample

This model as the same inner structure as `LearningSampleTransformer`, but it contains several decoders and so is able to provide several Lagrangian Multipliers prediction using the same model (maximum one for block).
The model `LearningSampleTransformer` can be seen as this model with only one decoder at the end of the Block Chain.
The sampling mechanism is the same as `LearningSampleTransformer` for each predicted Lagrangian Multipliers vector.


## References:

[^1]: F. Demelas, J. Le Roux, M. Lacroix, A. Parmentier "Predicting Lagrangian Multipliers for Mixed Integer Linear Programs", ICML 2024.

[^2]: Gasse, M., Chételat, D., Ferroni, N., Charlin, L., and Lodi, A. Exact Combinatorial Optimization with Graph Convolutional Neural Networks. In Wallach, H., Larochelle, H., Beygelzimer, A., Alché-Buc, F. d., Fox, E., and Garnett,R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[^3]: Nair, V., Bartunov, S., Gimeno, F., von Glehn, I., Lichocki, P., Lobov, I., O’Donoghue, B., Sonnerat, N., Tjandraatmadja, C., Wang, P., Addanki, R., Hapuarachchi, T., Keck, T., Keeling, J., Kohli, P., Ktena, I., Li, Y., Vinyals, O., and Zwols, Y. Solving mixed integer programs using neural networks. CoRR, abs/2012.13349, 2020.
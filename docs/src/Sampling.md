In this section we will provided further details about the Sampling Mechanism implemented in the Package.

## Sampling Type

The package dispose of only one type of Sampling.

Anyway this sampling can be enbedded into the models in three differents ways, providing different architectures:
1) In the hidden space representation as [^1], before use the decoder.
2) In all the hidden space representations. As the previous the sampling is performed in the hidden space, the difference is that the sampling in this case is performed between all the blocks and not only before the decoder. 
3) In the output space, in this case the decoder returns one vector of size two for each dualized constraint representing the mean and the standard deviation of a Gaussian distribution that directly sample in the Lagrangian Multipliers Space.

## References

[^1]: F. Demelas, J. Le Roux, M. Lacroix, A. Parmentier "Predicting Lagrangian Multipliers for Mixed Integer Linear Programs", ICML 2024.

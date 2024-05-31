# Learning_Lagrangian_Multipliers.jl



This package provides the implementation of the learning framework presented in: 

"Predicting Lagrangian Multipliers for Mixed Integer Linear Programs"

Francesco Demelas, Joseph Le Roux, Mathieu Lacroix, Axel Parmentier 

It depends on the package:

https://github.com/FDemelas/Instances

where we develop the instance encoding and the resolution of the Lagrangian Sub-Problem and the Continuous Relaxation.

The dataset used in the paper can be found at:

https://github.com/FDemelas/datasets_learningPi

## Getting started

```
julia --project=.
```

or equivalently:

```
julia
```

and then once in Julia

```julia
using Pkg;
Pkg.activate(".");
```

To install all the dependencies it is sufficient to use this commands:

```julia
using Pkg;
Pkg.instantiate();
```

If you find some problem with the package Instances.jl you can install it using:

 ```julia
using Pkg
Pkg.add(url="https://github.com/FDemelas/Instances")
```

then the package can be used 






## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

```julia
using LearningPi
```

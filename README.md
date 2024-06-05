# Learning_Lagrangian_Multipliers.jl
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://fdemelas.github.io/Learning_Lagrangian_Multipliers.jl/dev/)
[![Documentation](https://github.com/FDemelas/Learning_Lagrangian_Multipliers.jl/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/FDemelas/Learning_Lagrangian_Multipliers.jl/actions/workflows/documentation.yml)
[![CC BY 4.0][cc-by-shield]][cc-by]

This package provides the implementation of the learning framework presented in: 

 F. Demelas, J. Le Roux, M. Lacroix, A. Parmentier "Predicting Lagrangian Multipliers for Mixed Integer Linear Programs", ICML 2024. 

It depends on the package:

https://github.com/FDemelas/Instances

where we develop the instance encoding and the resolution of the Lagrangian Sub-Problem and the Continuous Relaxation.

The dataset used in the paper can be found at:

https://github.com/FDemelas/datasets_learningPi

## Getting started

To install all the dependencies it is sufficient to use this commands in Julia:

```julia
using Pkg;
Pkg.activate(".");
Pkg.instantiate();
```

If you find some problem with the package Instances.jl you can install it using:

 ```julia
using Pkg
Pkg.activate(".");
Pkg.add(url="https://github.com/FDemelas/Instances")
Pkg.instantiate();
```

then the package can be used 

```julia
using LearningPi
```

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

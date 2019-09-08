### Requirements
* Julia Lang v1.0+
* Additional Julia packages: MAT, MathProgBase, Clp, Distributions

### Usage
1. Install the aforementioned packages in Julia with the command `using Pkg; Pkg.add("<package name>")`.

2. Start Julia with `julia -p n` where `n` denotes the number of worker processes.

3. In Julia, type `include("XX_main_YY.jl");` to load a main function and then run the main function, e.g.,
``` julia
# Example: run centralized methods: SCG, Projected SGD, STORM, and SCG++
include("movie_main_cen_concave.jl");
res_CenSCG, res_CenPSGD, res_CenSTORM, res_CenSCGPP = movie_main_cen_concave(2000, 1900, 5, 10, 10, true);
```
See the main files for detailed descriptions of the function arguments.

### Notes
* The number of computing agents should be 50, otherwise one has to generate the network and the partitioned data set before running the main functions.

* There are two data sets for the movie recommendation application, one contains 100K entries of rating, and another contains 1M. The "100K" data set is used by default. To use the "1M" data set, please changed "100K" to "1M" in the main files.

### Directory Structure
* {movie, nqp}_main_{det, stochastic}.jl -- main functions to test algorithms, where {"movie", "nqp"} is the set of available problems, "det" ("stochastic") denotes that the main file is used to test deterministic ("stochastic") methods.
* models/ -- containing files that define different models (facility location and NQP)
* algorithms/ -- containing files that define different algoritms (centralized CG, Decentralized CG proposed by Mokhtari, and DeGTFW proposed by us).
* data/ -- containing data files
* comm.jl -- handy functions for loading data

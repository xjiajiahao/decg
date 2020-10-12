### Requirements
* Julia Lang v1.0+
* Additional Julia packages: MAT, MathProgBase, Clp, Distributions

### Usage
1. Download the movie recommendation datasets: [MovieLens 100K](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and [MovieLens 1M](http://files.grouplens.org/datasets/movielens/ml-1m.zip).
Extract the zip files to the folder `./data/`.
Then launch MATLAB from the directory `./utils` and type
``` matlab
gen_data_movie_1M;  % generate the base dataset ./data/Movies_1M.mat
gen_data_movie_100K;  % generate the base dataset ./data/Movies_100K.mat
```

2. Install the aforementioned packages in Julia using the command `using Pkg; Pkg.add("<package name>")`.

3. Generate the data file and run the main script.
``` julia
# Example 1. test centralized methods on the concave over modular loss
#
# step 1. launch MATLAB from the directory `./utils` and type
num_nodes = 1;
gen_partitioned_data_movie_1M(num_nodes);  # the default dataset is MovieLens 1M, see line 19 of `./movie_main_cen_concave.jl`
#
# step 2. launch JULIA from the the current directory `./` and type
include("movie_main_cen_concave.jl");
res_CenSCG, res_CenPSGD, res_CenSTORM, res_CenSCGPP = movie_main_cen_concave(10, 10, 100, 2, 10, true);  # see the main file for detailed descriptions of the function arguments.
```
``` julia
# Example 2. test stochastic decentralized methods on the facility location loss
#
# step 1. launch MATLAB from the directory `./utils` and type
num_nodes = 16;
gen_partitioned_data_movie_100K(num_nodes);  # the default dataset is MovieLens 100K, see line 20 of `./movie_main_stoch.jl`
gen_weight_matrix(num_nodes, 'er', 0.4);  # generate the weight matrix of the network
#
# step 2. launch JULIA from the the current directory `./` and type
include("movie_main_stochastic.jl");
res_DeSCG, res_DeSGTFW, res_AccDeSGTFW, res_CenSCG =  movie_main_stochastic(10, 10, 100, 2, "er", 16, 10, false);  # see the main file for detailed descriptions of the function arguments.
```
Note that in Example 2, one can use the command `julia -p n` to launch JULIA with `n` worker processes, e.g., `julia -p 4`.

### Directory Structure
``` bash
├── README.md
├── algorithms  # centralized and decentralized algorithms
│   ├── AccDeGTFW.jl  # accelerated DeGTFW
│   ├── CenCG.jl  # centralized Conditional Gradient (CG)
│   ├── CenPGD.jl  # centralized Projected Gradient Descent (PGD)
│   ├── CenSCGPP.jl  # centralized SCG++ [1]
│   ├── CenSTORM.jl  # centralized STORM
│   ├── DeCG.jl  # decentralized Conditional Gradient [2]
│   └── DeGTFW.jl  # decentralized Gradient Tracking Frank-Wolfe [3]
├── data  # contains datasets and results in MAT format
│   ├── ...
│   └── ...
├── models  # objective function definitions
│   ├── concave_over_modular.jl  # concave over modular
│   ├── facility_location.jl  # facility location
│   └── nqp.jl  # nonconvex quadratic programming
├── movie_main_cen_concave.jl  # main function to test centralized methods on the concave over modular loss
├── movie_main_cen_facility.jl  # main function to test centralized methods on the facility location loss
├── movie_main_det.jl  # main function to test deterministic decentralized methods on the facility location loss
├── movie_main_stochastic.jl  # main function to test stochastic decentralized methods on the facility location loss
├── nqp_main_det.jl  # main function to test deterministic decentralized methods on the NQP
├── nqp_main_stochastic.jl  # main function to test stochastic decentralized methods on the NQP
└── utils
    ├── README.md
    ├── comm.jl  # handy functions to load data
    ├── gen_data_movie_100K.m  # generate the base dataset ./data/Movies_100K.mat
    ├── gen_data_movie_1M.m  # generate the base dataset ./data/Movies_1M.mat
    ├── gen_data_nqp_toy.m  # generate a toy dataset in MAT format for NQP
    ├── gen_partitioned_data_movie_100K.m  # partition the ./data/Movies_100K.mat dataset to different nodes in the network
    ├── gen_partitioned_data_movie_1M.m    # partition the base ./data/Movies_1M.mat dataset to different nodes in the network
    └── gen_weight_matrix.m  # generate a weight matrix of the network
```

[1] Hassani, H., Karbasi, A., Mokhtari, A., & Shen, Z. (2019). Stochastic conditional gradient++. arXiv preprint arXiv:1902.06992.

[2] Mokhtari, A., Hassani, H. & Karbasi, A.. (2018). Decentralized Submodular Maximization: Bridging Discrete and Continuous Settings. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:3616-3625.

[3] Xie, J., Zhang, C., Shen, Z., Mi, C. & Qian, H.. (2019). Decentralized Gradient Tracking for Continuous DR-Submodular Maximization. Proceedings of Machine Learning Research, in PMLR 89:2897-2906.

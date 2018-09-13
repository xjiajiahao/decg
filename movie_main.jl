using LaTeXStrings

include("facility.jl");
include("algorithms.jl");
include("comm.jl");

# Step 1: initialization
const k_int = 10;  # the cardinality constraint
const num_agents = 100;
const num_iters = floor(Int, 1e3);

# load data
# data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user
const data_cell, num_movies, num_users = load_movie_partitioned_data(num_agents);
const dim = num_movies;
const k = Float64(k_int);

x0 = zeros(dim);

# generate LMO
d = ones(dim);
a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
LMO = generate_linear_prog_function(d, a_2d, k);

res_CenFW = CenFW(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, num_iters);

using LaTeXStrings

include("facility.jl");
include("algorithms/CenFW.jl"); include("algorithms/DeFW.jl"); include("algorithms/DeSAGAFW.jl");
include("comm.jl");

# Step 1: initialization
const k_int = 10;  # the cardinality constraint
# const num_agents = 100;
const num_agents = 50;
const num_iters = floor(Int, 1e1);
alpha = 1/sqrt(num_iters);
phi = 1/num_iters^(2/3);

# load data
# data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user
const data_cell, num_movies, num_users = load_movie_partitioned_data(num_agents);

# load weights matrix
# const weights = generate_network(num_agents, avg_degree);
const weights = load_network_50();
num_out_edges = count(i->(i>0), weights) - num_agents;

const dim = num_movies;
const k = Float64(k_int);

x0 = zeros(dim);

# generate LMO
d = ones(dim);
a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
LMO = generate_linear_prog_function(d, a_2d, k);

# res_CenFW = CenFW(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, num_iters);
# res_DeFW = DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, alpha);
# res_DESAGAFW = DeSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters);

# res_DeSFW = DeSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters, alpha, phi);

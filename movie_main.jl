using LaTeXStrings

include("facility.jl");
include("algorithms/CenFW.jl"); include("algorithms/DeFW.jl"); include("algorithms/DeSAGAFW.jl"); include("algorithms/CenGreedy.jl");
include("comm.jl");

# Step 1: initialization
const k_int = 10;  # the cardinality constraint
# const num_agents = 100;
const num_agents = 50;
# const num_iters = Int(2e2);
# alpha = 1/sqrt(num_iters);
# phi = 1/num_iters^(2/3);

# load data
# data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user
const data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents);

# load weights matrix
# const weights = generate_network(num_agents, avg_degree);
const weights = load_network_50("complete");
# const weights = load_network_50("line");
# const weights = load_network_50("er");
num_out_edges = count(i->(i>0), weights) - num_agents;

const dim = num_movies;
const k = Float64(k_int);

x0 = zeros(dim);

# generate LMO
d = ones(dim);
a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
LMO = generate_linear_prog_function(d, a_2d, k);

# const num_iters_arr = Int[2e2, 4e2, 6e2, 8e2, 10e2];
# const num_iters_arr = Int[1e0, 2e0, 3e0, 4e0, 5e0];
# const num_iters_arr = Int[1:14;];
const num_iters_arr = Int[10:10:200;];
# const num_iters_arr = Int[20;];
# const num_iters_arr = Int[1:3;];
final_res = zeros(length(num_iters_arr), 5);

t_start = time();
for i = 1 : length(num_iters_arr)
    println("repeated: $(i), T: $(num_iters_arr[i]), time: $(Dates.hour(now())):$(Dates.minute(now()))");
    num_iters = num_iters_arr[i];
    alpha = 1/sqrt(num_iters);
    phi = 1/num_iters^(2/3);

    res_DeFW = DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, alpha);
    final_res[i, 2] = res_DeFW[4];
    final_res[i, 4] = res_DeFW[3];

    res_DeSAGAFW = DeSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters);
    final_res[i, 3] = res_DeSAGAFW[4];
    final_res[i, 5] = res_DeSAGAFW[3];

    # res_CenFW = CenFW(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, num_iters);
    # final_res[i, 2] = res_CenFW[3];

    final_res[i, 1] = num_iters;
end

# res_CenGreedy = CenGreedy(dim, data_mat, f_discrete_batch, k_int, f_extension_batch, num_agents, data_cell);
# res_CenFW = CenFW(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, num_iters);
#
# res_DeFW = DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, alpha);
#
# res_DeSFW = DeSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters, alpha, phi);
#
# res_DESAGAFW = DeSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters);
#
# res_DeSSAGAFW = DeSSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters);

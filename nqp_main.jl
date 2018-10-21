using LaTeXStrings

include("nqp.jl");
include("algorithms/CenFW.jl"); include("algorithms/DeFW.jl"); include("algorithms/DeGSFW.jl"); include("algorithms/AccDeGSFW.jl");
include("comm.jl");

# Step 1: initialization
const num_agents = 50;
# const num_iters = Int(20);
# alpha = 1/sqrt(num_iters);
# phi = 1/num_iters^(2/3);

# load data
# data_cell[i][j] is a dim-by-dim matrix H, i for agent, j for index in the batch
const data_cell, A, dim, u, b = load_nqp_partitioned_data(num_agents);
# the NQP problem is defined as f_i(x) = ( x/2 - u )^T H_i x, s.t. {x | 0<=x<=u, Ax<=b}, where A is the constraint_mat of size num_constraints-by-dim

# load weights matrix
# const weights, beta = load_network_50("complete");
const weights, beta = load_network_50("line");
# const weights, beta = load_network_50("er");
num_out_edges = count(i->(i>0), weights) - num_agents;

x0 = zeros(dim);
# generate LMO
LMO = generate_linear_prog_function(u, A, b);

# const num_iters_arr = Int[2e2, 4e2, 6e2, 8e2, 10e2];
# const num_iters_arr = Int[1e0, 2e0, 3e0, 4e0, 5e0];
# const num_iters_arr = Int[1:14;];
# const num_iters_arr = Int[10:10:200;];
const num_iters_arr = Int[1:20;];
final_res = zeros(length(num_iters_arr), 5);

for i = 1 : length(num_iters_arr)
    num_iters = num_iters_arr[i];
    alpha = 1/sqrt(num_iters);
    phi = 1/num_iters^(2/3);

    # res_DeFW = DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha);
    # final_res[i, 2] = res_DeFW[4];
    # final_res[i, 4] = res_DeFW[3];

    res_AccDESAGAFW = AccDeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, beta);
    final_res[i, 2] = res_AccDESAGAFW[4];
    final_res[i, 4] = res_AccDESAGAFW[3];

    res_DeGSFW = DeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters);
    final_res[i, 3] = res_DeGSFW[4];
    final_res[i, 5] = res_DeGSFW[3];

    # res_CenFW = CenFW(dim, data_cell, LMO, f_batch, gradient_batch, num_iters);
    # final_res[i, 2] = res_CenFW[end, 3];

    final_res[i, 1] = num_iters;
end

# res_CenFW = CenFW(dim, data_cell, LMO, f_batch, gradient_batch, num_iters);
#
# res_DeFW = DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha);
#
# res_DESAGAFW = DeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters);
# res_AccDESAGAFW = AccDeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, beta);



# res_CenSFW = CenSFW(dim, data_cell, LMO, f_batch, stochastic_gradient_batch, num_iters);
# #
# res_DeSFW = DeSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, num_iters, alpha, phi);
# #
# res_DeSGSFW = DeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, num_iters);
# res_AccDESSAGAFW = res_AccDeSGSFW = DeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, num_iters, beta);

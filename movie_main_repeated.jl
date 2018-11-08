using LaTeXStrings

include("facility.jl");
include("algorithms/CenFW.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGSFW.jl"); include("algorithms/CenGreedy.jl"); include("algorithms/AccDeGSFW.jl");
include("comm.jl");

function main()
# Step 1: initialization
const k_int = 10;  # the cardinality constraint
# const num_agents = 100;
const num_agents = 50;
# const num_iters = Int(1e1);
# alpha = 1/sqrt(num_iters);
# phi = 1/num_iters^(2/3);
repeated = 1;

# load data
# data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user
const data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents, "100K");

# load weights matrix
# const weights, beta = generate_network(num_agents, avg_degree);
const weights, beta = load_network_50("complete");
# const weights, beta = load_network_50("line");
# const weights, beta = load_network_50("er");
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
# const num_iters_arr = Int[1:1:10;];
const num_iters_arr = Int[10];
res = zeros(length(num_iters_arr), 7);

t_start = time();
for i = 1 : repeated
    final_res = zeros(length(num_iters_arr), 7);

    for i = 1 : length(num_iters_arr)
        # set the value of K (the degree of the chebyshev polynomial)
        if 1/(1-beta) <= ((e^2 + 1)/(e^2 - 1))^2
            K = 1;
        else
            K = ceil(sqrt((1 + beta)/(1 - beta))) + 1;
        end
        num_iters = num_iters_arr[i];
        non_acc_num_iters = num_iters * K;
        decg_num_iters = num_iters * K;
        # non_acc_num_iters = num_iters;
        # decg_num_iters = round(Int, num_iters*(num_iters+1)*(2*num_iters+1)/6);
        alpha = 1/sqrt(num_iters);
        phi = 1/num_iters^(2/3);

        # res_DeCG = DeCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, alpha);
        # final_res[i, 2] = res_DeCG[4];

        # res_DeGSFW = DeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters);
        # final_res[i, 3] = res_DeGSFW[4];

        # println("repeated: $(i), algorithm: DeSCG, T: $(decg_num_iters), time: $(Dates.hour(now())):$(Dates.minute(now())):$(Dates.second(now()))");
        # res_DeSCG = DeSCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, decg_num_iters, alpha, phi);
        # final_res[i, 2] = res_DeSCG[4];
        # final_res[i, 4] = res_DeSCG[3];

        println("repeated: $(i), algorithm: DeSGSFW, T: $(non_acc_num_iters), time: $(Dates.hour(now())):$(Dates.minute(now())):$(Dates.second(now()))");
        res_DeSGSFW = DeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, non_acc_num_iters);
        final_res[i, 3] = res_DeSGSFW[4];
        final_res[i, 5] = res_DeSGSFW[3];

        # println("repeated: $(i), algorithm: AccDeSGSFW, T: $(num_iters), time: $(Dates.hour(now())):$(Dates.minute(now())):$(Dates.second(now()))");
        # res_AccDeSGSFW = AccDeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters, beta, K);
        # final_res[i, 6] = res_AccDeSGSFW[4];
        # final_res[i, 7] = res_AccDeSGSFW[3];

        #
        # res_CenSFW = CenSFW(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters);
        # final_res[i, 2] = res_CenSFW[3];

        final_res[i, 1] = num_iters;
    end
    res = res + final_res;
end
final_res = res ./ repeated;

# res_CenGreedy = CenGreedy(dim, data_mat, f_discrete_batch, k_int, f_extension_batch, num_agents, data_cell);
# res_CenFW = CenFW(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, num_iters);
#
# res_DeCG = DeCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, alpha);
#
# res_DeSCG = DeSCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters, alpha, phi);
#
# res_DESAGAFW = DeGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters);
#
# res_DeSGSFW = DeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters);
return final_res;
end

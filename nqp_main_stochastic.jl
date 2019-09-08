using Dates, MAT

include("models/nqp.jl");
include("algorithms/CenCG.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGTFW.jl"); include("algorithms/AccDeGTFW.jl");
include("comm.jl");

function nqp_main_stochastic(min_num_iters::Int, interval_num_iters::Int, max_num_iters::Int, num_trials::Int, graph_style::String, num_agents::Int, FIX_COMM::Bool)
# the number of iterations are [min_num_iters : interval_num_iters : max_num_iters]
# num_trials: the number of trials/repetitions
# graph_style: can be "complete" for complete graph, or "er" for Erdos-Renyi random graph, or "line" for line graph
# num_agents: number of computing agents in the network
# FIX_COMM: all algorithms have the same #communication if FIX_COMM==true, otherwise all algorithms have the same #gradient evaluation
# return value: (res_DeSCG, res_DeSGTFW, res_AccDeSGTFW, res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of [min_num_iters : interval_num_iters : max_num_iters], and each row of res_XXX contains [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function]

    # Step 1: initialization
    num_agents = 50;

    # load data
    # data_cell[i][j] is a dim-by-dim matrix H, i for agent, j for index in the batch
    data_cell, A, dim, u, b = load_nqp_partitioned_data(num_agents);
    # the NQP problem is defined as f_i(x) = ( x/2 - u )^T H_i x, s.t. {x | 0<=x<=u, Ax<=b}, where A is the constraint_mat of size num_constraints-by-dim

    # load weights matrix
    available_graph_style = ["complete", "line", "er"];
    if ~(graph_style in available_graph_style)
        error("graph_style should be \"complete\", \"line\", or \"er\"");
    end
    weights, beta = load_network(graph_style, num_agents);
    num_out_edges = count(i->(i>0), weights) - num_agents;

    x0 = zeros(dim);
    # generate LMO
    LMO = generate_linear_prog_function(u, A, b);

    num_iters_arr = min_num_iters:interval_num_iters:max_num_iters;
    res_DeSCG= zeros(length(num_iters_arr), 5);
    res_DeSGTFW = zeros(length(num_iters_arr), 5);
    res_AccDeSGTFW = zeros(length(num_iters_arr), 5);
    res_CenSCG = zeros(length(num_iters_arr), 5);

    # Step 2: test algorithms for multiple times and return averaged results
    t_start = time();
    for j = 1 : num_trials
        println("trial: $(j)");
        for i = 1 : length(num_iters_arr)
            # set the value of K (the degree of the chebyshev polynomial)
            if 1/(1-beta) <= ((e^2 + 1)/(e^2 - 1))^2
                K = 1;
            else
                K = round(Int, ceil(sqrt((1 + beta)/(1 - beta))) + 1);
            end
            num_iters = num_iters_arr[i];
            if FIX_COMM
                non_acc_num_iters = num_iters * K;
                decg_num_iters = num_iters * K;
            else
                non_acc_num_iters = num_iters;
                decg_num_iters = round(Int, num_iters*(num_iters+1)*(2*num_iters+1)/6);
            end
            alpha = 1/sqrt(num_iters);
            phi = 1/num_iters^(2/3);

            println("DeSCG, T: $(decg_num_iters), time: $(Dates.Time(now()))");
            res_DeSCG[i, :] = res_DeSCG[i, :] + DeSCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, decg_num_iters, alpha, phi);

            println("DeSGTFW, T: $(non_acc_num_iters), time:$(Dates.Time(now()))");
            res_DeSGTFW[i, :] = res_DeSGTFW[i, :] + DeSGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, non_acc_num_iters);

            println("AccDeSGTFW, T: $(num_iters), time: $(Dates.hour(now())):$(Dates.minute(now())):$(Dates.second(now()))");
            res_AccDeSGTFW[i, :] = res_AccDeSGTFW[i, :] + AccDeSGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, stochastic_gradient_batch, num_iters, beta, K);
        end
    end
    res_DeSCG = res_DeSCG ./ num_trials;
    res_DeSGTFW = res_DeSGTFW ./ num_trials;
    res_AccDeSGTFW = res_AccDeSGTFW ./ num_trials;
    res_CenSCG = res_CenSCG ./ num_trials;

    return res_DeSCG, res_DeSGTFW, res_AccDeSGTFW, res_CenSCG;
end

using Dates, MAT

include("facility.jl");
include("algorithms/CenCG.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGSFW.jl"); include("algorithms/CenGreedy.jl"); include("algorithms/AccDeGSFW.jl");
include("comm.jl");


function movie_main_stochastic(min_num_iters::Int, interval_num_iters::Int, max_num_iters::Int, num_trials::Int, graph_style::String, num_agents::Int, cardinality::Int, FIX_COMM::Bool)
# the number of iterations are [min_num_iters : interval_num_iters : max_num_iters]
# num_trials: the number of trials/repetitions
# graph_style: can be "complete" for complete graph, or "er" for Erdos-Renyi random graph, or "line" for line graph
# num_agents: number of computing agents in the network
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMM: all algorithms have the same #communication if FIX_COMM==true, otherwise all algorithms have the same #gradient evaluation
# return value: (res_DeSCG, res_DeSGSFW, res_AccDeSGSFW, res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of [min_num_iters : interval_num_iters : max_num_iters], and each row of res_XXX contains [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function]

    # Step 1: initialization
    # load data
    # data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user, data_mat is a sparse matrix containing the same data set
    data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents, "100K");  # the second argument can be "100K" or "1M"

    # load weights matrix
    available_graph_style = ["complete", "line", "er"];
    if ~(graph_style in available_graph_style)
        error("graph_style should be \"complete\", \"line\", or \"er\"");
    end
    weights, beta = load_network(graph_style, num_agents);
    num_out_edges = count(i->(i>0), weights) - num_agents;

    dim = num_movies;

    x0 = zeros(dim);

    # generate LMO
    d = ones(dim);
    a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
    LMO = generate_linear_prog_function(d, a_2d, cardinality*1.0);

    num_iters_arr = min_num_iters:interval_num_iters:max_num_iters;
    res_DeSCG= zeros(length(num_iters_arr), 5);
    res_DeSGSFW = zeros(length(num_iters_arr), 5);
    res_AccDeSGSFW = zeros(length(num_iters_arr), 5);
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
            res_DeSCG[i, :] = res_DeSCG[i, :] + DeSCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, decg_num_iters, alpha, phi);

            println("DeSGSFW, T: $(non_acc_num_iters), time:$(Dates.Time(now()))");
            res_DeSGSFW[i, :] = res_DeSGSFW[i, :] + DeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, non_acc_num_iters);

            println("AccDeSGSFW, T: $(num_iters), time: $(Dates.hour(now())):$(Dates.minute(now())):$(Dates.second(now()))");
            res_AccDeSGSFW[i, :] = res_AccDeSGSFW[i, :] + AccDeSGSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iters, beta, K);

            println("CenSCG, T: $(decg_num_iters), time: $(Dates.Time(now()))");
            res_CenSCG[i, :] = res_CenSCG[i, :] + CenSCG(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_batch, decg_num_iters);

            matwrite("data/movie_main_stochastic_auto_save.mat", Dict("res_DeSCG" => res_DeSCG ./ j, "res_DeSGSFW" => res_DeSGSFW ./ j, "res_AccDeSGSFW" => res_AccDeSGSFW ./ j, "res_CenSCG" => res_CenSCG ./ j));
        end
    end
    res_DeSCG = res_DeSCG ./ num_trials;
    res_DeSGSFW = res_DeSGSFW ./ num_trials;
    res_AccDeSGSFW = res_AccDeSGSFW ./ num_trials;
    res_CenSCG = res_CenSCG ./ num_trials;

    return res_DeSCG, res_DeSGSFW, res_AccDeSGSFW, res_CenSCG;
end

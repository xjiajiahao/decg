using Dates, MAT

include("models/facility_location.jl");
include("algorithms/CenCG.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGTFW.jl"); include("algorithms/CenGreedy.jl"); include("algorithms/AccDeGTFW.jl");
include("comm.jl");

function movie_main_det(min_num_iters::Int, interval_num_iters::Int, max_num_iters::Int, graph_style::String, num_agents::Int, cardinality::Int, FIX_COMM::Bool)
# the number of iterations are [min_num_iters : interval_num_iters : max_num_iters]
# graph_style: can be "complete" for complete graph, or "er" for Erdos-Renyi random graph, or "line" for line graph
# num_agents: number of computing agents in the network
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMM: all algorithms have the same #communication if FIX_COMM==true, otherwise all algorithms have the same #gradient evaluation
# return value: (res_DeSCG, res_DeSGTFW, res_AccDeSGTFW, res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of [min_num_iters : interval_num_iters : max_num_iters], and each row of res_XXX contains [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function]

    # Step 1: initialization
    # load data
    # data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user
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
    res_DeCG= zeros(length(num_iters_arr), 5);
    res_DeGTFW = zeros(length(num_iters_arr), 5);
    res_AccDeGTFW = zeros(length(num_iters_arr), 5);
    res_CenCG = zeros(length(num_iters_arr), 5);

    # Step 2: test algorithms for multiple times and return averaged results
    t_start = time();
    for i = 1 : length(num_iters_arr)
        # set the value of K (the degree of the chebyshev polynomial)
        if 1/(1-beta) <= ((e^2 + 1)/(e^2 - 1))^2
            K = 1;
        else
            K = ceil(sqrt((1 + beta)/(1 - beta))) + 1;
        end
        num_iters = num_iters_arr[i];
        if FIX_COMM
            non_acc_num_iters = num_iters * K;
        else
            non_acc_num_iters = num_iters;
        end
        alpha = 1/sqrt(num_iters);
        phi = 1/num_iters^(2/3);

        println("DeCG, T: $(non_acc_num_iters), time:$(Dates.Time(now()))");
        res_DeCG[i, :] = DeCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, non_acc_num_iters, alpha);

        println("DeGTFW, T: $(non_acc_num_iters), time: $(Dates.Time(now()))");
        res_DeGTFW[i, :] = DeGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, non_acc_num_iters);

        println("AccDeGTFW, T: $(num_iters), time:$(Dates.Time(now()))");
        res_AccDeGTFW[i, :] = AccDeGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_extension_batch, gradient_extension_batch, num_iters, beta, K);

        println("CenCG, T: $(non_acc_num_iters), time: $(Dates.Time(now()))");
        res_CenCG[i, :] = CenCG(dim, data_cell, LMO, f_extension_batch, gradient_extension_batch, non_acc_num_iters);
    end

    return res_DeCG, res_DeGTFW, res_AccDeGTFW, res_CenCG;
end

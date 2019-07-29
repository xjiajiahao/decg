using Dates, MAT

include("models/facility_location.jl");
include("algorithms/CenCG.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGSFW.jl"); include("algorithms/CenGreedy.jl"); include("algorithms/AccDeGSFW.jl"); include("algorithms/CenPGD.jl"); include("algorithms/CenSTORM.jl");
include("comm.jl");


function movie_main_stochastic_cen(num_iters::Int, print_freq::Int, num_trials::Int, cardinality::Int, FIX_COMP::Bool)
# num_iters: the number of iterations
# print_freq: the frequency to evaluate the loss value
# num_trials: the number of trials/repetitions
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMP: all algorithms have the same #gradient evaluations, otherwise the same #iterations
# return value: (res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of div(num_iters / print_freq) + 1, and each row of res_XXX contains [#iterations, elapsed time, #eavluations of simple functions, #doubles transferred in the network, averaged objective function]

    # Step 1: initialization
    # load data
    # data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user, data_mat is a sparse matrix containing the same data set
    num_agents = 1;
    data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents, "1M");  # the second argument can be "100K" or "1M"

    # # PSGD parameters (100K)
    # eta_coef_PSGD = 1e-2;
    # eta_exp_PSGD = 1/2;
    #
    # # SCG parameters (100K)
    # rho_coef_SCG = 1.0;
    # rho_exp_SCG = 2/3;

    mini_batch_size = 512;
    sample_times = 20;

    # PSGD parameters (1M)
    eta_coef_PSGD = 1e-4;
    eta_exp_PSGD = 1/2;

    # SCG parameters (1M)
    rho_coef_SCG = 1.0;
    rho_exp_SCG = 2/3;

    # STORM parameters (1M)
    # rho_coef_STORM = 7.5e-1;
    rho_coef_STORM = 8e-1;
    rho_exp_STORM = 1.0;
    interpolate_times_STORM = 1;

    # load weights matrix
    dim = num_movies;

    x0 = zeros(dim);

    # generate LMO
    d = ones(dim);
    a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
    LMO = generate_linear_prog_function(d, a_2d, cardinality*1.0);
    # generate PO
    PO = generate_projection_function(d, a_2d, cardinality*1.0);

    #
    res_CenSCG = zeros(1, 5);
    res_CenPSGD = zeros(1, 5);
    res_CenSTORM = zeros(1, 5);

    # Step 2: test algorithms for multiple times and return averaged results
    t_start = time();
    for j = 1 : num_trials
        println("trial: $(j)");
        num_iters_base = num_iters;
        if FIX_COMP
            num_iters_SCG = num_iters_base * (cardinality + 1);
            print_freq_SCG = print_freq * (cardinality + 1);

            num_iters_PSGD = num_iters_base * (cardinality + 1);
            print_freq_PSGD = print_freq * (cardinality + 1);

            num_iters_STORM = num_iters_base;
            print_freq_STORM = print_freq;
        else
            num_iters_SCG = num_iters_base;
            print_freq_SCG = print_freq;

            num_iters_PSGD = num_iters_base;
            print_freq_PSGD = print_freq;

            num_iters_STORM = num_iters_base;
            print_freq_STORM = print_freq;
        end

        println("CenSCG, T: $(num_iters_SCG), time: $(Dates.Time(now()))");
        res_CenSCG = CenSCG(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size, num_iters_SCG, rho_coef_SCG, rho_exp_SCG, print_freq_SCG, sample_times);

        # println("CenPSGD, T: $(num_iters_PSGD), time: $(Dates.Time(now()))");
        # res_CenPSGD = CenPSGD(dim, data_cell, PO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size, num_iters_PSGD, eta_coef_PSGD, eta_exp_PSGD, print_freq_SCG, sample_times);

        println("CenSTORM, T: $(num_iters_STORM), time: $(Dates.Time(now()))");
        res_CenSTORM = CenSTORM(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size, num_iters_STORM, rho_coef_STORM, rho_exp_STORM, cardinality, print_freq_STORM, interpolate_times_STORM, sample_times);

        matwrite("data/movie_main_stochastic_auto_save.mat", Dict("res_CenSCG" => res_CenSCG ./ j, "res_CenPSGD" => res_CenPSGD ./ j, "res_CenSTORM" => res_CenSTORM ./ j));
    end
    res_CenSCG = res_CenSCG ./ num_trials; res_CenSCG[:, 5] = res_CenSCG[:, 5] / num_users;
    res_CenPSGD = res_CenPSGD ./ num_trials;; res_CenPSGD[:, 5] = res_CenPSGD[:, 5] / num_users;
    res_CenSTORM = res_CenSTORM ./ num_trials; res_CenSTORM[:, 5] = res_CenSTORM[:, 5] / num_users;

    return res_CenSCG, res_CenPSGD, res_CenSTORM;
end

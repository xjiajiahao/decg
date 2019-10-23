using Dates, MAT

<<<<<<< HEAD
include("models/concave_over_modular.jl");
include("algorithms/CenCG.jl");
include("algorithms/DeCG.jl");
include("algorithms/DeGTFW.jl");
include("algorithms/AccDeGTFW.jl");
include("algorithms/CenPGD.jl");
include("algorithms/CenSTORM.jl");
include("algorithms/CenSCGPP.jl");
include("algorithms/CenSFW.jl");
include("utils/comm.jl");


function movie_main_cen_concave(min_num_iters::Int, interval_num_iters::Int, max_num_iters::Int, num_trials::Int, cardinality::Int, FIX_COMP::Bool)
# the number of iterations are [min_num_iters : interval_num_iters : max_num_iters]
# num_trials: the number of trials/repetitions
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMP: all algorithms have the same #gradient evaluations, otherwise the same #iterations
# return value: (res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of [min_num_iters : interval_num_iters : max_num_iters], and each row of res_XXX contains [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function]
=======
include("models/concave_modular.jl");
include("algorithms/CenCG.jl"); include("algorithms/DeCG.jl"); include("algorithms/DeGSFW.jl"); include("algorithms/CenGreedy.jl"); include("algorithms/AccDeGSFW.jl"); include("algorithms/CenPGD.jl"); include("algorithms/CenSTORM.jl"); # include("algorithms/CenSCGPP.jl");
include("comm.jl");


function movie_main_cen_concave(num_iters::Int, print_freq::Int, num_trials::Int, cardinality::Int, FIX_COMP::Bool)
# num_iters: the number of iterations
# print_freq: the frequency to evaluate the loss value
# num_trials: the number of trials/repetitions
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMP: all algorithms have the same #gradient evaluations, otherwise the same #iterations
# return value: (res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of div(num_iters / print_freq) + 1, and each row of res_XXX contains [#iterations, elapsed time, #eavluations of simple functions, #doubles transferred in the network, averaged objective function]
>>>>>>> trajectory

    # Step 1: initialization
    # load data
    # data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user, data_mat is a sparse matrix containing the same data set
    num_agents = 1;
    data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents, "1M");  # the second argument can be "100K" or "1M"
    # data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data(num_agents, "100K");  # the second argument can be "100K" or "1M"

    # # PSGD parameters (100K)
    # eta_coef_PSGD = 1e-2;
    # eta_exp_PSGD = 1/2;
    #
    # # SCG parameters (100K)
    # rho_coef_SCG = 1.0;
    # rho_exp_SCG = 2/3;

    # # STORM parameters (100K)
    # # rho_coef_STORM = 7.5e-1;
    # # rho_coef_STORM = 2e0;
    # rho_coef_STORM = 2e0;
    # rho_exp_STORM = 1.0;
    # interpolate_times_STORM = 1;
    # mini_batch_size_STORM = 10;

    # # SCGPP parameters (100K)
    # mini_batch_size_SCGPP = 10;
    # initial_sample_times_SCGPP = 100;
    # interpolate_times_SCGPP = 10;

    # mini_batch_size = 128;
<<<<<<< HEAD
    mini_batch_size_base = 40;
=======
    mini_batch_size_base = 200;
>>>>>>> trajectory
    sample_times = 1;
    # mini_batch_size = 64;
    # sample_times = 20;

    # PSGD parameters (1M)
    eta_coef_PSGD = 1e-4;
    eta_exp_PSGD = 1/2;

    # SCG parameters (1M)
<<<<<<< HEAD
    # rho_coef_SCG = 2.0;  # for k = 5, concave_over_modular
    # rho_coef_SCG = 0.25;  # for k = 5, concave_over_modular
    rho_coef_SCG = 0.5;  # for k = 10, concave_over_modular
=======
    # rho_coef_SCG = 1.0;  # for k = 5, concave_modular
    rho_coef_SCG = 0.5;  # for k = 10, concave_modular
>>>>>>> trajectory
    rho_exp_SCG = 2/3;

    # STORM parameters (1M)
    # rho_coef_STORM = 7.5e-1;
    # rho_coef_STORM = 2e0;
<<<<<<< HEAD
    rho_coef_STORM = 5e-1;
    rho_exp_STORM = 2/3;
    interpolate_times_STORM = 1;
    sample_times = 1;
    # mini_batch_size_STORM = 20;
    mini_batch_size_STORM = 40;
=======
    rho_coef_STORM = 2e0;
    rho_exp_STORM = 1.0;
    interpolate_times_STORM = 1;
    sample_times = 1;
    mini_batch_size_STORM = 100;
>>>>>>> trajectory

    # SCGPP parameters (1M)
    mini_batch_size_SCGPP = 10;
    initial_sample_times_SCGPP = 10000;
    interpolate_times_SCGPP = 100;

<<<<<<< HEAD
    # SFW paramters
    is_batch_size_increasing_SFW = true;
    mini_batch_size_SFW = 40;

=======
>>>>>>> trajectory
    # load weights matrix
    dim = num_movies;

    x0 = zeros(dim);

    # generate LMO
    d = ones(dim);
    a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
<<<<<<< HEAD
    # LMO = generate_linear_prog_function(d, a_2d, cardinality*1.0);
    LMO = generate_linear_prog_function(d, cardinality);
    # generate PO
    PO = generate_projection_function(d, a_2d, cardinality*1.0);

    num_iters_arr = min_num_iters:interval_num_iters:max_num_iters;
    res_CenSCG = zeros(length(num_iters_arr), 5);
    res_CenPSGD = zeros(length(num_iters_arr), 5);
    res_CenSTORM = zeros(length(num_iters_arr), 5);
    res_CenSCGPP = zeros(length(num_iters_arr), 5);
    res_CenSFW = zeros(length(num_iters_arr), 5);
=======
    LMO = generate_linear_prog_function(d, a_2d, cardinality*1.0);
    # generate PO
    PO = generate_projection_function(d, a_2d, cardinality*1.0);

    # initialization
    num_iters_base = num_iters;
    mini_batch_size = mini_batch_size_base;
    if FIX_COMP
        num_iters_SCG = Int(ceil(num_iters_base * (cardinality * 2 + 1) * (mini_batch_size_STORM / mini_batch_size)));
        print_freq_SCG = Int(ceil(print_freq * (cardinality * 2 + 1) * (mini_batch_size_STORM / mini_batch_size)));

        num_iters_PSGD = Int(ceil(num_iters_base * (cardinality * 2 + 1) * (mini_batch_size_STORM / mini_batch_size)));
        print_freq_PSGD = Int(ceil(print_freq * (cardinality * 2 + 1) * (mini_batch_size_STORM / mini_batch_size)));

        num_iters_STORM = num_iters_base;
        print_freq_STORM = print_freq;

        num_iters_SCGPP = num_iters_base;
        print_freq_SCGPP = print_freq;
    else
        num_iters_SCG = num_iters_base;
        print_freq_SCG = print_freq;

        num_iters_PSGD = num_iters_base;
        print_freq_PSGD = print_freq;

        num_iters_STORM = num_iters_base;
        print_freq_STORM = print_freq;

        num_iters_SCGPP = num_iters_base;
        print_freq_SCGPP = print_freq;
    end
    res_CenSCG = zeros(div(num_iters_SCG, print_freq_SCG) + 1, 6);
    res_CenPSGD = zeros(div(num_iters_PSGD, print_freq_PSGD) + 1, 5);
    res_CenSTORM = zeros(div(num_iters_STORM, print_freq_STORM) + 1, 6);
    res_CenSCGPP = zeros(div(num_iters_SCGPP, print_freq_SCGPP) + 1, 5);
>>>>>>> trajectory

    # Step 2: test algorithms for multiple times and return averaged results
    t_start = time();
    for j = 1 : num_trials
        println("trial: $(j)");
<<<<<<< HEAD
        for i = 1 : length(num_iters_arr)
            num_iters_base = num_iters_arr[i];
            if FIX_COMP
                num_iters_SCG = Int(ceil(num_iters_base * (1 * 2 * interpolate_times_STORM + 1) * (mini_batch_size_STORM  * 1.0 / mini_batch_size_base)));
                num_iters_PSGD = Int(ceil(num_iters_base * (1 * 2 * interpolate_times_STORM + 1) * (mini_batch_size_STORM * 1.0 / mini_batch_size_base)));
                num_iters_STORM = num_iters_base;

                num_iters_SCGPP = Int(ceil((num_iters_base * (1 * 2 * interpolate_times_STORM + 1) * mini_batch_size_STORM * sample_times - mini_batch_size_SCGPP * initial_sample_times_SCGPP) / ((1 * interpolate_times_SCGPP + 1) * mini_batch_size_SCGPP * sample_times) + 1));

                if is_batch_size_increasing_SFW
                    num_iters_SFW = Int(ceil((3 * (num_iters_base * (1 * 2 * interpolate_times_STORM + 1) * mini_batch_size_STORM / mini_batch_size_SFW))^(1/3)));
                else
                    num_iters_SFW = Int(ceil(num_iters_base * (1 * 2 * interpolate_times_STORM + 1) * mini_batch_size_STORM / mini_batch_size_SFW));
                end
            else
                num_iters_SCG = num_iters_base;
                num_iters_PSGD = num_iters_base;
                num_iters_STORM = num_iters_base;
                num_iters_SCGPP = num_iters_base;
                num_iters_SFW = num_iters_base;
            end

            println("CenSCG, T: $(num_iters_SCG), time: $(Dates.Time(now()))");
            tmp_res = CenSCG(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size_base, num_iters_SCG, rho_coef_SCG, rho_exp_SCG, sample_times);
            res_CenSCG[i, :] = res_CenSCG[i, :] + tmp_res;
            tmp_res[5] = tmp_res[5] / num_users;
            println("$(tmp_res)");

            # println("CenPSGD, T: $(num_iters_PSGD), time: $(Dates.Time(now()))");
            # tmp_res = CenPSGD(dim, data_cell, PO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size_base, num_iters_PSGD, eta_coef_PSGD, eta_exp_PSGD, sample_times);
            # res_CenPSGD[i, :] = res_CenPSGD[i, :] + tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            # println("CenSTORM, T: $(num_iters_STORM), time: $(Dates.Time(now()))");
            # tmp_res = CenSTORM(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_STORM, num_iters_STORM, rho_coef_STORM, rho_exp_STORM, cardinality, interpolate_times_STORM, sample_times);
            # res_CenSTORM[i, :] = res_CenSTORM[i, :] +  tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            # println("CenSCGPP, T: $(num_iters_SCGPP), time: $(Dates.Time(now()))");
            # tmp_res = CenSCGPP(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_SCGPP, initial_sample_times_SCGPP, num_iters_SCGPP, interpolate_times_SCGPP, sample_times);
            # res_CenSCGPP[i, :] = res_CenSCGPP[i, :] + tmp_res;

            # println("CenSFW, T: $(num_iters_SFW), time: $(Dates.Time(now()))");
            # tmp_res = CenSFW(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_SFW, num_iters_SFW, is_batch_size_increasing_SFW, cardinality, sample_times);
            # res_CenSFW[i, :] = res_CenSFW[i, :] +  tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            println("\n");

            matwrite("data/result_movie_main_cen_concave.mat", Dict("res_CenSCG" => res_CenSCG ./ j, "res_CenPSGD" => res_CenPSGD ./ j, "res_CenSTORM" => res_CenSTORM ./ j, "res_CenSCGPP" => res_CenSCGPP ./ j, "res_CenSFW" => res_CenSFW ./ j));
        end
    end
    res_CenSCG = res_CenSCG ./ num_trials; res_CenSCG[:, 5] = res_CenSCG[:, 5] / num_users;
    res_CenPSGD = res_CenPSGD ./ num_trials;; res_CenPSGD[:, 5] = res_CenPSGD[:, 5] / num_users;
    res_CenSTORM = res_CenSTORM ./ num_trials; res_CenSTORM[:, 5] = res_CenSTORM[:, 5] / num_users;
    res_CenSCGPP = res_CenSCGPP ./ num_trials; res_CenSCGPP[:, 5] = res_CenSCGPP[:, 5] / num_users;
    res_CenSFW = res_CenSFW ./ num_trials; res_CenSFW[:, 5] = res_CenSFW[:, 5] / num_users;


    return res_CenSCG, res_CenPSGD, res_CenSTORM, res_CenSCGPP, res_CenSFW;
=======

        # println("CenSCG, T: $(num_iters_SCG), time: $(Dates.Time(now()))");
        # tmp_res = CenSCG(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size_base, num_iters_SCG, rho_coef_SCG, rho_exp_SCG, sample_times);
        # res_CenSCG += tmp_res;

        println("CenPSGD, T: $(num_iters_PSGD), time: $(Dates.Time(now()))");
        tmp_res = CenPSGD(dim, data_cell, PO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size_base, num_iters_PSGD, eta_coef_PSGD, eta_exp_PSGD, print_freq_PSGD, sample_times);
        res_CenPSGD += tmp_res;

        # println("CenSTORM, T: $(num_iters_STORM), time: $(Dates.Time(now()))");
        # tmp_res = CenSTORM(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_STORM, num_iters_STORM, rho_coef_STORM, rho_exp_STORM, cardinality, interpolate_times_STORM, sample_times);
        # res_CenSTORM += tmp_res;

        # println("CenSCGPP, T: $(num_iters_SCGPP), time: $(Dates.Time(now()))");
        # tmp_res = CenSCGPP(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_SCGPP, initial_sample_times_SCGPP, num_iters_SCGPP, interpolate_times_SCGPP, sample_times);
        # res_CenSCGPP += tmp_res;

        matwrite("data/movie_main_concave_auto_save.mat", Dict("res_CenSCG" => res_CenSCG ./ j, "res_CenPSGD" => res_CenPSGD ./ j, "res_CenSTORM" => res_CenSTORM ./ j, "res_CenSCGPP" => res_CenSCGPP ./ j));
    end

    res_CenSCG = res_CenSCG ./ num_trials;
    res_CenSCG[:, 5] = res_CenSCG[:, 5] / num_users;
    res_CenSCG[:, 6] = res_CenSCG[:, 6] / num_users;

    res_CenPSGD = res_CenPSGD ./ num_trials;
    res_CenPSGD[:, 5] = res_CenPSGD[:, 5] / num_users;

    res_CenSTORM = res_CenSTORM ./ num_trials;
    res_CenSTORM[:, 5] = res_CenSTORM[:, 5] / num_users;
    res_CenSTORM[:, 6] = res_CenSTORM[:, 6] / num_users;

    res_CenSCGPP = res_CenSCGPP ./ num_trials;
    res_CenSCGPP[:, 5] = res_CenSCGPP[:, 5] / num_users;


    return res_CenSCG, res_CenPSGD, res_CenSTORM, res_CenSCGPP;
>>>>>>> trajectory
end

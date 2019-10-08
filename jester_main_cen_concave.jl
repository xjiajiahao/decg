using Dates, MAT

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


function jester_main_cen_concave(min_num_iters::Int, interval_num_iters::Int, max_num_iters::Int, num_trials::Int, cardinality::Int, FIX_COMP::Bool)
# the number of iterations are [min_num_iters : interval_num_iters : max_num_iters]
# num_trials: the number of trials/repetitions
# cardinality: the cardinality constraint parameter of the movie recommendation application
# FIX_COMP: all algorithms have the same #gradient evaluations, otherwise the same #iterations
# return value: (res_CenSCG), each res_XXX is a x-by-5 matrix, where x is the length of [min_num_iters : interval_num_iters : max_num_iters], and each row of res_XXX contains [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function]

    # Step 1: initialization
    # load data
    # data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user, data_mat is a sparse matrix containing the same data set
    num_agents = 1;
    data_cell, data_mat, num_movies, num_users = load_jester_partitioned_data(num_agents, 1);  # the second argument can be 1 or 2, which indicates the version of the Jester dataset

    mini_batch_size_base = 160;
    sample_times = 1;

    # PSGD parameters
    eta_coef_PSGD = 1e-4;
    eta_exp_PSGD = 1/2;

    # SCG parameters
    # rho_coef_SCG = 1.0;
    # rho_exp_SCG = 2/3;
    rho_coef_SCG = 0.5;
    # rho_coef_SCG = 0.25;
    # rho_coef_SCG = 1.0;
    # rho_exp_SCG = 0.5;
    rho_exp_SCG = 2/3;
    # rho_exp_SCG = 1.0;

    # STORM parameters
    rho_coef_STORM = 0.25;
    rho_exp_STORM = 2/3;
    # rho_coef_STORM = 0.3;
    # rho_exp_STORM = 0.5;
    interpolate_times_STORM = 1;
    mini_batch_size_STORM = 160;

    # SCGPP parameters
    mini_batch_size_SCGPP = 50;
    initial_sample_times_SCGPP = 10000;
    interpolate_times_SCGPP = 100;

    # SFW paramters
    is_batch_size_increasing_SFW = true;
    mini_batch_size_SFW = 160;

    # load weights matrix
    dim = num_movies;

    x0 = zeros(dim);

    # generate LMO
    d = ones(dim);
    a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
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

    # Step 2: test algorithms for multiple times and return averaged results
    t_start = time();
    for j = 1 : num_trials
        println("trial: $(j)");
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
            # tmp_res = res_CenPSGD[i, :] + CenPSGD(dim, data_cell, PO, f_extension_batch, stochastic_gradient_extension_mini_batch, mini_batch_size_base, num_iters_PSGD, eta_coef_PSGD, eta_exp_PSGD, sample_times);
            # res_CenPSGD[i, :] = res_CenPSGD[i, :] + tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            println("CenSTORM, T: $(num_iters_STORM), time: $(Dates.Time(now()))");
            tmp_res = CenSTORM(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_STORM, num_iters_STORM, rho_coef_STORM, rho_exp_STORM, cardinality, interpolate_times_STORM, sample_times);
            res_CenSTORM[i, :] = res_CenSTORM[i, :] + tmp_res;
            tmp_res[5] = tmp_res[5] / num_users;
            println("$(tmp_res)");

            # println("CenSCGPP, T: $(num_iters_SCGPP), time: $(Dates.Time(now()))");
            # tmp_res = res_CenSCGPP[i, :] + CenSCGPP(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_SCGPP, initial_sample_times_SCGPP, num_iters_SCGPP, interpolate_times_SCGPP, sample_times);
            # res_CenSCGPP[i, :] = res_CenSCGPP[i, :] + tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            # println("CenSFW, T: $(num_iters_SFW), time: $(Dates.Time(now()))");
            # tmp_res = CenSFW(dim, data_cell, LMO, f_extension_batch, stochastic_gradient_extension_mini_batch, stochastic_gradient_diff_extension_mini_batch, mini_batch_size_SFW, num_iters_SFW, is_batch_size_increasing_SFW, cardinality, sample_times);
            # res_CenSFW[i, :] = res_CenSFW[i, :] +  tmp_res;
            # tmp_res[5] = tmp_res[5] / num_users;
            # println("$(tmp_res)");

            println("\n");

            matwrite("data/result_jester_main_cen_concave.mat", Dict("res_CenSCG" => res_CenSCG ./ j, "res_CenPSGD" => res_CenPSGD ./ j, "res_CenSTORM" => res_CenSTORM ./ j, "res_CenSCGPP" => res_CenSCGPP ./ j, "res_CenSFW" => res_CenSFW ./ j));
        end
    end
    res_CenSCG = res_CenSCG ./ num_trials; res_CenSCG[:, 5] = res_CenSCG[:, 5] / num_users;
    res_CenPSGD = res_CenPSGD ./ num_trials;; res_CenPSGD[:, 5] = res_CenPSGD[:, 5] / num_users;
    res_CenSTORM = res_CenSTORM ./ num_trials; res_CenSTORM[:, 5] = res_CenSTORM[:, 5] / num_users;
    res_CenSCGPP = res_CenSCGPP ./ num_trials; res_CenSCGPP[:, 5] = res_CenSCGPP[:, 5] / num_users;
    res_CenSFW = res_CenSFW ./ num_trials; res_CenSFW[:, 5] = res_CenSFW[:, 5] / num_users;

    return res_CenSCG, res_CenPSGD, res_CenSTORM, res_CenSCGPP, res_CenSFW;
end

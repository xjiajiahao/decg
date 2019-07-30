# centralized STORM-FW, here num_iters denotes #iteration
function CenSTORM(dim, data_cell, LMO, f_batch, gradient_mini_batch, gradient_diff_mini_batch, mini_batch_size, num_iters, rho_coef, rho_exp, cardinality, full_gradient_batch, print_freq = 100, interpolate_times = 1, sample_times = 1)
    num_agents = size(data_cell, 2);  # num_agents should be 1
    function gradient(x, mini_batch_indices_arr, sample_times) # compute the sum of local gradients
        grad_x_ret = @sync @distributed (+) for i in 1:num_agents
            gradient_mini_batch(x, data_cell[i], mini_batch_indices_arr[i], sample_times)
        end
        return grad_x_ret;
    end

    function f_sum(x) # compute the global objective at x
        f_x_ret = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x_ret;
    end

    function generate_mini_batches()
        mini_batch_indices_arr_ret = [[] for i=1:num_agents];
        for i in 1:num_agents
            num_users = length(data_cell[i]);
            mini_batch_indices_arr_ret[i] = rand(1:num_users, mini_batch_size);
        end
        return mini_batch_indices_arr_ret;
    end

    function gradient_diff(x, y, mini_batch_indices_arr, interpolate_times, sample_times) # compute the sum of local gradients
        gradient_diff_ret = @sync @distributed (+) for i in 1:num_agents
            gradient_diff_mini_batch(x, y, data_cell[i], mini_batch_indices_arr[i], interpolate_times, sample_times)
        end
        return gradient_diff_ret;
    end

    function full_gradient_sum(x) # compute the sum of local gradients
        full_grad_x_ret = @sync @distributed (+) for i in 1:num_agents
            full_gradient_batch(x, data_cell[i])
        end
        return full_grad_x_ret;
    end


    t_start = time();
    x = zeros(dim);


    # initialize grad_estimate
    mini_batch_indices_arr = generate_mini_batches();
    grad_estimate = gradient(x, mini_batch_indices_arr, sample_times);

    results = zeros(div(num_iters, print_freq) + 1, 6);
    num_comm = 0.0;
    curr_obj = f_sum(x);
    num_simple_fn = 0.0;
    curr_grad_error = norm(full_gradient_sum(x) - grad_estimate);
    results[1, :] = [0, 0.0, num_simple_fn, num_comm, curr_obj, curr_grad_error];

    for iter in 1:num_iters
        # LMO
        v = LMO(grad_estimate);  # find argmax <grad_x, v>
        # update x
        x_old = x;
        x += v / num_iters;
        # sample a mini_batch
        mini_batch_indices_arr = generate_mini_batches();
        rho = rho_coef/(iter + max(1, rho_coef - 1))^rho_exp;
        grad_x = gradient(x, mini_batch_indices_arr, sample_times);
        hvp_x = gradient_diff(x, x_old, mini_batch_indices_arr, interpolate_times, sample_times);
        grad_estimate = (1 - rho) * (grad_estimate + hvp_x) + rho * grad_x;

        if mod(iter, print_freq) == 0
            t_elapsed = time() - t_start;
            curr_obj = f_sum(x);
            num_simple_fn = iter * (1 + cardinality * 2 * interpolate_times) * num_agents * mini_batch_size * sample_times;
            full_grad_x = full_gradient_sum(x);
            curr_grad_error = norm(full_grad_x - grad_estimate);
            results[div(iter, print_freq) + 1, :] = [iter, t_elapsed, num_simple_fn, num_comm, curr_obj, curr_grad_error];
        end
    end
    return results;
end

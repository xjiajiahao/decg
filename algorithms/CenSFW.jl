# centralized SFW (mini-batch version of Frank-Wolfe), here num_iters denotes #iteration
function CenSFW(dim, data_cell, LMO, f_batch, gradient_mini_batch, gradient_diff_mini_batch, mini_batch_size, num_iters, is_batch_size_increasing, cardinality, sample_times = 1)
    num_agents = size(data_cell, 2);  # num_agents should be 1
    function gradient(x, mini_batch_indices_arr, sample_times) # compute the sum of local gradients
        grad_x = @sync @distributed (+) for i in 1:num_agents
            gradient_mini_batch(x, data_cell[i], mini_batch_indices_arr[i], sample_times)
        end
        return grad_x;
    end

    function f_sum(x) # compute the global objective at x
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    function generate_mini_batches(curr_mini_batch_size)
        mini_batch_indices_arr = [[] for i=1:num_agents];
        if curr_mini_batch_size == 0
            return mini_batch_indices_arr;
        end
        for i in 1:num_agents
            num_users = length(data_cell[i]);
            mini_batch_indices_arr[i] = rand(1:num_users, curr_mini_batch_size);
        end
        return mini_batch_indices_arr;
    end

    t_start = time();
    x = zeros(dim);
    results = zeros(num_iters+1, 3);
    # initialize grad_estimate
    mini_batch_indices_arr = generate_mini_batches(mini_batch_size);
    grad_estimate = gradient(x, mini_batch_indices_arr, sample_times);
    for iter in 1:num_iters
        # sample a mini_batch
        if is_batch_size_increasing
            mini_batch_indices_arr = generate_mini_batches(mini_batch_size * iter^2);
        else
            mini_batch_indices_arr = generate_mini_batches(mini_batch_size);
        end
        grad_x = gradient(x, mini_batch_indices_arr, sample_times);
        # LMO
        v = LMO(grad_x);  # find argmax <grad_x, v>
        # update x
        x .+= v / num_iters;
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    if is_batch_size_increasing
        num_simple_fn = num_iters * (num_agents * (num_agents + 1) * (2 * num_agents + 1) / 6) * mini_batch_size * sample_times;
    else
        num_simple_fn = num_iters * num_agents * mini_batch_size * sample_times;
    end
    results = [num_iters, t_elapsed, num_simple_fn, num_comm, curr_obj];
    return results;
end

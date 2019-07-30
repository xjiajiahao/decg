# centralized SCGPP-FW, here num_iters denotes #iteration
function CenSCGPP(dim, data_cell, LMO, f_batch, gradient_mini_batch, gradient_diff_mini_batch, mini_batch_size, initial_sample_times, num_iters, cardinality, interpolate_times = 1, sample_times = 1)
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

    function generate_mini_batches()
        mini_batch_indices_arr = [[] for i=1:num_agents];
        if mini_batch_size == 0
            return mini_batch_indices_arr;
        end
        for i in 1:num_agents
            num_users = length(data_cell[i]);
            mini_batch_indices_arr[i] = rand(1:num_users, mini_batch_size);
        end
        return mini_batch_indices_arr;
    end

    function gradient_diff(x, y, mini_batch_indices_arr, interpolate_times, sample_times) # compute the sum of local gradients
        ret = @sync @distributed (+) for i in 1:num_agents
            gradient_diff_mini_batch(x, y, data_cell[i], mini_batch_indices_arr[i], interpolate_times, sample_times)
        end
        return ret;
    end

    t_start = time();
    x = zeros(dim);
    results = zeros(num_iters+1, 3);
    x_old = zeros(dim);
    grad_estimate = zeros(dim);
    for iter = 1 : num_iters
        # sample a mini_batch
        mini_batch_indices_arr = generate_mini_batches();
        # compute the gradient estimate
        if iter == 1
            grad_estimate = gradient(x, mini_batch_indices_arr, initial_sample_times);
        else
            hvp_x = gradient_diff(x, x_old, mini_batch_indices_arr, interpolate_times, sample_times);
            grad_estimate = grad_estimate + hvp_x;
        end
        # LMO
        v = LMO(grad_estimate);  # find argmax <grad_x, v>
        # update x
        x_old = x;
        x += v / num_iters;
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    # num_simple_fn = (num_iters - 1) * (1 + cardinality * interpolate_times) * num_agents * mini_batch_size * sample_times + num_agents * mini_batch_size * initial_sample_times;
    num_simple_fn = (num_iters - 1) * (1 + cardinality * interpolate_times) * num_agents * mini_batch_size * sample_times + num_agents * mini_batch_size * initial_sample_times;
    results = [num_iters, t_elapsed, num_simple_fn, num_comm, curr_obj];
    return results;
end

# centralized frank wolfe, here num_iters denotes #iteration
function CenCG(dim, data_cell, LMO, f_batch, gradient_batch, num_iters, print_freq = 100)
    num_agents = size(data_cell, 2);  # num_agents should be 1
    num_users = 0;
    for i = 1 : num_agents
        num_users = num_users + length(data_cell[i]);
    end

    function gradient_sum(x) # compute the sum of local gradients
        grad_x = @sync @distributed (+) for i in 1:num_agents
            gradient_batch(x, data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute the global objective function at x
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    t_start = time();
    x = zeros(dim);

    results = zeros(div(num_iters, print_freq) + 1, 5);
    num_comm = 0.0;
    curr_obj = f_sum(x);
    num_simple_fn = 0.0;
    results[1, :] = [0, 0.0, num_simple_fn, num_comm, curr_obj];

    for iter in 1:num_iters
        grad_x = gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x .+= v / num_iters;

        if mod(iter, print_freq) == 0
            t_elapsed = time() - t_start;
            curr_obj = f_sum(x);
            num_simple_fn = iter * num_users;
            results[div(iter, print_freq) + 1, :] = [num_iters, t_elapsed, num_simple_fn, num_comm, curr_obj];
        end
    end
    return results;
end



function CenSCG(dim, data_cell, LMO, f_batch, gradient_mini_batch, mini_batch_size, num_iters, rho_coef, rho_exp, full_gradient_batch, print_freq = 100, sample_times = 1)
    num_agents = size(data_cell, 2);  # num_agents should be 1
    function gradient_sum(x, sample_times) # compute the sum of local gradients
        mini_batch_grad_ret = @sync @distributed (+) for i in 1:num_agents
            num_users = length(data_cell[i]);
            mini_batch_indices = rand(1:num_users, mini_batch_size);
            gradient_mini_batch(x, data_cell[i], mini_batch_indices, sample_times)
        end
        return mini_batch_grad_ret;
    end

    function full_gradient_sum(x) # compute the sum of local gradients
        full_grad_x_ret = @sync @distributed (+) for i in 1:num_agents
            full_gradient_batch(x, data_cell[i])
        end
        return full_grad_x_ret;
    end

    function f_sum(x) # compute the global objective at x
        f_x_ret = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x_ret;
    end

    t_start = time();
    x = zeros(dim);
    grad_x = zeros(dim);

    results = zeros(div(num_iters, print_freq) + 1, 6);
    num_comm = 0.0;
    curr_obj = f_sum(x);
    num_simple_fn = 0.0;
    curr_grad_error = norm(full_gradient_sum(x) - grad_x)^2;
    results[1, :] = [0, 0.0, num_simple_fn, num_comm, curr_obj, curr_grad_error];

    for iter in 1:num_iters
        rho = rho_coef/(iter + 1)^rho_exp;
        new_grad =  gradient_sum(x, sample_times);
        grad_x = (1 - rho) * grad_x + rho * new_grad;
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x .+= v / num_iters;

        if mod(iter, print_freq) == 0
            t_elapsed = time() - t_start;
            curr_obj = f_sum(x);
            full_grad_x = full_gradient_sum(x - v / num_iters);
            curr_grad_error = norm(full_grad_x - grad_x)^2;
            num_simple_fn = iter * num_agents * mini_batch_size * sample_times;  # * 2 * dim @NOTE
            results[div(iter, print_freq) + 1, :] = [iter, t_elapsed, num_simple_fn, num_comm, curr_obj, curr_grad_error];
            # println("rho: ", rho);
            # println("grad_x: ", grad_x[1:5]);
            # println("new_grad: ", new_grad[1:5]);
            # println("full_gradient: ", full_grad_x[1:5]);
        end
    end
    return results;
end

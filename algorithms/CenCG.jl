# centralized frank wolfe, here num_iters denotes #iteration
function CenCG(dim, data_cell, LMO, f_batch, gradient_batch, num_iters)
    num_agents = size(data_cell, 2);
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
    for iter in 1:num_iters
        grad_x = gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x += v / num_iters;
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, curr_obj];
    return results;
end



function CenSCG(dim, data_cell, LMO, f_batch, gradient_mini_batch, mini_batch_size, num_iters, rho_coef, rho_exp)
    num_agents = size(data_cell, 2);
    function gradient_sum(x) # compute the sum of local gradients
        grad_x = @sync @distributed (+) for i in 1:num_agents
            num_users = length(data_cell[i]);
            mini_batch_indices = rand(1:num_users, mini_batch_size);
            gradient_mini_batch(x, data_cell[i], mini_batch_indices)
        end
        return grad_x;
    end

    function f_sum(x) # compute the global objective at x
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    t_start = time();
    x = zeros(dim);
    results = zeros(num_iters+1, 3);
    grad_x = zeros(dim);
    for iter in 1:num_iters
        rho = rho_coef/(iter + 1)^rho_exp;
        grad_x = (1 - rho) * grad_x + rho * gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x += v / num_iters;
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, curr_obj];
    return results;
end

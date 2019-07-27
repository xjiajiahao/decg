# centralized frank wolfe, here num_iters denotes #iteration
function CenPGD(dim, data_cell, PO, f_batch, gradient_batch, num_iters, eta_coef, eta_exp)
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
        eta = eta_coef * 1.0 / (iter * 1.0)^eta_exp;
        x = PO(x + eta * grad_x);
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, curr_obj];
    return results;
end



function CenPSGD(dim, data_cell, PO, f_batch, gradient_mini_batch, mini_batch_size, num_iters, eta_coef, eta_exp)
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
    for iter in 1:num_iters
        grad_x = gradient_sum(x);
        eta = eta_coef * 1.0 / (iter * 1.0 + 1)^eta_exp;
        x = PO(x + eta * grad_x);
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    num_comm = 0.0;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, curr_obj];
    return results;
end

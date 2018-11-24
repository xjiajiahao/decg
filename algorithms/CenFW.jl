# centralized frank wolfe, here num_iters denotes #iteration
function CenFW(dim, data_cell, LMO, f_batch, gradient_batch, num_iters)
    num_agents = size(data_cell, 2);
    function gradient_sum(x) # compute the sum of local gradients
        grad_x = @sync @distributed (+) for i in 1:num_agents
            gradient_batch(x, data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute the sum of local functions
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    t_start = time();
    x = zeros(dim);
    # results = zeros(num_iters+1, 3);
    # results[1, :] = [0, 0, f_sum(x)];
    for iter in 1:num_iters
        grad_x = gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x += v / num_iters;

        # curr_obj = f_sum(x);
        # t_elapsed = time() - t_start;
        # println("$(iter), $(t_elapsed), $(curr_obj)");
        # results[iter+1, :] = [iter, t_elapsed, curr_obj];
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    results = [num_iters, t_elapsed, curr_obj];
    return results;
end



function CenSFW(dim, data_cell, LMO, f_batch, gradient_batch, num_iters)
    num_agents = size(data_cell, 2);
    function gradient_sum(x) # compute the sum of local gradients
        grad_x = @sync @distributed (+) for i in 1:num_agents
            gradient_batch(x, data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute the sum of local functions
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    t_start = time();
    x = zeros(dim);
    results = zeros(num_iters+1, 3);
    # results[1, :] = [0, 0, f_sum(x)];
    grad_x = zeros(dim);
    for iter in 1:num_iters
        rho = 4.0/(iter + 8)^(2.0/3);
        grad_x = (1 - rho) * grad_x + rho * gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x += v / num_iters;

        # curr_obj = f_sum(x);
        # t_elapsed = time() - t_start;
        # println("$(iter), $(t_elapsed), $(curr_obj)");
        # results[iter+1, :] = [iter, t_elapsed, curr_obj];
    end
    t_elapsed = time() - t_start;
    curr_obj = f_sum(x);
    results = [num_iters, t_elapsed, curr_obj];
    return results;
end

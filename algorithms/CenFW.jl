# centralized frank wolfe, here num_iters denotes #iteration
function CenFW(dim, data_cell, LMO, f_batch, gradient_batch, num_iters)
    num_agents = size(data_cell, 2);
    function gradient_sum(x) # compute the sum of T gradients
        grad_x = @sync @parallel (+) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            gradient_batch(x, data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute the sum of T gradients
        f_x = @sync @parallel (+) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    t_start = time();
    x = zeros(dim);
    results = zeros(num_iters+1, 3);
    results[1, :] = [0, 0, f_sum(x)];
    # results[1, :] = [0, 0, 0];
    for iter in 1:num_iters
        # println("iter: ", iter, ", ", Dates.format(now(), "HH:MM:SS"));
        grad_x = gradient_sum(x);
        v = LMO(grad_x);  # find argmax <grad_x, v>
        x += v / num_iters; # @NOTE why choose 1/num_iters as step size

        curr_obj = f_sum(x);
        # curr_obj = 0;
        t_elapsed = time() - t_start;
        println("$(iter), $(t_elapsed), $(curr_obj)");
        results[iter+1, :] = [iter, t_elapsed, curr_obj];
    end
    return results;
end

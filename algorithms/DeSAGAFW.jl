function DeSAGAFW(dim, data_cell, num_agents, weights, LMO, f_batch, gradient_batch, num_iters)
    function gradient_cat(x) # compute the sum of T gradients
        grad_x = @sync @parallel (hcat) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            gradient_batch(x[:, i], data_cell[i])
        end
        # grad_x = pmap(gradient_batch,  data_cell);
        return grad_x;
    end

    function f_sum(x) # compute the sum of T gradients
        f_x = @sync @parallel (+) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    function LMO_cat(d)
        res = @sync @parallel (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end

    t_start = time();
    x = zeros(dim, num_agents);
    d = zeros(dim, num_agents);
    g = zeros(dim, num_agents);
    grad_x_old = zeros(dim, num_agents);
    results = zeros(num_iters+1, 3);
    results[1, :] = [0, 0, f_sum(mean(x, 2))];
    for iter in 1:num_iters
        grad_x = gradient_cat(x);
        if iter == 1
            g = grad_x;
        else
            g = d + grad_x - grad_x_old;
        end
        d = g * weights;
        v = LMO_cat(d);  # find argmax <grad_x, v>
        x = x*weights + v / num_iters; # @NOTE why choose 1/num_iters as step size
        grad_x_old = grad_x;
        # evaluate obj function
        x_bar = mean(x, 2);
        curr_obj = f_sum(x_bar);
        t_elapsed = time() - t_start;
        println("$(iter), $(t_elapsed), $(curr_obj)");
        results[iter+1, :] = [iter, t_elapsed, curr_obj];
    end
    return results;
end

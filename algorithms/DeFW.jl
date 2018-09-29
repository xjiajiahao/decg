# Mokhtari's decentralized FW
function DeFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha)
    function gradient_cat(x)  # compute local gradients simultaneously
        grad_x = @sync @parallel (hcat) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x)  # compute local objective functions simultaneously, and then output the sum
        f_x = @sync @parallel (+) for i in 1:num_agents
            f_batch(x[:, i], data_cell[i])
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
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # local gradient estimators
    num_comm = 0.0;
    results = zeros(num_iters+1, 4);
    # results[1, :] = [0, 0, 0, f_sum(mean(x, 2))];  # [#iter, time, #comm, obj_value]
    for iter in 1:num_iters
        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        d = (1 - alpha) * d * weights + alpha * grad_x; # first communication: exchange local gradient estimators
        v = LMO_cat(d);  # find argmax <d[:, i], v> for each agent i simultaneously
        x = x*weights + v / num_iters; # second communication: exchange local variables

        # evaluate obj function at mean(x[i])
        # x_bar = mean(x, 2);
        # curr_obj = f_sum(x_bar);
        t_elapsed = time() - t_start;
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        println("$(iter), $(t_elapsed)");
        # results[iter+1, :] = [iter, t_elapsed, num_comm, curr_obj];
    end
    t_elapsed = time() - t_start;
    results = [num_iters, t_elapsed, num_comm, f_sum(x)];
    return results;
end


function DeSFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha, phi)
    function gradient_cat(x) # compute local gradients simultaneously
        grad_x = @sync @parallel (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute local objective functions simultaneously, and then output the sum
        f_x = @sync @parallel (+) for i in 1:num_agents
            f_batch(x[:, i], data_cell[i])
        end
        return f_x;
    end

    function LMO_cat(d)  # perform the linear programming simultaneously
        res = @sync @parallel (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end

    t_start = time();
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # local gradient estimators
    g = zeros(dim, num_agents);  # local stochastic averaged gradient
    num_comm = 0.0;
    results = zeros(num_iters+1, 4);
    # results[1, :] = [0, 0, 0, f_sum(mean(x, 2))];  # [#iter, time, #comm, obj_value]
    for iter in 1:num_iters
        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        g = (1 - phi) * g + phi * grad_x;  # update the local stochastic averaged gradient
        d = (1 - alpha) * d * weights + alpha * g;  # first communication: exchange gradient estimators
        v = LMO_cat(d);  # find argmax <d[i], v> for each agent i
        x = x*weights + v / num_iters;  # second communication: exchange local variables

        # evaluate obj function at mean(x)
        # x_bar = mean(x, 2);
        # curr_obj = f_sum(x_bar);
        t_elapsed = time() - t_start;
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        println("$(iter), $(t_elapsed)");
        # results[iter+1, :] = [iter, t_elapsed, num_comm, curr_obj];
    end
    t_elapsed = time() - t_start;
    results = [num_iters, t_elapsed, num_comm, f_sum(x)];
    return results;
end

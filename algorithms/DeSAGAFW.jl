# our methods
function DeSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters)
    function gradient_cat(x) # compute local gradients simultaneously
        grad_x = @sync @parallel (hcat) for i in 1:num_agents # the documentation says that @paralel for can handle situations where each iteration is tiny
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x)  # compute local objective functions simultaneously, and then output the sum
        f_x = @sync @parallel (+) for i in 1:num_agents
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
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # aggregated gradient estimators
    g = zeros(dim, num_agents);  # local SAGA-style gradient estimators
    grad_x_old = zeros(dim, num_agents);  # used to store the old local gradients
    num_comm = 0.0;
    results = zeros(num_iters+1, 4);
    # results[1, :] = [0, 0, 0, f_sum(mean(x, 2))];  # [#iter, time, #comm, obj_value]
    for iter in 1:num_iters
        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        if iter == 1
            g = grad_x;
        else
            g = d + grad_x - grad_x_old;
        end
        d = g * weights;  # first communication: exchange g
        v = LMO_cat(d);  # find argmax <d[i], v> for all agents i simultaneously
        x = x*weights + v / num_iters;  # second communication: exchange local variables
        grad_x_old = grad_x;
        # evaluate obj function
        # x_bar = mean(x, 2);
        # curr_obj = f_sum(x_bar);
        # t_elapsed = time() - t_start;
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        # println("$(iter), $(t_elapsed)");
        # results[iter+1, :] = [iter, t_elapsed, num_comm, curr_obj];
    end
    t_elapsed = time() - t_start;
    avg_f = 0;
    for i = 1:num_agents
        avg_f += f_sum(x[:, i]);
    end
    avg_f = avg_f / num_agents;
    results = [num_iters, t_elapsed, num_comm, avg_f];
    return results;
end


function DeSSAGAFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters)
    function gradient_cat(x, sample_times) # compute local gradients simultaneously
        grad_x = @sync @parallel (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i], sample_times)  # @TODO t^2 should be smaller than the batch size b
        end
        return grad_x;
    end

    function f_sum(x)  # compute local objective functions simultaneously, and then output the sum
        f_x = @sync @parallel (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
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
    d = zeros(dim, num_agents);  # aggregated gradient estimators
    g = zeros(dim, num_agents);  # local SAGA-style gradient
    grad_x_old = zeros(dim, num_agents);  # used to store the old gradient
    num_comm = 0.0;
    results = zeros(num_iters+1, 4);
    # results[1, :] = [0, 0, 0, f_sum(mean(x, 2))];  # [#iter, time, #comm, obj_value]
    for iter in 1:num_iters
        sample_times = iter^2;
        grad_x = gradient_cat(x, sample_times);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        if iter == 1
            g = grad_x;
        else
            g = d + grad_x - grad_x_old;
        end
        d = g * weights;  # first communication: exchange gradient estimators
        v = LMO_cat(d);  # find argmax <d[i], v> each all agent i
        x = x*weights + v / num_iters;  # second communication: exchange local variables
        grad_x_old = grad_x;
        # evaluate obj function
        # x_bar = mean(x, 2);
        # curr_obj = f_sum(x_bar);
        # t_elapsed = time() - t_start;
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        # println("$(iter), $(t_elapsed)");
        # results[iter+1, :] = [iter, t_elapsed, num_comm, curr_obj];
    end
    t_elapsed = time() - t_start;
    avg_f = 0;
    for i = 1:num_agents
        avg_f += f_sum(x[:, i]);
    end
    avg_f = avg_f / num_agents;
    results = [num_iters, t_elapsed, num_comm, avg_f];
    return results;
end

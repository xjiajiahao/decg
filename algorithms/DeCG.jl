# Mokhtari's decentralized FW
function DeCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha)
    function gradient_cat(x)  # compute local gradients simultaneously
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x)  # compute the global objective function at x
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    function LMO_cat(d)
        res = @sync @distributed (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end

    t_start = time();
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # local gradient estimators
    num_comm = 0.0;
    for iter in 1:num_iters
        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        d = (1 - alpha) * d * weights + alpha * grad_x; # first communication: exchange local gradient estimators
        v = LMO_cat(d);  # find argmax <d[:, i], v> for each agent i simultaneously
        x = x*weights + v / num_iters; # second communication: exchange local variables
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
    end
    t_elapsed = time() - t_start;
    avg_f = 0;
    for i = 1:num_agents
        avg_f += f_sum(x[:, i]);
    end
    avg_f = avg_f / num_agents;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, avg_f];
    return results;
end


function DeSCG(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, alpha, phi)
    function gradient_cat(x) # compute local gradients simultaneously
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x) # compute the global objective function at x
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    function LMO_cat(d)
        res = @sync @distributed (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end

    t_start = time();
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # local gradient estimators
    g = zeros(dim, num_agents);  # local stochastic averaged gradient
    num_comm = 0.0;
    for iter in 1:num_iters
        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        g = (1 - phi) * g + phi * grad_x;  # update the local stochastic averaged gradient
        d = (1 - alpha) * d * weights + alpha * g;  # first communication: exchange gradient estimators
        v = LMO_cat(d);  # find argmax <d[i], v> for each agent i
        x = x*weights + v / num_iters;  # second communication: exchange local variables
        num_comm += 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
    end
    t_elapsed = time() - t_start;
    avg_f = 0;
    for i = 1:num_agents
        avg_f += f_sum(x[:, i]);
    end
    avg_f = avg_f / num_agents;
    num_local_grad = num_iters;
    results = [num_iters, t_elapsed, num_local_grad, num_comm, avg_f];
    return results;
end

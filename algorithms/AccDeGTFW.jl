# AccDeGTFW or AccDeSGTFW return a 5-by-1 vector [#iterations, elapsed time, #local exact/stochastoc gradient evaluations per node, #doubles transferred in the network, averaged objective function];
function AccDeGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, beta, K)
    function gradient_cat(x) # compute local gradients simultaneously
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i])
        end
        return grad_x;
    end

    function f_sum(x)  # compute global objective at x
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
    d = zeros(dim, num_agents);  # aggregated gradient estimators
    g = gradient_cat(x);  # local SAGA-style gradient estimators
    grad_x_old = g;  # used to store the old local gradients
    num_comm = 0.0;
    num_local_grad = num_iters;
    for iter in 1:num_iters
        xhat, dhat = ChebyshevComm(x, g, weights, beta, K);
        num_comm += K * 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        v = LMO_cat(dhat);  # find argmax <d[i], v> for all agents i simultaneously
        x = xhat + v / num_iters;  # second communication: exchange local variables

        grad_x = gradient_cat(x);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        g = dhat + grad_x - grad_x_old;
        grad_x_old = grad_x;
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


function AccDeSGTFW(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters, beta, K)
    function gradient_cat(x, sample_times) # compute local gradients simultaneously
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:, i], data_cell[i], sample_times)  # @TODO t^2 should be smaller than the batch size b
        end
        return grad_x;
    end

    function f_sum(x)  # compute local objective functions simultaneously, and then output the sum
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data_cell[i])
        end
        return f_x;
    end

    function LMO_cat(d)  # perform the linear programming simultaneously
        res = @sync @distributed (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end

    t_start = time();
    x = zeros(dim, num_agents);  # local variables
    d = zeros(dim, num_agents);  # aggregated gradient estimators
    g = gradient_cat(x, 1);  # local SAGA-style gradient estimators
    grad_x_old = g;  # used to store the old local gradients
    num_comm = 0.0;
    num_local_stoch_grad = 1.0;
    for iter in 1:num_iters
        xhat, dhat = ChebyshevComm(x, g, weights, beta, K);
        num_comm += K * 2*dim*num_out_edges;  # 1 for local gradients, 1 for local variables
        v = LMO_cat(dhat);  # find argmax <d[i], v> for all agents i simultaneously
        x = xhat + v / num_iters;  # second communication: exchange local variables

        if iter == num_iters
            break;
        end

        sample_times = (iter+1)^2;
        grad_x = gradient_cat(x, sample_times);  # compute the true local gradients, grad_x is a dim-by-num_agents matrix
        num_local_stoch_grad += sample_times;
        g = dhat + grad_x - grad_x_old;
        grad_x_old = grad_x;
    end
    t_elapsed = time() - t_start;
    avg_f = 0;
    for i = 1:num_agents
        avg_f += f_sum(x[:, i]);
    end
    avg_f = avg_f / num_agents;
    results = [num_iters, t_elapsed, num_local_stoch_grad, num_comm, avg_f];
    return results;
end

function ChebyshevComm(x, g, weights, beta, K)
    if K <= 1
        return (x * weights, g * weights);
    end
    a_old = 1;
    a_curr = 1/beta;
    x_old = x * 1.0;
    g_old = g * 1.0;
    x_curr = (1/beta)* x_old * weights;
    g_curr = (1/beta)* g_old * weights;
    a_new = a_curr;
    x_new = x_curr;
    g_new = g_curr;
    for k = 1:K-1
        a_new = (2 / beta) * a_curr - a_old;
        x_new = (2 / beta) * x_curr * weights - x_old;
        g_new = (2 / beta) * g_curr * weights - g_old;

        a_old = a_curr;
        a_curr = a_new;
        x_old = x_curr;
        x_curr = x_new;
        g_old = g_curr;
        g_curr = g_new;
    end
    x_ret = x_new/a_new;
    g_ret = g_new/a_new;
    return (x_ret, g_ret);
end

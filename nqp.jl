@everywhere function f(x, data, u) # data is a 1-by-batch_size cell where each entry is a dim-by-dim matrix H
    res = 0;
    for H in data
        res += (0.5 .* x - u)' * H * x;
    end
    return res;
end

@everywhere function f_batch(x, data_cell_and_u)
    sum_f = 0;
    for data in data_cell
        sum_f += f(x, data, u);
    end
    return sum_f;
end

@everywhere function gradient(x, data, u) # compute the gradient
    dim = length(x);
    res = zeros(dim);
    for H in data
        res += H * (x - u);
    end
    return res;
end

@everywhere function gradient_batch(x, data_cell, u) # compute the true gradient
    sum_g = 0;
    for data in data_cell
        sum_g += gradient(x, data, u);
    end
    return sum_g;
end

@everywhere function stochastic_gradient(x, data, u) # compute stochastic gradient
    dim = length(x);
    rand_idx = rand(1:length(data));
    H = data[rand_idx];
    res = H * ( x - u );
    return res;
end

@everywhere function stochastic_gradient_batch(x, data_cell, u)
    res = 0;
    for data in data_cell
        res += stochastic_gradient(x, data, u);
    end
    return res;
end

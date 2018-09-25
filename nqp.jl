@everywhere function f(x, data) # data is a 1-by-batch_size cell where each entry is a dim-by-dim matrix H
    dim = length(x);
    batch_size = length(data);
    u = ones(dim, 1);
    res = 0;
    for H in data
        res += (0.5 .* x - u)' * H * x;
    end
    return reshape(res, 1)[1]/batch_size;
end

@everywhere function f_batch(x, data_cell)
    sum_f = 0;
    batch_size = length(data_cell);
    for data in data_cell
        sum_f += f(x, data);
    end
    return sum_f/batch_size;
end

@everywhere function gradient(x, data) # compute the gradient
    dim = length(x);
    u = ones(dim, 1);
    res = zeros(dim);
    batch_size = length(data);
    for H in data
        res += H * (x - u);
    end
    return squeeze(res, 2)/batch_size;
end

@everywhere function gradient_batch(x, data_cell) # compute the true gradient
    sum_g = 0;
    batch_size = length(data_cell);
    for data in data_cell
        sum_g += gradient(x, data);
    end
    return sum_g/batch_size;
end

@everywhere function stochastic_gradient(x, data, sample_times) # compute stochastic gradient
    dim = length(x);
    u = ones(dim, 1);
    res = zeros(dim, 1);
    for i = 1 : sample_times
        rand_idx = rand(1:length(data));
        H = data[rand_idx];
        res = H * ( x - u );
    end
    res = squeeze(res./sample_times, 2);
    return res;
end

@everywhere function stochastic_gradient_batch(x, data_cell, sample_times = 1)
    res = 0;
    batch_size = length(data_cell);
    for data in data_cell
        res += stochastic_gradient(x, data, sample_times);
    end
    return res/batch_size;
end

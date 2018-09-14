@everywhere function f_discrete_batch(s::IntSet, data_mat)
    indices = [i for i in s];
    max_values, max_idx = findmax(data_mat[indices, :], 1);
    sum_f = sum(max_values);
    return sum_f;
end

@everywhere function f_extension_sample(x, ratings, sample_times = 1e5) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    res = 0;
    for j in 1:sample_times
        # 1. Generate a random set X
        tmp_f = 0;
        for index in 1:size(ratings, 1)
            if rand() <= x[round(Int, ratings[index, 1])]
                tmp_f = ratings[index, 2];
                break;
            end
        end
        res += tmp_f;
    end
    res /= sample_times;
    return res;
end

@everywhere function f_extension(x, ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    res = 0;
    prod = 1;
    for index in 1:size(ratings, 1) # for all rated movies from high to low
        x_current_coord = x[round(Int, ratings[index, 1])];
        res += ratings[index, 2] * x_current_coord * prod;
        prod *= 1 - x_current_coord;
        if prod == 0
            break;
        end
    end
    return res;
end

@everywhere function f_extension_batch(x, batch_ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    sum_f = 0;
    for ratings in batch_ratings
        sum_f += f_extension(x, ratings);
    end
    return sum_f;
end

@everywhere function partial_extension_sample(x, ratings, i, sample_times = 1e5) # compute partial derivative : O(#sample * #ratings)
    res = 0;

    the_index = findfirst(ratings[:, 1], i);
    if the_index == 0 # this means f(R+{i}) = f(R\{i}), for any R, then no need to sample
        return 0;
    end

    for j in 1:sample_times
        # 1. Generate a random set X\i
        tmp_f = 0;
        for index in 1:size(ratings, 1)
            if index == the_index # exclude the i'th index
                continue;
            end
            if rand() <= x[round(Int, ratings[index, 1])]
                tmp_f = ratings[index, 2];
                break;
            end
        end
        # 2. compute f(X+i) - f(X\i)
        tmp_partial = max(tmp_f, ratings[the_index, 2]) - tmp_f;
        res += tmp_partial;
    end
    res /= sample_times;
    return res;
end

@everywhere function partial_extension(x, ratings, i) # compute partial derivative : O(#ratings)
    res_union = 0;
    res_diff = 0;
    prod = 1;

    # the_index = findfirst(ratings[:, 1], i);  # @NOTE bottleneck
    the_index = 0;  # the for loop is equivilant to the above line, but faster
    for tmp_i in 1 : size(ratings, 1)
        if i == ratings[tmp_i, 1]
            the_index = tmp_i;
            break;
        end
    end

    if the_index == 0 # this means f(R+{i}) = f(R\{i}), for any R, then no need to sample
        return 0;
    end

    for index in 1:size(ratings, 1) # for all rated movies from high to low
        if index == the_index
            res_union = res_diff + ratings[index, 2] * 1 * prod;
        else
            x_current_coord = x[round(Int, ratings[index, 1])];
            res_diff += ratings[index, 2] * x_current_coord * prod;  # @NOTE bottleneck
            prod *= 1 - x_current_coord;
        end
        if prod == 0
            break;
        end
    end
    return res_union - res_diff;
end

@everywhere function gradient_extension(x, ratings) # compute the gradient: O(n + #sample * #ratings^2)
    dim = length(x);
    res = zeros(dim);
    for i in 1:dim
        res[i] = partial_extension(x, ratings, i);
    end
    return res;
end

@everywhere function gradient_extension_batch(x, batch_ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    sum_gradient = 0;
    for ratings in batch_ratings
        sum_gradient += gradient_extension(x, ratings);
    end
    return sum_gradient;
end

@everywhere function stochastic_gradient_extension(x, ratings, sample_times) # compute stochastic gradient: O(???)
    function stochastic_partial_extension(x, ratings, i, rand_vec)
        the_index = findfirst(ratings[:, 1], i);
        if the_index == 0 # this means f(R+i) = f(R\i), for any R, then no need to sample
            return 0;
        end

        # 1. Generate a random set X\i
        tmp_f = 0;
        for index in 1:size(ratings, 1)
            if index == the_index # exclude the i'th index
                continue;
            end
            tmp_index = round(Int, ratings[index, 1]);
            if rand_vec[tmp_index] <= x[tmp_index]
                tmp_f = ratings[index, 2];
                break;
            end
        end
        # 2. compute f(X+i) - f(X\i)
        stochastic_partial = max(tmp_f, ratings[the_index, 2]) - tmp_f;
        return stochastic_partial;
    end

    dim = length(x);
    stochastic_grad = zeros(dim);
    for i = 1:sample_times
        rand_vec = rand(dim);
        for i in 1:dim
            stochastic_grad[i] += stochastic_partial_extension(x, ratings, i, rand_vec);
        end
    end
    return stochastic_grad/sample_times;
end

@everywhere function stochastic_gradient_extension_batch(x, batch_ratings, sample_times = 1) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    sum_stochastic_gradient = 0;
    for ratings in batch_ratings
        sum_stochastic_gradient += stochastic_gradient_extension(x, ratings, sample_times);
    end
    return sum_stochastic_gradient;
end

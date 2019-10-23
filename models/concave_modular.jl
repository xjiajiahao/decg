using Distributed
@everywhere using Random
# @everywhere function f_discrete_batch(s, data_mat)
#     indices = [i for i in s];
#     max_values, max_idx = findmax(data_mat[indices, :], 1);
#     sum_f = sum(max_values);
#     return sum_f;
# end

@everywhere function f_extension_sample(x, ratings, indices_in_ratings, rand_vec, sample_times = 1e5) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    res = 0;
    dim = length(x);
    fill!(indices_in_ratings, zero(Int));
    tmp_idx = 0;
    nnz = size(ratings, 2);
    for i = 1 : nnz
        tmp_idx = round(Int, ratings[1, i]);
        indices_in_ratings[i] = tmp_idx;
    end
    rand_vec_view = view(rand_vec, 1:nnz);
    for j in 1:sample_times
        # 1. Generate a random set X
        Random.rand!(rand_vec_view);
        # compute the sum of ratings in S
        base_sum = 0;
        for i = 1 : nnz
            tmp_index = indices_in_ratings[i];
            if rand_vec_view[i] <= x[tmp_index]
                base_sum += ratings[2, i];
            end
        end
        res += sqrt(base_sum);
    end
    res /= sample_times;
    return res;
end

# @everywhere function f_extension(x, ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
#     res = 0;
#     prod = 1;
#     for index in 1:size(ratings, 2) # for all rated movies from high to low
#         x_current_coord = x[round(Int, ratings[1, index])];
#         res += ratings[2, index] * x_current_coord * prod;
#         prod *= 1 - x_current_coord;
#         if prod == 0
#             break;
#         end
#     end
#     return res;
# end

@everywhere function f_extension_batch(x, batch_ratings, sample_times=1e3) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    sum_f = 0;
    dim = length(x);
    indices_in_ratings = zeros(Int64, dim);
    rand_vec = zeros(dim);
    for ratings in batch_ratings
        # sum_f += f_extension(x, ratings);
        sum_f += f_extension_sample(x, ratings, indices_in_ratings, rand_vec, sample_times);
    end
    return sum_f;
end

@everywhere function stochastic_gradient_extension!(x::Vector{Float64}, ratings::Array{Float64, 2}, sample_times::Int64, indices_in_ratings::Vector{Int64}, ret_stochastic_grad::Vector{Float64}, rand_vec::Vector{Float64})
    dim = length(x);
    fill!(indices_in_ratings, zero(Int));
    tmp_idx = 0;
    nnz = size(ratings, 2);
    for i = 1 : nnz
        tmp_idx = round(Int, ratings[1, i]);
        indices_in_ratings[i] = tmp_idx;
    end

    rand_vec_view = view(rand_vec, 1:nnz);
    for j = 1:sample_times
        Random.rand!(rand_vec_view);

        base_sum = 0;
        # 1. compute the sum of ratings in S
        for i = 1 : nnz
            tmp_index = indices_in_ratings[i];
            if rand_vec_view[i] <= x[tmp_index]
                base_sum += ratings[2, i];
            end
        end
        sqrt_base_sum = sqrt(base_sum);
        # 2. compute the partial derivative of each coordinate
        for i = 1 : nnz
            tmp_index = indices_in_ratings[i];
            if rand_vec_view[i] <= x[tmp_index]
                ret_stochastic_grad[tmp_index] += sqrt_base_sum - sqrt(base_sum - ratings[2, i]);
            else
                ret_stochastic_grad[tmp_index] += sqrt(base_sum + ratings[2, i]) - sqrt_base_sum;
            end
        end
    end
    nothing
end

@everywhere function stochastic_gradient_extension_batch(x, batch_ratings, sample_times = 1) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    dim = length(x);
    stochastic_gradient = zeros(dim);
    indices_in_ratings = zeros(Int64, dim);
    rand_vec = zeros(dim);
    for ratings in batch_ratings
        stochastic_gradient_extension!(x, ratings, sample_times, indices_in_ratings, stochastic_gradient, rand_vec);
    end
    stochastic_gradient = stochastic_gradient ./ sample_times;
    return stochastic_gradient;
end

@everywhere function stochastic_gradient_extension_mini_batch(x, batch_ratings, mini_batch_indices, sample_times = 1) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    dim = length(x);
    num_users = length(batch_ratings);
    mini_batch_size = length(mini_batch_indices);
    stochastic_gradient = zeros(dim);
    indices_in_ratings = zeros(Int64, dim);
    rand_vec = zeros(dim);
    if length(mini_batch_indices) > 0
        for i in mini_batch_indices
            ratings = batch_ratings[i];
            stochastic_gradient_extension!(x, ratings, sample_times, indices_in_ratings, stochastic_gradient, rand_vec);
        end
        stochastic_gradient = stochastic_gradient ./ sample_times .* (num_users / mini_batch_size);
    else
        for ratings in batch_ratings
            stochastic_gradient_extension!(x, ratings, sample_times, indices_in_ratings, stochastic_gradient, rand_vec);
        end
        stochastic_gradient = stochastic_gradient ./ sample_times;
    end
    return stochastic_gradient;
end

@everywhere function stochastic_gradient_diff_extension_mini_batch(x, y, batch_ratings, mini_batch_indices, interpolate_times = 1, sample_times = 1) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    dim = length(x);
    num_users = length(batch_ratings);
    mini_batch_size = length(mini_batch_indices);
    stochastic_gradient_diff = zeros(dim);
    indices_in_ratings = zeros(Int64, dim);
    rand_vec = zeros(dim);
    x_y_diff = x - y;
    # interpolate between y and x
    for curr_interpolate = 1 : interpolate_times
        convex_combination_ratio = rand();
        x_y_interpolate = y + convex_combination_ratio * x_y_diff;
        if length(mini_batch_indices) > 0
            for i in mini_batch_indices
                ratings = batch_ratings[i];
                stochastic_hvp_extension!(x_y_interpolate, x_y_diff, ratings, sample_times, indices_in_ratings, stochastic_gradient_diff, rand_vec);
            end
        else
            for ratings in batch_ratings
                stochastic_hvp_extension!(x_y_interpolate, x_y_diff, ratings, sample_times, indices_in_ratings, stochastic_gradient_diff, rand_vec);
            end
        end
    end

    if length(mini_batch_indices) > 0
        stochastic_gradient_diff = stochastic_gradient_diff ./ interpolate_times ./ sample_times .* (num_users / mini_batch_size);
    else
        stochastic_gradient_diff = stochastic_gradient_diff ./ interpolate_times ./ sample_times;
    end
    return stochastic_gradient_diff;
end

@everywhere function stochastic_hvp_extension!(x::Vector{Float64}, v::Vector{Float64}, ratings::Array{Float64, 2}, sample_times::Int64, indices_in_ratings::Vector{Int64}, ret_stochastic_hvp::Vector{Float64}, rand_vec::Vector{Float64})  # to estimate the hessian-vector product \nabla^2 f(x) v
    dim = length(x);
    fill!(indices_in_ratings, zero(Int));
    tmp_idx = 0;
    nnz = size(ratings, 2);
    for i = 1 : nnz
        tmp_idx = round(Int, ratings[1, i]);
        indices_in_ratings[i] = tmp_idx;
    end

    rand_vec_view = view(rand_vec, 1:nnz);
    for curr_sample_count = 1:sample_times
        # Random.rand!(rand_vec_view);

        for j = 1:dim  # add/subtract the element j to/from the sampled set S
            curr_scalar = v[j];
            if -1e-8 <= curr_scalar && curr_scalar <= 1e-8
                continue;
            end

            for tmp_coord = 1:nnz
                if tmp_coord == j
                    continue;
                end
                Random.rand!(rand_vec_view);
                # 1. evaluate the sum of ratings of S+{j}
                base_sum = 0;
                for i = 1 : nnz
                    tmp_index = indices_in_ratings[i];
                    if rand_vec_view[i] <= x[tmp_index] || indices_in_ratings[i] == j
                        base_sum += ratings[2, i];
                    end
                end
                sqrt_base_sum = sqrt(base_sum);
                # 2. compute the partial derivative of the ith coordinate
                tmp_index = indices_in_ratings[tmp_coord];
                if rand_vec_view[tmp_coord] <= x[tmp_index] && indices_in_ratings[tmp_coord] != j  # @NOTE the diagonal entries of the hessian matrix is 0
                    ret_stochastic_hvp[tmp_index] += curr_scalar * (sqrt_base_sum - sqrt(base_sum - ratings[2, tmp_coord]));
                elseif rand_vec_view[tmp_coord] > x[tmp_index]
                    ret_stochastic_hvp[tmp_index] += curr_scalar * (sqrt(base_sum + ratings[2, tmp_coord]) - sqrt_base_sum);
                end

                # 3. evaluate the sum of ratings of S\{j}
                base_sum = 0;
                for i = 1 : nnz
                    tmp_index = indices_in_ratings[i];
                    if rand_vec_view[i] <= x[tmp_index] && indices_in_ratings[i] != j
                        base_sum += ratings[2, i];
                    end
                end
                sqrt_base_sum = sqrt(base_sum);

                # 4. compute the partial derivative of each coordinate
                tmp_index = indices_in_ratings[tmp_coord];
                if rand_vec_view[tmp_coord] <= x[tmp_index] && indices_in_ratings[tmp_coord] != j  # @NOTE the diagonal entries of the hessian matrix is 0
                    ret_stochastic_hvp[tmp_index] += - curr_scalar * (sqrt_base_sum - sqrt(base_sum - ratings[2, tmp_coord]));
                elseif rand_vec_view[tmp_coord] > x[tmp_index]
                    ret_stochastic_hvp[tmp_index] += - curr_scalar * (sqrt(base_sum + ratings[2, tmp_coord]) - sqrt_base_sum);
                end
            end
        end
    end
    nothing
end

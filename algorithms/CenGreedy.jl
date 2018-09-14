# the standard discrete greedy algorithm
function CenGreedy(dim, data_mat, f_batch, k::Int64)
    t_start = time();
    indices = [1:dim;];
    s = IntSet();
    for i = 1 : k  # we have to loop k times, at each time we add exactly one element to the set S
        max_obj = 0;
        max_idx = 0;
        # find j that maximizes f(union(S, j))
        for j = 1 : dim - (i - 1)
            tmp_idx = j;
            tmp_s = union(s, [j]);
            tmp_obj = f_batch(tmp_s, data_mat);
            if (tmp_obj > max_obj)
                max_obj = tmp_obj;
                max_idx = j;
            end
        end
        # swap the max_idx and update obj_value
        tmp_idx = indices[max_idx];
        indices[max_idx] = indices[dim - (i - 1)];
        indices[dim - (i - 1)] = tmp_idx;
        s = union(s, tmp_idx);
        t_elapsed = time() - t_start;
        println("iter: $(i), elapsed time: $(t_elapsed), obj: $(max_obj)");
    end
    res = f_batch(s, data_mat);
    println("centralized greedy result: $(res)");
    return res;
end

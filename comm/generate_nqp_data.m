function [data_cell, A, dim, u, b] = generate_nqp_data(dim, num_agents, batch_size, num_constraints, magnitude_data)
% data_cell is a 1-by-num_agents cell, each element of data_cell is a 1-by-batch_size cell which contains #batch_size matrices H_i of size dim-by-dim
% the NQP problem is defined as f_i(x) = ( x/2 - u )^T H_i x, s.t. {x | 0<=x<=u, Ax<=b}, where A is the constraint_mat of size num_constraints-by-dim


ROOT = '../data/';
rng(1); % For reproducibility
data_cell = cell(1, num_agents);
for i = 1 : num_agents
    tmp_cell = cell(1, batch_size);
    for j = 1 : batch_size
        tmp_cell{j} = -abs(magnitude_data) .* rand(dim);
    end
    data_cell{i} = tmp_cell;
end

A = rand(num_constraints, dim);
u = ones(dim, 1);
b = ones(num_constraints, 1);

filename = [ROOT, 'NQP_50_agents.mat'];
save(filename, 'data_cell', 'A', 'dim', 'u', 'b');

end

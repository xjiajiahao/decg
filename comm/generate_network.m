n_nodes = 50;
ROOT = '../data/';
num_tests = 1e4;

min_beta = inf;
max_beta = -inf;
avg_beta = 0;

for k = 1 : num_tests
% pl = 10.0/(n_nodes - 1);
pl = 0.4;
adj_matrix = gennetwork(n_nodes,pl);  %%% random graph with probability pl and size n

% D=diag(sum(A));
% L=D-A;
% tau=(2/3)*max(eig(L));
% W=eye(n_nodes)-L/tau;

degrees = sum(adj_matrix, 2);

weights = zeros(n_nodes);
for i = 1 : n_nodes
    for j = 1 : n_nodes
        if (adj_matrix(i, j) == 0)
            continue;
        end
        % weights(i, j)  = 1.0 / (1.0 + max(degrees(i), degrees(j)));
        weights(i, j)  = 1.0 / (max(degrees(i), degrees(j)));
    end
end

sum_weights = sum(weights, 2);
weights = weights + diag(1 - sum_weights);
% filename = [ROOT, 'weights_er_50.mat'];
% save(filename, 'weights');

max_two_eigs_in_magnitude = eigs(weights, 2);
beta = min(max_two_eigs_in_magnitude);

if beta < min_beta
    min_beta = beta;
end
if beta > max_beta
    max_beta = beta;
end
avg_beta = avg_beta + beta/num_tests;
end

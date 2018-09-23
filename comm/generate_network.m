n_nodes = 100;

pl = 20.0/(n_nodes - 1);
% pl = 0.4;
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
        weights(i, j)  = 1.0 / (1.0 + max(degrees(i), degrees(j)));
    end
end

sum_weights = sum(weights, 2);
weigths = weights + diag(1 - sum_weights);

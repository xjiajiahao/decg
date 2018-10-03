n_nodes = 50;

adj_matrix = zeros(n_nodes);
for i = 1:n_nodes
    j = mod(i, n_nodes) + 1;
    if j == 1
        continue;
    end
    adj_matrix(i, j) = 1;
    adj_matrix(j, i) = 1;
end

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
weights = weights + diag(1 - sum_weights);

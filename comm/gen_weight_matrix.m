function weights = gen_weight_matrix(num_nodes, graph_style, pl=0.4)
    ROOT = '../data/';

    if strcmp(graph_style, 'er') == true
        adj_matrix = gen_network_topology(num_nodes,pl);  %%% random graph with probability pl and size n
        degrees = sum(adj_matrix, 2);
        weights = zeros(num_nodes);
        for i = 1 : num_nodes
            for j = 1 : num_nodes
                if (adj_matrix(i, j) == 0)
                    continue;
                end
                % weights(i, j)  = 1.0 / (1.0 + max(degrees(i), degrees(j)));
                weights(i, j)  = 1.0 / (max(degrees(i), degrees(j)));
            end
        end
        sum_weights = sum(weights, 2);
        weights = weights + diag(1 - sum_weights);

    else if strcmp(graph_style, 'line') == true
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

    else if strcmp(graph_style, 'complete') == true
        adj_matrix = ones(n_nodes);
        adj_matrix = adj_matrix - diag(ones(n_nodes, 1));

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

    else
        error('graph_style must be ''er'', ''line'', or ''complete''!');
    end

    filename = [ROOT, 'weights_er_50.mat'];
    save(filename, 'weights');
end

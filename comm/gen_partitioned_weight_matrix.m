function weights = gen_partitioned_weight_matrix(num_nodes, graph_style)
    num_nodes = int32(num_nodes);
    ROOT = '../data/';
    filename = [ROOT, 'weights_', graph_style, '_', num2str(num_nodes), '.mat'];
    load(filename);
    networkInfo = struct;
    eigs = sort(eig(weights), 'descend');
    assert(abs(eigs(1) - 1.0) <= 1e-8);

    networkInfo.beta = max(abs(eigs(2)), eigs(num_nodes));
    networkInfo.numNodes = num_nodes;

    for i = 1 : num_nodes
        networkInfo.nodeID = int32(i-1);
        tmpWeightVec = weights(:, i);
        tmpWeightVec = sparse(tmpWeightVec);
        networkInfo.neighborWeights = tmpWeightVec;

        filename = [ROOT, 'weights_', graph_style, '_rank=', num2str(networkInfo.nodeID, '%02d'), '_', num2str(num_nodes), '.mat'];
        save(filename, 'networkInfo');
    end

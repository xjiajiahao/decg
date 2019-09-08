function weights = gen_weight_matrix(num_nodes, graph_style, pl)
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

    elseif strcmp(graph_style, 'line') == true
        adj_matrix = zeros(num_nodes);
        for i = 1:num_nodes
            j = mod(i, num_nodes) + 1;
            if j == 1
                continue;
            end
            adj_matrix(i, j) = 1;
            adj_matrix(j, i) = 1;
        end
        degrees = sum(adj_matrix, 2);
        weights = zeros(num_nodes);
        for i = 1 : num_nodes
            for j = 1 : num_nodes
                if (adj_matrix(i, j) == 0)
                    continue;
                end
                weights(i, j)  = 1.0 / (1.0 + max(degrees(i), degrees(j)));
            end
        end
        sum_weights = sum(weights, 2);
        weights = weights + diag(1 - sum_weights);

    elseif strcmp(graph_style, 'complete') == true
        adj_matrix = ones(num_nodes);
        adj_matrix = adj_matrix - diag(ones(num_nodes, 1));

        degrees = sum(adj_matrix, 2);

        weights = zeros(num_nodes);
        for i = 1 : num_nodes
            for j = 1 : num_nodes
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

    filename = [ROOT, 'weights_', graph_style, '_', num2str(num_nodes), '.mat'];
    save(filename, 'weights');
end


function cmat=gen_network_topology(N,pl)

% connectivity matrix
while(true)
    tempmat=rand(N);
    cmat=zeros(N);
    tempmat=(tempmat+tempmat').*(1-eye(N));
    cmat(tempmat>2*(1-pl))=1;
    cmat(tempmat<=2*(1-pl))=0;

    % connectivity
    nodeclass.conmatrix=cmat;
    flag1con=verify1con(nodeclass);
    if flag1con==1
        disp('Network Connected!');
        break;
    % else
        % error('Network Disconnected!');
        % continue;
    end
end



% verify1con: verify the 1-node-connectedness
%
% Data structure
%
%   Node properties (nodeclass)
%     conmatrix: connection matrix of node in the undirected graph, with
%                weights as the elements. node v_i and v_j are connected
%                only when they are in the transmission range of each other
%
%   Output
%     flag1con: flag1con==1 indicates at least 1-node connectedness

% Designed by LQ, 11-28-2006
% Note: for 1000 nodes and connected, processing time is around 0.31s

function flag1con=verify1con(nodeclass)
    conmatrix=nodeclass.conmatrix;
    nodenum=size(conmatrix,1);
    spanningtree=stbfs(nodeclass);
    flag1con=(nodenum==sum(spanningtree.nodeflag));
end


% stbfs: calculate spanning tree with breadth-first search
%
% Data structure
%
%   Node properties (nodeclass)
%     conmatrix: connection matrix of node in the undirected graph, with
%                weights as the elements. node v_i and v_j are connected
%                only when they are in the transmission range of each other
%
%   Spanning tree (spanningtree)
%     nodeflag: nodeflag=1 indicates the existence in the spanning tree
%     nodelabel: label of nodes in the spanning tree
%     edgelabel: label of edges in the spanning tree

% Designed by LQ, 11-28-2006
% Note: for 1000 nodes and connected, processing time is around 0.26s

function spanningtree=stbfs(nodeclass)

    conmatrix=nodeclass.conmatrix;

    nodeflag=zeros(size(conmatrix,1),1);
    nodelabel=1;
    nodeflag(nodelabel(end))=1;
    edgelabel=[];
    qset=1;

    while 1
        if isempty(qset)
            break;
        else
            parent=qset(1);
            qset(1)=[];
            temp=conmatrix(:,parent)~=0;
            temp(nodeflag==1)=0;
            todo=find(temp~=0);
            if ~isempty(todo)
                nodelabel=[nodelabel;todo];
                nodeflag(todo)=1;
                edgelabel=[edgelabel;[parent*ones(size(todo)),todo]];
                qset=[qset;todo];
            end
        end
    end

    spanningtree.nodeflag=nodeflag;
    spanningtree.nodelabel=nodelabel;
    spanningtree.edgelabel=edgelabel;
end

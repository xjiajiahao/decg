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

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

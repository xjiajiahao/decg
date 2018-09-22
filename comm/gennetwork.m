function cmat=gennetwork(N,pl)

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

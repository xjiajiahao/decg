% K = 3;
% W = [0.6 0.4 0;
%      0.4 0.5 0.1;
%      0   0.1 0.9];
I = eye(size(W, 1));
eigvalues = sort(eig(W), 'descend');
beta = max(abs(eigvalues(2)), abs(eigvalues(end)));
K = floor(sqrt((1+beta)/(1-beta)));
assert(K >= 2);

tmpW = 1/beta * W;
tmpc = 1/beta;

Told = I;
Tcurr = tmpW;

aold = 1;
acurr = tmpc;

for i = 2:K
    Tnew = 2 * tmpW * Tcurr - Told;
    anew = 2 * tmpc * acurr - aold;

    Pnew = Tnew / anew;

    Told = Tcurr;
    aold = acurr;
    Tcurr = Tnew;
    acurr = anew;
end
Pnew
new_eigvalues = sort(eig(Pnew), 'descend')
new_beta =  max(abs(new_eigvalues(2)), abs(new_eigvalues(end)))

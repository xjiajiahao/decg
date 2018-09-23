ROOT = '../data/';
linewidth = 2;
load([ROOT, 'res_DeFW_DeSAGAFW_200.mat']);
res = final_res;
load([ROOT, 'res_CenFW.mat']);
res = [res, final_res(:, 2)];

cols = [2, 3, 6];
labels = {'DeCG', 'DeGSFW', 'CenFW'};
curve_styles = {'-', '-', '-'};
ylimits = [4.5, inf];

res = [zeros(1, size(res, 2)); res];
for i = 1 : length(labels)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(res(:, 1), res(:, col)/6000, curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
legend('show');
grid on;
ylim(ylimits);

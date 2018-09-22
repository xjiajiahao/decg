ROOT = '../data/';
linewidth = 2;
load([ROOT, 'res_DeFW_DeSAGAFW_200.mat']);

cols = [2, 3];
labels = {'DeCG', 'DeGSFW'};
curve_styles = {'-', '-'};
ylimits = [4.5, inf];

final_res = [zeros(1, size(final_res, 2)); final_res];
for i = 1 : length(labels)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(final_res(:, 1), final_res(:, col)/6000, curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
legend('show');
grid on;
ylim(ylimits);

width=434;
height=420;
linewidth = 2;
ROOT = '../data/';
% OUTPUT_DIR = '/home/stephen/';
OUTPUT_DIR = ROOT;

cols = [2, 3, 6];
% cols = [2, 3, 6, 7];
labels = {'DeSCG', 'DeSGSFW', 'CenSFW', 'CenGreedy'};
curve_styles = {'-', '-', '-', '-'};
ylimits = [4.5, inf];
xlimits = [-inf, 1000];

num_samples = 5;
clear res;
for i = 1 : num_samples
    load([ROOT, 'res_DeSFW_DeSSAGAFW_0',num2str(i), '.mat']);
    if exist('res', 'var')
        res =  res + final_res;
    else
        res =  final_res;
    end
end
res = res ./ num_samples;
res_CenGreedy = 27571;
num_users = 6000;
load([ROOT, 'res_CenSFW.mat']);
res = [res, final_res(:, 2)];
res = [res, res_CenGreedy*ones(size(res, 1), 1)];
res = [[zeros(1, size(res, 2)-1), res_CenGreedy]; res];
num_gradients = res(:, 1) * num_users;

figures = {};
figure_names = {};


% the_figure = figure('position', [0, 0, width, height]);
% fig_name =[OUTPUT_DIR, 'stoch_iters', '.eps'];
% figures{end+1} = the_figure;
% figure_names{end+1} = fig_name;
% for i = 1 : length(cols)
%     col = cols(i);
%     curve_style = curve_styles{i};
%     label = labels{i};
%
%     plot(res(:, 1), res(:, col)/num_users, curve_style, 'linewidth', linewidth, 'DisplayName', label);
%     hold on;
% end
% hold on;
% xlabel('T (#iterations)');
% ylabel('objective value');
% legend('show');
% grid on;
% ylim(ylimits);
% xlim(xlimits);


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'stoch_grads', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(num_gradients, res(:, col)/num_users, curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('#gradient evaluations');
ylabel('objective value');
legend('show');
grid on;
ylim(ylimits);
xlim(xlimits*num_users);

for i = 1:length(figures)
    the_figure = figures{i};
    the_name = figure_names{i};
    saveas(the_figure, the_name, 'epsc');
end

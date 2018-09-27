width=600;
height=450;
linewidth = 2;
batch_size = 50;
num_agents = 50;
ROOT = '../data/';
% OUTPUT_DIR = '/home/stephen/';
OUTPUT_DIR = [ROOT, 'imgs/'];

cols = [2, 3]; %, 6];
% cols = [2, 3, 6, 7];
labels = {'DeSCG', 'DeSGSFW', 'CenSCG', 'CenGreedy'};
curve_styles = {'-.', '-', '-', '-'};
ylimits = [6e3, inf];
xlimits_grads = [-inf, 1e6];
xlimits_iters = [-inf, inf];
xlimits_comm = [-inf, 3.5e5];
% xlimits_comm = [-inf, inf];

figures = {};
figure_names = {};

num_samples = 5;
clear res;
for i = 1 : num_samples
    load([ROOT, 'res_DeFW_DeSAGAFW_nqp_0',num2str(i), '.mat']);
    if exist('res', 'var')
        res =  res + final_res;
    else
        res =  final_res;
    end
end
res = res ./ num_samples;
res = [zeros(1, size(res, 2)); res];
num_gradients = res(:, 1) * batch_size*num_agents;


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'nqp_stoch_grads', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(num_gradients, res(:, col), curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('#gradient evaluations');
ylabel('objective value');
legend('show');
grid on;
ylim(ylimits);
xlim(xlimits_grads);





load([ROOT, 'res_DeSFW_nqp_20.mat']);
final_res = [zeros(1, size(final_res, 2)); final_res];
res(:, 1) = final_res(:, 1);
res(:, 2) = final_res(:, 2);
res(:, 4) = final_res(:, 4);
%
% the_figure = figure('position', [0, 0, width, height]);
% fig_name =[OUTPUT_DIR, 'nqp_stoch_iters', '.eps'];
% figures{end+1} = the_figure;
% figure_names{end+1} = fig_name;
% for i = 1 : length(cols)
%     col = cols(i);
%     curve_style = curve_styles{i};
%     label = labels{i};
%
%     plot(res(:, 1), res(:, col), curve_style, 'linewidth', linewidth, 'DisplayName', label);
%     hold on;
% end
% hold on;
% xlabel('T (#iterations)');
% ylabel('objective value');
% legend('show');
% grid on;
% ylim(ylimits);
% xlim(xlimits_iters);
%
cols = [2, 3];
the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'nqp_stoch_comm', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};
    if cols(i) == 2
        comm_col = 4;
    else
        comm_col = 5;
    end
    plot(res(:, comm_col), res(:, col), curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('#doubles');
ylabel('objective value');
legend('show');
grid on;
ylim(ylimits);
xlim(xlimits_comm);

for i = 1:length(figures)
    the_figure = figures{i};
    the_name = figure_names{i};
    saveas(the_figure, the_name, 'epsc');
end

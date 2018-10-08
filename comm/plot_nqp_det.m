width=600;
height=450;
linewidth = 2;
batch_size = 50;
num_agents = 50;
ROOT = '../data/';
% OUTPUT_DIR = '/home/stephen/';
OUTPUT_DIR = [ROOT, 'imgs/'];

cols = [2, 3];
% cols = [2, 3, 6, 7];
% labels = {'DeCG', 'DeGSFW', 'CenCG', 'CenGreedy'};
labels = {'DeCG', 'DeGSFW'};
curve_styles = {'-.', '-', '-', '-'};
ylimits = [6e3, inf];
xlimits_grads = [-inf, inf];
xlimits_comm = [-inf, 3.5e5];


load([ROOT, 'res_DeCG_DeGSFW_nqp_b=1_20.mat']);

res = final_res;
res = [zeros(1, size(res, 2)); res];
num_gradients = res(:, 1) * batch_size*num_agents;

figures = {};
figure_names = {};


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'nqp_det_iters', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(res(:, 1), res(:, col), curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('T (#iterations)');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'nqp_det_grads', '.eps'];
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
xlabel('#full gradients');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);
xlim(xlimits_grads);


cols = [2, 3];
the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'nqp_det_comm', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(res(:, 4), res(:, col), curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('#doubles');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);
xlim(xlimits_comm);

for i = 1:length(figures)
    the_figure = figures{i};
    the_name = figure_names{i};
    saveas(the_figure, the_name, 'epsc');
end

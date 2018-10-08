width=600;
height=450;
linewidth = 2;
ROOT = '../data/';
% OUTPUT_DIR = '/home/stephen/';
OUTPUT_DIR = [ROOT, 'imgs/'];

cols = [2, 3]; %, 6];
% cols = [2, 3, 6, 7];
labels = {'DeCG', 'DeGSFW', 'CenCG', 'CenGreedy'};
curve_styles = {'-.', '-', '-', '-'};
ylimits = [4.5, inf];
xlimits_comm = [-inf, 7e8];
xlimits_grads = [-inf, 12e5];

res_CenGreedy = 27571;
num_users = 6000;
load([ROOT, 'res_DeCG_DeGSFW_200.mat']);
res = final_res;
load([ROOT, 'res_CenFW.mat']);
res = [res, final_res(:, 2)];
res = [res, res_CenGreedy*ones(size(res, 1), 1)];
res = [[zeros(1, size(res, 2)-1), res_CenGreedy]; res];
num_gradients = res(:, 1) * num_users;

figures = {};
figure_names = {};


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'det_iters', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(res(:, 1), res(:, col)/num_users, curve_style, 'linewidth', linewidth, 'DisplayName', label);
    hold on;
end
hold on;
xlabel('T (#iterations)');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);


the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'det_grads', '.eps'];
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
xlabel('#full gradients');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);
xlim(xlimits_grads);


cols = [2, 3];
the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'det_comm', '.eps'];
figures{end+1} = the_figure;
figure_names{end+1} = fig_name;
for i = 1 : length(cols)
    col = cols(i);
    curve_style = curve_styles{i};
    label = labels{i};

    plot(res(:, 4), res(:, col)/num_users, curve_style, 'linewidth', linewidth, 'DisplayName', label);
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

width=600;
height=450;
linewidth = 2;
ROOT = '../data/';
% OUTPUT_DIR = '/home/stephen/';
OUTPUT_DIR = [ROOT, 'imgs/'];

cols = [2, 3]; %, 6];
% cols = [2, 3, 6, 7];
labels = {'DeSCG', 'DeSGSFW', 'CenSCG', 'CenGreedy'};
curve_styles = {'-.', '-', '-', '-'};
ylimits = [4.455, inf];
xlimits_grads = [-inf, 1000];
xlimits_iters = [-inf, 14];

figures = {};
figure_names = {};

num_samples = 1;
clear res;
for i = 1 : num_samples
    load([ROOT, 'res_DeSCG_DeSGSFW_0',num2str(i), '.mat']);
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
xlabel('#stochastic gradients');
ylabel('objective value');
legend('show', 'location', 'southeast');
grid on;
ylim(ylimits);
xlim(xlimits_grads*num_users);





load([ROOT, 'res_DeSCG_DeSGSFW_14.mat']);
final_res = [zeros(1, size(final_res, 2)); final_res];
res(:, 1) = final_res(:, 1);
res(:, 2) = final_res(:, 2);
res(:, 4) = final_res(:, 4);
load([ROOT, 'res_CenSFW_14.mat']);
final_res = [zeros(1, size(final_res, 2)); final_res];
res(:, 6) = final_res(:, 2);

the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'stoch_iters', '.eps'];
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
xlim(xlimits_iters);

cols = [2, 3];
the_figure = figure('position', [0, 0, width, height]);
fig_name =[OUTPUT_DIR, 'stoch_comm', '.eps'];
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
xlim([0, 48708352]);

for i = 1:length(figures)
    the_figure = figures{i};
    the_name = figure_names{i};
    saveas(the_figure, the_name, 'epsc');
end

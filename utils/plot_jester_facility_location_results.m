width=600;
height=450;
line_width = 2;
marker_size = 10;
font_size = 20;

plot(res_CenSCG(:, 3), res_CenSCG(:, 5), 'DisplayName', 'SCG', 'LineWidth', line_width); hold on;
plot(res_CenSFW(:, 3), res_CenSFW(:, 5), 'DisplayName', 'SFW', 'LineWidth', line_width); hold on;
plot(res_CenSTORM(:, 3), res_CenSTORM(:, 5), 'DisplayName', 'STORM', 'LineWidth', line_width); hold on;

% plot(avg_res_CenSCG(:, 3), avg_res_CenSCG(:, 5), 'DisplayName', 'SCG', 'LineWidth', line_width); hold on;
% plot(avg_res_CenSFW(:, 3), avg_res_CenSFW(:, 5), 'DisplayName', 'SFW', 'LineWidth', line_width); hold on;
% plot(avg_res_CenSTORM(:, 3), avg_res_CenSTORM(:, 5), 'DisplayName', 'STORM', 'LineWidth', line_width); hold on;

legend('show', 'Location', 'southeast');
xlim([0, 1.2e6]);
ylim([16.65, 16.665]);
set(gca, 'FontName', 'Times New Roman');
set (gca, 'FontSize', font_size);
xlabel('#function evaluations');
ylabel('objective value');
grid on;
title('facility location, MovieLens1M');

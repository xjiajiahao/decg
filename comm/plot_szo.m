width=600;
height=450;
linewidth = 2;
font_size = 16;

plot(tmpx(:, 1), tmpx(:, 2), 'linewidth', linewidth);

grid on;

set(gca, 'FontName', 'Times New Roman');
set (gca, 'FontSize', font_size);

% legend('show');
title('facility location, MovieLens1M')
xlabel('#function evaluations for SCG++');
ylabel('#function evaluations for SCG');
% ylim([4.48, inf]);

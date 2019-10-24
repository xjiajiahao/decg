clear;
width=600;
height=450;
line_width = 2;
marker_size = 10;
font_size = 20;
title_font_size = 19;
text_size = 20;
avg_nnz = 1.655975165562914e+02;

res_CenSCG = [
0.0    0.0          0.0  0.0  0.0      329.999
25.0    6.39141   1000.0  0.0  1.43098    4.00843
50.0   18.5452    2000.0  0.0  2.34254    3.20572
75.0   30.6701    3000.0  0.0  2.94551    2.16805
100.0   42.9062    4000.0  0.0  3.36212    1.27709
125.0   55.1457    5000.0  0.0  3.66241    0.759503
150.0   67.3002    6000.0  0.0  3.88713    0.454925
175.0   79.5846    7000.0  0.0  4.061      0.273802
200.0   91.9255    8000.0  0.0  4.19916    0.177979
225.0  104.318     9000.0  0.0  4.31092    0.125754
250.0  116.657    10000.0  0.0  4.40267    0.0876921
275.0  128.893    11000.0  0.0  4.47908    0.0623367
300.0  141.292    12000.0  0.0  4.54341    0.0453236
325.0  153.572    13000.0  0.0  4.59805    0.0342047
350.0  165.832    14000.0  0.0  4.64484    0.0258394
375.0  178.061    15000.0  0.0  4.68505    0.0196359
400.0  190.266    16000.0  0.0  4.71978    0.0160921
425.0  202.594    17000.0  0.0  4.74994    0.0123953
450.0  214.846    18000.0  0.0  4.77625    0.0101562
475.0  227.321    19000.0  0.0  4.79932    0.00808516
500.0  239.911    20000.0  0.0  4.81962    0.00636926
];

res_CenSTORM = [
  0.0    0.0          0.0  0.0  0.0      51.0804
 25.0    7.392    41000.0  0.0  1.43079   0.761514
 50.0   14.77     82000.0  0.0  2.34251   0.179494
 75.0   22.116   123000.0  0.0  2.9451    0.0726884
100.0   29.538   164000.0  0.0  3.36184   0.0393931
125.0   36.9039  205000.0  0.0  3.6626    0.0225862
150.0   44.3191  246000.0  0.0  3.88834   0.0147348
175.0   51.6441  287000.0  0.0  4.06333   0.010036
200.0   58.9547  328000.0  0.0  4.20207   0.00731571
225.0   66.2913  369000.0  0.0  4.31425   0.00538869
250.0   73.6261  410000.0  0.0  4.40652   0.00415486
275.0   80.959   451000.0  0.0  4.48335   0.0031565
300.0   88.3801  492000.0  0.0  4.54796   0.00256338
325.0   95.7221  533000.0  0.0  4.60276   0.00201425
350.0  103.058   574000.0  0.0  4.64952   0.00170651
375.0  110.399   615000.0  0.0  4.68967   0.00142368
400.0  117.751   656000.0  0.0  4.7243    0.00119537
425.0  125.05    697000.0  0.0  4.75437   0.00114314
450.0  132.379   738000.0  0.0  4.78053   0.00107179
475.0  139.733   779000.0  0.0  4.80343   0.000940341
500.0  147.015   820000.0  0.0  4.82351   0.000893243
];

semilogy(res_CenSCG(:, 1), res_CenSCG(:, 6), '-*', 'DisplayName', 'SCG', 'LineWidth', 2);
hold on;
semilogy(res_CenSTORM(:, 1), res_CenSTORM(:, 6), '-d', 'DisplayName', 'SCG++', 'LineWidth', 2);
hold on;

set(gca, 'FontName', 'Times New Roman');
set (gca, 'FontSize', font_size);
xlabel('#iterations');
ylabel('gradient approximation error');
grid on;
title('facility location, MovieLens1M', 'FontSize', title_font_size);
legend('show', 'Location', 'northeast');

file_name = ['../data/result_variance_facility_location.eps'];
saveas(gcf, file_name, 'epsc');

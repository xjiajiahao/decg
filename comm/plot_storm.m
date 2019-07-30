clear;
ROOT = '../data/';

filename = [ROOT, 'storm_results/', 'movie_1M_STORM_200_batch_size_128_sample_10_trail_50.mat'];
load(filename);
plot(res_CenSTORM(:, 3), res_CenSTORM(:, 5), 'DisplayName', 'STORM'); hold on;

filename = [ROOT, 'storm_results/', 'movie_1M_SCG_200_batch_size_128_sample_10.mat'];
load(filename);
plot(res_CenSCG(:, 3), res_CenSCG(:, 5), 'DisplayName', 'SCG'); hold on;
grid on;
legend('show');
xlabel('#SZO');
ylabel('objective value');

clear;
ROOT = '../data/';

% STORM
filename = [ROOT, 'storm_results/', 'movie_1M_STORM_200_batch_size_128_sample_10_trial_50.mat'];
load(filename);
plot(res_CenSTORM(:, 3), res_CenSTORM(:, 5), 'DisplayName', 'STORM'); hold on;

% SCG
filename = [ROOT, 'storm_results/', 'movie_1M_SCG_200_batch_size_128_sample_10.mat'];
load(filename);
plot(res_CenSCG(:, 3), res_CenSCG(:, 5), 'DisplayName', 'SCG'); hold on;

% PSGD
filename = [ROOT, 'storm_results/', 'movie_1M_PSGD_200_batch_size_128_sample_10_trial_1.mat'];
load(filename);
plot(res_CenPSGD(2:end, 3), res_CenPSGD(2:end, 5), 'DisplayName', 'SGA'); hold on;

grid on;
legend('show');
xlabel('#SZO');
ylabel('objective value');
ylim([4.48, inf]);

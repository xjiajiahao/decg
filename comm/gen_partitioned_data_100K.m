function [user_ratings_cell_arr, num_movies, num_users, num_agents] = gen_partitioned_data_100K(num_agents)
%% initialization
ROOT = '../data/';
% ROOT = './';
num_movies = 1682;
num_users = 943;

load([ROOT, 'Movies_100K.mat']);
rng(1); % For reproducibility
% indices = randperm(num_users);
indices = [1:num_users];

num_users_per_agent = floor(num_users / num_agents);
user_ratings_cell_arr = cell(1, num_agents);
% compute the true num_users
num_users = num_users_per_agent * num_agents;

user_ratings_mat = user_ratings_matrix(:, indices(1:num_users));

for i = 1 : num_agents
    tmp_idx = indices((i-1)*num_users_per_agent + 1: i*num_users_per_agent);
    user_ratings_cell_arr{i} = user_ratings_cell(tmp_idx);
end

filename = [ROOT, 'Movies_100K_', num2str(num_agents), '_agents.mat'];
save(filename, 'user_ratings_cell_arr', 'user_ratings_mat', 'num_movies', 'num_users', 'num_agents');

end
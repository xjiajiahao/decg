function gen_partitioned_data_1M(num_agents)
% data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user

%% initialization
ROOT = '../data/';
num_movies = 3883;
num_users = 6040;

load([ROOT, 'Movies_1M.mat']);
rng(1); % For reproducibility
% indices = randperm(num_users);
indices = [1:num_users];

num_users_per_agent = floor(num_users / num_agents);
user_ratings_cell_arr = cell(1, num_agents);
% compute the true num_users
num_users = num_users_per_agent * num_agents;

user_ratings_matrix = user_ratings_matrix(:, indices(1:num_users));

for i = 1 : num_agents
    tmp_idx = indices((i-1)*num_users_per_agent + 1: i*num_users_per_agent);
    user_ratings_cell_arr{i} = user_ratings_cell(tmp_idx);
end

filename = [ROOT, 'Movies_1M_', num2str(num_agents), '_agents.mat'];
save(filename, 'user_ratings_cell_arr', 'user_ratings_matrix', 'num_movies', 'num_users', 'num_agents');

end

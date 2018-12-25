function gen_partitioned_data_100K(num_agents)
% data_cell[i][j] is a n_j-by-2 matrix representing the ratings of agent i's jth user

%% initialization
ROOT = '../data/';
num_movies = int32(1682);
num_users = 943;

load([ROOT, 'Movies_100K.mat']);
if size(user_ratings_cell, 1) == 1
    user_ratings_cell = user_ratings_cell';
end
rng(1); % For reproducibility
% indices = randperm(num_users);
indices = [1:num_users];

num_users_per_agent = floor(num_users / num_agents);
user_ratings_cell_arr = cell(1, num_agents);
% compute the true num_users
num_users = num_users_per_agent * num_agents;

% save the global dataset
data_cell = user_ratings_cell(1:num_users);
filename = [ROOT, 'Movies_100K_global_', num2str(num_agents), '_agents.mat'];
save(filename, 'data_cell', 'num_movies');

for i = 1 : num_agents
    tmp_idx = indices((i-1)*num_users_per_agent + 1: i*num_users_per_agent);
    data_cell = user_ratings_cell(tmp_idx);
    filename = [ROOT, 'Movies_100K_rank=', num2str(int32(i-1), '%02d'), '_' num2str(num_agents), '_agents.mat'];
    save(filename, 'data_cell', 'num_movies');
end


end

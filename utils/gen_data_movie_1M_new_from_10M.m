% function gen_data_movielens_10M()
%% initialization
ROOT = '../data/';
OUTPUT_DIR = '../data/';
max_num_movies = 4000;
max_num_users = 6000;
threshold = 46;
new_user_ratings_cell = cell(1, max_num_users);

filename = [OUTPUT_DIR, 'Movies_10M.mat'];  % user_ratings_cell, user_ratings_matrix
load(filename);

truncated_user_ratings_matrix = user_ratings_matrix(1:max_num_movies, :);

num_movies_per_user = sum(truncated_user_ratings_matrix~=0, 1);
tmp_idx = [1:size(user_ratings_matrix, 2)];
selected_user_idx = tmp_idx(num_movies_per_user >= threshold);
selected_user_idx = selected_user_idx(1:max_num_users);
% thresholded_num_movies_per_user = num_movies_per_user(num_movies_per_user >= threshold);
for i = 1 : max_num_users
    tmp_mat = user_ratings_cell{selected_user_idx(i)};
    tmp_idx = [1:size(tmp_mat, 2)];
    valid_idx = tmp_idx(tmp_mat(1, :) <= max_num_movies);
    new_user_ratings_cell{i} = tmp_mat(:, valid_idx);
end

user_ratings_cell = new_user_ratings_cell;
user_ratings_matrix = truncated_user_ratings_matrix(:, selected_user_idx);
assert(nnz(user_ratings_matrix) >= 1e6);

filename = [OUTPUT_DIR, 'Movies_1M_new', '.mat'];
save(filename, 'user_ratings_cell', 'user_ratings_matrix');

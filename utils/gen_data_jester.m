function gen_data_jester2plus()
ROOT = '../data/jester/';
OUTPUT_DIR = '../data/';

% load the excel table
file_name = [ROOT, 'jesterfinal151cols.xls'];
user_ratings_matrix = transpose(table2array(readtable(file_name, 'ReadVariableNames', false)));  % (num_movies+1)-by-num_users
% preprocessing
user_ratings_matrix = user_ratings_matrix(2:end, :);  % remove the first row, which is the number of non-zero ratings of each user
num_movies = size(user_ratings_matrix, 1);
num_users = size(user_ratings_matrix, 2);

% rescale and sparsify the matrix
user_ratings_matrix = user_ratings_matrix + 10.0;   % rescale from [-10, 10] to [0, 20]
user_ratings_matrix(user_ratings_matrix == 99.0 + 10.0) = 0.0;
user_ratings_matrix = sparse(user_ratings_matrix);

% for each user, sort the movie ratings from high to low
user_ratings_cell = cell(1, num_users);
for i = 1 : num_users
    [tmp_indices, ~, tmp_values] = find(user_ratings_matrix(:, i));
    tmp_matrix = [tmp_indices, tmp_values];
    user_ratings_cell{i} = sortrows(tmp_matrix, 2, 'descend');
    user_ratings_cell{i} = user_ratings_cell{i}';
end

filename = [OUTPUT_DIR, 'Jester', '.mat'];
save(filename, 'user_ratings_cell', 'user_ratings_matrix');

end  % end of the function

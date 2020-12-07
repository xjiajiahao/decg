% function gen_data_movielens_10M()
%% initialization
ROOT = '../data/ml-10M100K/';
OUTPUT_DIR = '../data/';
num_movies = 10681;
num_users = 71567;
num_genres = 18;

% create a genres_dict mapping from genre name to number
% genres_dict = py.dict(pyargs( ...
%     'Action', 1, ...
% 	'Adventure', 2, ...
% 	'Animation', 3, ...
% 	'Children', 4, ...
% 	'Comedy', 5, ...
% 	'Crime', 6, ...
% 	'Documentary', 7, ...
% 	'Drama', 8, ...
% 	'Fantasy', 9, ...
% 	'Film-Noir', 10, ...
% 	'Horror', 11, ...
% 	'Musical', 12, ...
% 	'Mystery', 13, ...
% 	'Romance', 14, ...
% 	'Sci-Fi', 15, ...
% 	'Thriller', 16, ...
% 	'War', 17, ...
% 	'Western', 18));
% create a matrix that records the movie-genres information
% movie_genre = zeros(num_movies, num_genres);

% create a movie_id_dict mapping from old movie id to new movie id
movie_id_dict = py.dict();

% allocate a ratings matrix for each user
user_ratings_cell = cell(1, num_users);

%% Process movies.dat file
f_movies = fopen([ROOT, 'movies.dat'], 'r');
counter = 0;

while ~feof(f_movies)
    % Step 1: read one line
    counter = counter + 1;
    line = fgetl(f_movies); % read line by line
    fields = strsplit(line, '::');

    % Step 2: remap the movie_id
    old_movie_id_str = fields{1};
    update(movie_id_dict, py.dict(pyargs(old_movie_id_str, counter)));

    % Step 3: record current movie's genres
    % cur_genres = strsplit(fields{end}, '|');
    % cur_genres_num = size(cur_genres, 2);
    % for i = 1 : cur_genres_num
    %     tmp_genre = genres_dict{cur_genres{i}};
    %     movie_genre(counter, tmp_genre) = 1;
    % end
end

% movie_genre = sparse(movie_genre);

%% Process ratings.dat file
tmp_arr = zeros(num_movies, 2);
user_count = 0;
f_ratings = fopen([ROOT, 'ratings.dat'], 'r');
previous_user = 0;
while ~feof(f_ratings)
    % Step 1: read one line
    line = fgetl(f_ratings); %# read line by line
    fields = strsplit(line, '::');
    user_id = str2num(fields{1});
    old_movie_id_str = fields{2};
    score = str2num(fields{3});
    % Step 2: build user_ratings_cell
    if user_id ~= previous_user
        % assert(user_id == previous_user + 1, 'user_id: %d\n', user_id);
        if user_count > 0
            user_ratings_cell{user_count} = tmp_arr(1:count_movies_of_curr_user, :);
        end
        user_count = user_count + 1;
        previous_user = user_id;
        count_movies_of_curr_user = 1;
    else
         count_movies_of_curr_user = count_movies_of_curr_user + 1;
    end
    new_movie_id = movie_id_dict{old_movie_id_str};
    tmp_arr(count_movies_of_curr_user, :) = [new_movie_id, score];
    % user_ratings_cell{user_id} = [user_ratings_cell{user_id}; new_movie_id, score];
end
user_ratings_cell{user_count} = tmp_arr(1:count_movies_of_curr_user, :);
user_ratings_cell = user_ratings_cell(1:user_count);

num_users = user_count;
user_ratings_matrix = zeros(num_movies, num_users);
% sort the movie ratings from high to low
for i = 1 : num_users
    tmp_matrix = user_ratings_cell{i};
    for j = 1 : size(tmp_matrix, 1)
        user_ratings_matrix(tmp_matrix(j, 1), i) = tmp_matrix(j, 2);
    end
    user_ratings_cell{i} = sortrows(user_ratings_cell{i}, 2, 'descend');
    user_ratings_cell{i} = user_ratings_cell{i}';
end

user_ratings_matrix = sparse(user_ratings_matrix);

filename = [OUTPUT_DIR, 'Movies_10M', '.mat'];
save(filename, 'user_ratings_cell', 'user_ratings_matrix');

This folder contains Matlab scripts to generate weight matrices and partitioned data sets.

* To generate a weight matrix, call the function `gen_weight_matrix(num_nodes, graph_style, pl)`, where num_nodes is the number of nodes in the network, graph_style can be 'complete' for complete graph, 'line' for line graph, 'er' for Erdos-Renyi random graph, the last argument pl is the probability that a potential edge appears in the network, and 0 < pl <= 1.

* To generate partitioned data sets for the NQP problem, call the function `[data_cell, A, dim, u, b] = generate_nqp_data(dim, num_agents, batch_size, num_constraints, magnitude_data)`. See the file `generate_nqp_data.m` for detailed descriptions of arguments.

* To generate partitioned data sets for the movie recommendation, call the function `[user_ratings_cell_arr, num_movies, num_users, num_agents] = gen_partitioned_data_movie_100K(num_agents)`. See the file `gen_partitioned_data_movie_100K.m` for detailed descriptions of arguments.

Note that there are two datasets for the movie recommendation application, one contains 100K entries of rating, and another contains 1M. The file `gen_partitioned_data_movie_1M.m` is for the larger data set. Both data sets can be downloaded from https://grouplens.org/datasets/movielens/.

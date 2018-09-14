### Usage
Launch julia, and then type `include("movie_main.jl");` to test algorithms.

### Data
The data is stored in a cell array `user_ratings_cell_arr`. Each entry of the cell array is again a cell array containing `num_users_per_agent` cells. And each cell (which is a num_watched_movies) represents each user's ratings for the movies he/she has watched.

### Directory Structure
movie_main.jl -- the main function
comm.jl -- handy functions for loading data and plotting results
facility.jl -- functions for computing the objective function (called "facility location") and its gradient
algorithms/ -- containing files that define different algoritms (centralized FW, decentralized FW proposed by Mokhtari, and decentralized SAGA-FW proposed by us).
data/ -- containing data files and scripts to generate data/network

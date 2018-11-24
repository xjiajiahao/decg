using MAT, LinearAlgebra
using Distributed
@everywhere using MathProgBase, Clp

const e  = exp(1);

# load data set, which has been randomly and equally partitioned
function load_movie_partitioned_data(num_agents, size="1M")
    ROOT = "./data/";
    # file = matopen("data/Movies20M.mat");
    filename = "$(ROOT)Movies_$(size)_$(num_agents)_agents.mat";
    file = matopen(filename);
    user_ratings_cell_arr = read(file, "user_ratings_cell_arr"); # @NOTE we would use the cell arrary data structure where for each user, the ratings are sorted from high to low
    user_ratings_mat = read(file, "user_ratings_mat");
    num_movies = round(Int, read(file, "num_movies"));
    num_users = round(Int, read(file, "num_users"));
    close(file);

    return (user_ratings_cell_arr, user_ratings_mat, num_movies, num_users)
end

function load_nqp_partitioned_data(num_agents)
    ROOT = "./data/";
    filename = "$(ROOT)NQP_$(num_agents)_agents.mat";
    file = matopen(filename);
    data_cell = read(file, "data_cell"); # data_cell is a 1-by-num_agents cell, each element of data_cell is a 1-by-batch_size cell which contains #batch_size matrices H_i of size dim-by-dim
    A = read(file, "A");
    dim = round(Int, read(file, "dim"));
    u = read(file, "u");
    u = dropdims(u; dims=2);
    b = read(file, "b");
    b = dropdims(b; dims=2);
    close(file);
    return (data_cell, A, dim, u, b)
end

function load_network(network_type="er", num_agents=50)
    ROOT = "./data/";
    filename = "$(ROOT)weights_$(network_type)_$(num_agents).mat";
    file = matopen(filename);
    weights = read(file, "weights");
    close(file);
    # find the first and second largest (in magnitude) eigenvalues
    dim = size(weights, 1);
    eigvalues = (LinearAlgebra.eigen(weights)).values;
    if abs(eigvalues[dim] - 1.0) > 1e-8
        error("the largest eigenvalue of the weight matrix is $(eigvalues[dim]), but it must be 1");
    end
    beta = max(abs(eigvalues[1]), abs(eigvalues[dim - 1]));
    if beta < 1e-8
        beta = 0.0;
    end
    return (weights, beta);
end

# Linear Maximization Oracle (LMO):
# find min c^T x, s.t. a^T x < k, 0 <= x <= d, where x \in R^n, a is a m-by-n matrix, where m denotes the number of constraints
function generate_linear_prog_function(d::Vector{Float64}, a::Array{Float64, 2}, k)
    function linear_prog(x0) # quadratic programming: min_x ||x - x0 ||/2
        sol = linprog(-x0, a, '<', k, 0.0, d, ClpSolver());
        if sol.status == :Optimal
            return sol.sol;
        end
        error("No solution was found.");
    end
    return linear_prog;
end

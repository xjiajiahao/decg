using MAT, Base.Iterators.cycle
# using PyPlot
# PyPlot.matplotlib[:rcParams]["figure.autolayout"] = "True"
# srand(0)
# PyPlot.matplotlib[:rc]("font", family="serif", serif="Times New Roman", size=16)
# global_line_width = 3

# load data set, which has been randomly and equally partitioned
function load_movie_partitioned_data(num_agents)
    ROOT = "./data/";
    # file = matopen("data/Movies20M.mat");
    filename = "$(ROOT)Movies_$(num_agents)_agents.mat";
    file = matopen(filename);
    user_ratings_cell_arr = read(file, "user_ratings_cell_arr"); # @NOTE we would use the cell arrary data structure where for each user, the ratings are sorted from high to low
    num_movies = round(Int, read(file, "num_movies"));
    num_users = round(Int, read(file, "num_users"));
    close(file);

    # return (user_ratings_cell_arr, num_movies, num_users)
    return (user_ratings_cell_arr, num_movies, num_users)
end

# Linear Maximization Oracle (LMO):
# find min c^T x, s.t. a^T x < k, 0 <= x <= d, where x \in R^n, a is a 1-by-n row vector
function generate_linear_prog_function(d, a, k)
    function linear_prog(x0) # quadratic programming: min_x ||x - x0 ||/2
        sol = linprog(-x0, a, '<', k, 0.0, d, ClpSolver());
        if sol.status == :Optimal
            return sol.sol;
        end
        error("No solution was found.");
    end
    return linear_prog;
end


function generate_figure_obj()
    res_list = [];
    push!(res_list, res_CenFW);
    labels = ["CenFW"];
    # labels = ["Online Gradient Ascent", "Coin Betting", "Meta Frank-Wolfe"];
    marker = cycle((".", ",", "+", "_", "o", "x", "*"))
    linecycler = cycle(("-", "--", "-.", ":"))
    for zipped in zip(res_list, labels, marker, linecycler)
        res, label, marker_iter, line_iter = zipped;
        plot(res[:, 1], res[:, 3], label=label, linestyle=line_iter, linewidth=global_line_width);
    end
    # ticklabel_format(style="sci", axis="y", scilimits=(0, 0));
    legend(loc="best");
    xlabel("Iteration index");
    ylabel("objective value");
    grid();
end

function [] = gen_partitioned_yahoo(num_agents, max_volumn, num_players, % @TODO)
ROOT = './data/';

filename = [ROOT, 'yahoo_raw.mat'];
load(filename);
rng(1);

% preprocessing the data
avg_bidding_prices = bidding_prices ./ sum(graph, 2);
influence_probabilities = graph ./ sum(graph, 1);

[num_keywords, num_customers] = size(influence_probabilities);
num_customers_per_agent = floor(num_customers / num_agents);
num_customers = num_customers_per_agent * num_agents;

data_cell_arr = cell(1, num_agents);

for i = 1 : num_agents
    tmp_idx = indices((i-1)*num_users_per_agent + 1: i*num_users_per_agent);
    data_cell_arr{i} = influence_probabilities(:, tmp_idx);
end

filename = [ROOT, 'yahoo_', num2str(num_agents), '_agents.mat'];
save(filename, 'influence_probabilities', 'avg_bidding_prices');

end

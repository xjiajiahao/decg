num_keywords = 1e3;
num_customers = 10475;
ROOT = './data/';

graph = zeros(num_keywords, num_customers);
bidding_prices = zeros(num_keywords);

f_bidding = fopen([ROOT, 'ydata-ysm-advertiser-bids-v1_0.txt'], 'r');

while ~feof(f_bidding)
    % step 1: read line by line
    line = fgetl(f_bidding);
    fields = strsplit(line, {' ', '	'});

    % step 2: update the graph
    keyword_id = str2int(fields{3});
    customer_id = str2int(fields{4}) + 1;
    price = str2double(fields{5});
    graph(keyword_id, customer_id) = graph(keyword_id, customer_id) + 1;
    bidding_prices(keyword_id) = bidding_prices(keyword_id) + price;
end

filename = [ROOT, 'yahoo_raw.mat'];
save(filename, 'bidding_prices', 'graph', 'num_keywords', 'num_customers');

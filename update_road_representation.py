from torch_geometric.nn import MessagePassing
from torch_geometric.utils.scatter import scatter

class MeanAggregator(MessagePassing):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def aggregate(self, x_j, edge_index):
        row, col = edge_index
        aggr_out = scatter(x_j, col, dim=0, reduce='mean')  # [num_nodes, feature_size]
        return aggr_out

class WeightedMeanAggregator(MessagePassing):
    def __init__(self):
        super(WeightedMeanAggregator, self).__init__()

    def forward(self, x, edge_index, weights):
        return self.propagate(edge_index, x=x, weights=weights)

    def message(self, x_j, weights):
        return x_j * weights.view(-1, 1)

    def aggregate(self, x_j, edge_index):
        row, col = edge_index
        aggr_out = scatter(x_j, col, dim=0, reduce='sum')  # [num_nodes, feature_size]
        return aggr_out
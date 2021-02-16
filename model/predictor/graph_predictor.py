import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class GraphPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class IdentityGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, x


class LinearGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.W(x), x


class MLPGraphPredictor(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super(MLPGraphPredictor, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = nn.Linear(dim_node_features, hidden_units)
        self.out = nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.out(F.relu(self.fc_global(x)))

class CGMMGraphPredictor(nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        dim_features = dim_features[1]

        hidden_units = config['hidden_units']

        self.fc_global = torch.nn.Linear(dim_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        extra = data[1]
        x = torch.reshape(extra.g_outs.squeeze().float(), (extra.g_outs.shape[0], -1))
        return self.out(F.relu(self.fc_global(x)))


# Used for CL+GRL experiments
class GraphSAGEGraphPredictor(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super(GraphSAGEGraphPredictor, self).__init__()

        hidden_units = config['hidden_units']

        # For graph classification
        self.fc1 = nn.Linear(dim_node_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        node_emb, batch = data.node_emb, data.batch
        x = global_max_pool(node_emb, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, node_emb, data.edge_index

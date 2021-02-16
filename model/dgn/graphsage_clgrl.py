import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SAGEConv


class GraphSAGESuperpixels(nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['hidden_units']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        self.predictor = predictor_class(dim_node_features=dim_embedding*num_layers,
                                         dim_edge_features=dim_edge_features,
                                         dim_target=dim_target, config=config)

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = torch.cat((x, pos), dim=1)

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        data.node_emb = x

        return self.predictor(data)



# Similar to https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/ppa/conv.py
class OGBGPPAConv(MessagePassing):
    def __init__(self, hidden_units, edge_encoder):
        super(OGBGPPAConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_units, hidden_units), torch.nn.ReLU())
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = edge_encoder

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps)*x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GraphSAGEOGBGPPA(nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['hidden_units']
        self.aggregation = config['aggregation']  # can be mean or sum

        edge_encoder = torch.nn.Linear(dim_edge_features, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding

            conv = OGBGPPAConv(dim_embedding, edge_encoder)
            # Overwrite aggregation method (default is set to mean)
            conv.aggr = self.aggregation

            self.layers.append(conv)

        self.predictor = predictor_class(dim_node_features=dim_embedding*num_layers,
                                         dim_edge_features=dim_edge_features,
                                         dim_target=dim_target, config=config)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        data.node_emb = x        

        return self.predictor(data)

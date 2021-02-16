import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class SuperpixelsBaseline(nn.Module):
    """
    Simple MLP that can be used to test the library
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super().__init__()

        n_layers = config['num_layers']
        hidden_units = config['hidden_units']

        feat_mlp_modules = [
            nn.Linear(dim_node_features, hidden_units, bias=True),
            nn.ReLU()
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_units, hidden_units, bias=True))
            feat_mlp_modules.append(nn.ReLU())
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        # Readout part
        L = 2
        self.L = L
        list_FC_layers = [ nn.Linear( hidden_units//2**l, hidden_units//2**(l+1), bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( hidden_units//2**L, dim_target, bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)


    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        x = torch.cat((x, pos), dim=1)

        h = self.feat_mlp(x)
        hg = global_mean_pool(h, batch)

        out = hg
        for l in range(self.L):
            out = self.FC_layers[l](out)
            out = F.relu(out)
        out = self.FC_layers[self.L](out)

        return out, hg


class OGBGBaseline(nn.Module):
    """
    Simple MLP that can be used to test the library
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super().__init__()

        n_layers = config['num_layers']
        hidden_units = config['hidden_units']

        feat_mlp_modules = [
            nn.Linear(dim_edge_features, hidden_units, bias=True),
            nn.ReLU()
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_units, hidden_units, bias=True))
            feat_mlp_modules.append(nn.ReLU())
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        # Readout part
        L = 2
        self.L = L
        list_FC_layers = [ nn.Linear( hidden_units//2**l, hidden_units//2**(l+1), bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( hidden_units//2**L, dim_target, bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)


    def forward(self, data):
        edge_attr, edge_index, batch = data.edge_attr, data.edge_index, data.batch

        edge_batch = batch[edge_index[0]]

        h = self.feat_mlp(edge_attr)
        hg = global_mean_pool(h, edge_batch)

        out = hg
        for l in range(self.L):
            out = self.FC_layers[l](out)
            out = F.relu(out)
        out = self.FC_layers[self.L](out)

        return out, h

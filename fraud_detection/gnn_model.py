import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, 1)  # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = torch.mean(x, dim=0)  # Aggregate node features
        x = self.fc(x)
        return x

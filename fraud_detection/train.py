from gnn_model import GNNModel
import torch
from torch_geometric.data import DataLoader
import numpy as np

# Dummy dataset for example
class DummyDataset:
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        x = torch.rand((10, 16))  # 10 nodes with 16 features
        edge_index = torch.randint(0, 10, (2, 20))  # Random edges
        y = torch.tensor([1.0])  # Dummy target
        return Data(x=x, edge_index=edge_index, y=y)

def train_gnn_model():
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    model.train()

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(10):  # Number of epochs
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'gnn_model.pth')

if __name__ == "__main__":
    train_gnn_model()
s

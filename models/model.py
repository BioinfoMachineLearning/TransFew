import torch.nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch.nn import Linear, Sigmoid


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.hid = 1000
        self.in_head = 1
        self.out_head = 1

        self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.6, edge_dim=1)
        self.conv2 = GATv2Conv(self.hid * self.in_head, num_classes, heads=self.out_head,
                               concat=False, dropout=0.6, edge_dim=1)

        self.sigmoid = Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        x = self.sigmoid(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
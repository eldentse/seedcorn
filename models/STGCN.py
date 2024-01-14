import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


class STGCN(torch.nn.Module):
    def __init__(self, num_nodes, node_features, num_classes):
        super(STGCN, self).__init__()
        self.recurrent = STConv(num_nodes, node_features, hidden_channels=32, out_channels=32, kernel_size=3, K=1)
        self.recurrent1 = STConv(num_nodes, 32, hidden_channels=32, out_channels=32, kernel_size=3, K=1)
        self.linear = torch.nn.Linear(13, num_classes)
        self.drop_out = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight):
        x = x.unsqueeze(0)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.drop_out(h)
        h = self.recurrent1(h, edge_index, edge_weight)
        h = h.mean(3).mean(2)
        h = self.linear(h)
        return h[0]
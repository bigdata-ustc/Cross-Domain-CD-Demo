import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation  # 激励函数

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_feats, out_feats)
        self.apply_mod = NodeApplyModule(out_feats, out_feats, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 执行图卷积
        x = self.conv(x, edge_index)
        # 应用节点更新操作
        x = self.apply_mod(x)
        return x

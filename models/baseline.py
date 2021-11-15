import dgl
import torch
from torch import nn
from dgl.nn.pytorch import GraphConv, GATConv

class EdgePredictor(nn.Module):
    def __init__(self, edge_feature_size, node_feature_size, hidden_size, num_classes):
        super(EdgePredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(in_features= edge_feature_size + (2 * node_feature_size), out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def apply_edges(self, edges):
        h = torch.cat([edges.data['feat'], edges.src['hn'], edges.dst['hn']], dim=1)

        return { 'scores': self.predictor(h) }

    def forward(self, block, n_feats):
        with block.local_scope():
            block.ndata['hn'] = n_feats
            block.apply_edges(self.apply_edges)

            return block.edata['scores']


class GCN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, num_classes):
        super(GCN, self).__init__()
        
        self.feature = nn.Sequential(
            GraphConv(in_feats=node_feature_size, out_feats=16, activation=nn.ReLU(), allow_zero_in_degree=True),
            nn.Dropout(),
            nn.BatchNorm1d(num_features=16),
            GraphConv(in_feats=16, out_feats=64, allow_zero_in_degree=True),
        )

        self.classifier = EdgePredictor(edge_feature_size=edge_feature_size, node_feature_size=64, hidden_size=16, num_classes=num_classes)

    def forward(self, blocks, n_feats):
        x = n_feats
        index = 0

        for layer in self.feature:
            if isinstance(layer, GraphConv):
                if isinstance(blocks, dgl.DGLGraph):
                    x = layer(blocks, x)
                else:
                    x = layer(blocks[index], x)
                    index += 1
            else:
                x = layer(x)

        if isinstance(blocks, dgl.DGLGraph):
            return self.classifier(blocks, x)
        else:
            return self.classifier(blocks[-1], x)


class GAT(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, num_classes):
        super(GAT, self).__init__()
        
        self.feature = nn.Sequential(
            GATConv(in_feats=node_feature_size, out_feats=16, num_heads=8, activation=nn.ReLU(), allow_zero_in_degree=True),
            nn.Dropout(),
            GATConv(in_feats=16*8, out_feats=32, num_heads=1, activation=nn.ReLU(), allow_zero_in_degree=True),
        )

        self.classifier = EdgePredictor(edge_feature_size=edge_feature_size, node_feature_size=32, hidden_size=8, num_classes=num_classes)
    
    def forward(self, blocks, n_feats):
        x = n_feats
        batch_size = x.shape[0]
        index = 0

        for layer in self.feature:
            if isinstance(layer, GATConv):
                if isinstance(blocks, dgl.DGLGraph):
                    x = layer(blocks, x)
                    x = x.view(batch_size, -1)
                else:
                    x = layer(blocks[index], x)
                    index += 1
            else:
                x = layer(x)

        if isinstance(blocks, dgl.DGLGraph):
            return self.classifier(blocks, x)
        else:
            return self.classifier(blocks[-1], x)

import dgl
import torch


def load_graph():
    data, _ = dgl.load_graphs('./data/graph.bin')
    graph = data[0]
    
    graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1)

    ndata = ['feat']
    edata = ['feat', 'train_mask', 'test_mask', 'label']
    
    return dgl.to_homogeneous(graph, ndata=ndata, edata=edata)

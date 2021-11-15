import dgl
import torch


def load_graph():
    data, _ = dgl.load_graphs('./data/graph.bin')
    graph = data[0]
    
    graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1)
    
    return graph

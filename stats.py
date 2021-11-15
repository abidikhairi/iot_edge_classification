import torch
from utils import load_graph

if __name__ == '__main__':
    
    graph = load_graph()

    print('Number of Nodes:', graph.num_nodes())
    print('Number of Edges:', graph.num_edges())
    print('Node Features:', graph.ndata['feat'].shape)
    print('Edge Features:', graph.edata['feat'].shape)
    print('Number of Training Edges:', torch.nonzero(graph.edata['train_mask'], as_tuple=False).shape[0])
    print('Number of Testing Edges:', torch.nonzero(graph.edata['test_mask'], as_tuple=False).shape[0])

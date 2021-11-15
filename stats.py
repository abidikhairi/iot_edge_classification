import dgl
import torch

if __name__ == '__main__':

    data, _ = dgl.load_graphs('./data/graph.bin')
    graph = data[0]
    
    graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1)

    print('Number of Nodes:', graph.num_nodes())
    print('Number of Edges:', graph.num_edges())
    print('Node Features:', graph.ndata['feat'].shape)
    print('Edge Features:', graph.edata['feat'].shape)
    print('Number of Training Edges:', torch.nonzero(graph.edata['train_mask'], as_tuple=False).shape[0])
    print('Number of Testing Edges:', torch.nonzero(graph.edata['test_mask'], as_tuple=False).shape[0])

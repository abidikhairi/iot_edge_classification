import argparse
import torch
import dgl
import wandb

import torchmetrics.functional as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from dgl.dataloading import EdgeDataLoader, MultiLayerNeighborSampler

from models import SAGE
from utils import load_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, criterion, optimizer, loader):
    model.train()
    
    train_loss = []
    train_accuracy = []
    
    for _, edges_subgraph, _ in loader:
        optimizer.zero_grad()

        labels = edges_subgraph.edata['label'].flatten().long().to(device)
        nfeats = edges_subgraph.ndata['feat'].to(device)

        logits = model(edges_subgraph, nfeats)
    
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        accuracy = metrics.accuracy(logits, labels)

        train_loss.append(loss.item())
        train_accuracy.append(accuracy.item())

    return {
        'train_loss': torch.tensor(train_loss).mean(),
        'train_accuracy': torch.tensor(train_accuracy).mean()
    }

def evaluate(model, criterion, loader):
    model.eval()
    
    with torch.no_grad():
        test_loss = []
        test_accuracy = []

        for _, edges_subgraph, _ in loader:
            labels = edges_subgraph.edata['label'].flatten().long().to(device)
            nfeats = edges_subgraph.ndata['feat'].to(device)
            
            logits = model(edges_subgraph, nfeats)
        
            loss = criterion(logits, labels)
            accuracy = metrics.accuracy(logits, labels)

            test_loss.append(loss.item())
            test_accuracy.append(accuracy.item())

        return {
            'test_loss': torch.tensor(test_loss).mean(),
            'test_accuracy': torch.tensor(test_accuracy).mean()
        }

def main(args):
    experiment = wandb.init(project='edge-classifcation', entity='u-jendouba-ai', name='SAGE IOT Edge Classification')

    epochs = args.epochs
    learning_rate = args.learning_rate
    
    graph = load_graph()
    graph = dgl.to_homogeneous(graph, ndata=['feat'], edata=['feat', 'label', 'train_mask', 'test_mask', 'label'])

    edge_feature_size = graph.edata['feat'].shape[1]
    node_feature_size = graph.ndata['feat'].shape[1]
    num_classes = len(torch.unique(graph.edata['label']))

    train_idx = torch.nonzero(graph.edata['train_mask'], as_tuple=False).flatten()
    test_idx = torch.nonzero(graph.edata['test_mask'], as_tuple=False).flatten()

    sampler = MultiLayerNeighborSampler([20, 25])
    trainloader = EdgeDataLoader(graph, train_idx, block_sampler=sampler, device=device, batch_size=1024)
    testloader = EdgeDataLoader(graph, test_idx, block_sampler=sampler, device=device, batch_size=1024)

    model = SAGE(node_feature_size=node_feature_size, edge_feature_size=edge_feature_size, num_classes=num_classes)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    graph = graph.to(device)
    
    wandb.watch(model)

    for _ in range(epochs):
        train_state = train(model, criterion, optimizer, trainloader) 
     
        test_state = evaluate(model, criterion, testloader)
        
        wandb.log(train_state)
        wandb.log(test_state)
    
    with torch.no_grad():
        labels = graph.edata['label'].flatten().long().to(device)
        nfeats = graph.ndata['feat'].to(device)
        logits = model(graph, nfeats)
        
        cm = metrics.confusion_matrix(logits[test_idx].cpu(), labels[test_idx].cpu(), num_classes=num_classes).cpu().numpy()
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=torch.arange(num_classes).numpy())
        
        disp.plot()
        plt.savefig('images/confusion_matrix_sage.png')

    experiment.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10, help='training epochs. Defaults: 10')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning rate. Defaults: {}'.format(1e-2))

    args = parser.parse_args()
    main(args)
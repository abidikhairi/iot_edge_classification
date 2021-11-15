import argparse
import torch
import wandb

import torchmetrics.functional as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from models import GAT
from utils import load_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, criterion, optimizer, graph, train_idx):
    model.train()
    
    optimizer.zero_grad()

    labels = graph.edata['label'].flatten().long().to(device)
    nfeats = graph.ndata['feat'].to(device)
    logits = model(graph, nfeats)
    
    loss = criterion(logits[train_idx], labels[train_idx])

    loss.backward()
    optimizer.step()

    accuracy = metrics.accuracy(logits, labels)

    return {
        'train_loss': loss,
        'train_accuracy': accuracy
    }

def evaluate(model, criterion, graph, test_idx):
    model.eval()
    
    with torch.no_grad():
        labels = graph.edata['label'].flatten().long().to(device)
        nfeats = graph.ndata['feat'].to(device)
        logits = model(graph, nfeats)
        
        loss = criterion(logits[test_idx], labels[test_idx])
        accuracy = metrics.accuracy(logits, labels)

        return {
            'test_loss': loss,
            'test_accuracy': accuracy
        }

def main(args):
    experiment = wandb.init(project='edge-classifcation', entity='flursky', name='GAT IOT Edge Classification')

    epochs = args.epochs
    learning_rate = args.learning_rate
    
    graph = load_graph()

    edge_feature_size = graph.edata['feat'].shape[1]
    node_feature_size = graph.ndata['feat'].shape[1]
    num_classes = len(torch.unique(graph.edata['label']))

    train_idx = torch.nonzero(graph.edata['train_mask'], as_tuple=False).flatten()
    test_idx = torch.nonzero(graph.edata['test_mask'], as_tuple=False).flatten()

    model = GAT(node_feature_size=node_feature_size, edge_feature_size=edge_feature_size, num_classes=num_classes)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    model.to(device)
    graph = graph.to(device)
    
    wandb.watch(model)

    for _ in range(epochs):
        train_state = train(model, criterion, optimizer, graph, train_idx) 
        test_state = evaluate(model, criterion, graph, test_idx)
        
        wandb.log(train_state)
        wandb.log(test_state)
    
    with torch.no_grad():
        labels = graph.edata['label'].flatten().long().to(device)
        nfeats = graph.ndata['feat'].to(device)
        logits = model(graph, nfeats)
        
        cm = metrics.confusion_matrix(logits[test_idx].cpu(), labels[test_idx].cpu(), num_classes=num_classes).cpu().numpy()
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=torch.arange(num_classes).numpy())
        
        disp.plot()
        plt.savefig('images/confusion_matrix_gat.png')

    experiment.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10, help='training epochs. Defaults: 10')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning rate. Defaults: {}'.format(1e-2))

    args = parser.parse_args()
    main(args)
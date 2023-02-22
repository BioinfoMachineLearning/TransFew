import os
import argparse
import torch
import warnings

from torch import optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import time

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import from_networkx

from Graph.DiamondDataset import DiamondDataset
from Graph.Testp import Planetoid
from Classes.Diamond import Diamond
from models.model import GAT

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--train_batch', type=int, default=1, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=10, help='Validation batch size.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--seq', type=float, default=0.9, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='biological_process', type=str, help='Ontology under consideration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'
device = 'cpu'
# wandb.init(project="transfun", entity='frimpz',
#            name="{}_{}".format(args.seq, args.ont))


def train_model(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))

        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            for data in data_loader['train']:
                optimizer.zero_grad()
                output = model(data.to(device))

                loss = criterion(output, getattr(data, args.ont))
                loss = (loss * class_weights).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.item()
                epoch_accuracy += accuracy_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5)
                epoch_precision += precision_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples")
                epoch_recall += recall_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples")
                epoch_f1 += f1_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples")

                print(f1_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples"))

            epoch_accuracy = epoch_accuracy / len(loaders['train'])
            epoch_precision = epoch_precision / len(loaders['train'])
            epoch_recall = epoch_recall / len(loaders['train'])
            epoch_f1 = epoch_f1 / len(loaders['train'])


dataset = DiamondDataset()


print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

dataset = Planetoid(root='data/Planetoid', split='random', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

exit()
train_dataloader = DataLoader(dataset, batch_size=args.train_batch, drop_last=False, shuffle=True)

model = GAT()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
criterion = torch.nn.BCELoss(reduction='none')


import math
import os
import argparse

import numpy as np
import torch
import warnings

import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import optim
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import time

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import from_networkx

import CONSTANTS
from Graph.DiamondDataset import DiamondDataset
from Graph.Testp import Planetoid
from Classes.Diamond import Diamond
from Loss.CustomLoss import FocalLoss
from models.model import GAT
from preprocessing.utils import load_ckp

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--train_batch', type=int, default=1, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=10, help='Validation batch size.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument("--ont", default='cc', type=str, help='Ontology under consideration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'
device = 'cpu'


# wandb.init(project="transfun2", entity='frimpz',
#            name="diamond_cc")


def train_model(start_epoch, min_val_loss, model, optimizer, criterion, data):
    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):
            ###################
            # train the model #
            ###################
            model.train()

            optimizer.zero_grad()
            output = model(data.to(device))

            loss = criterion(output, data.y)
            loss = (loss * weights).mean()
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            loss = loss.data.item()
            accuracy = accuracy_score(data.y.cpu(), output.cpu() > 0.5)
            precision = precision_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            recall = recall_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            f1 = f1_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            print(loss, accuracy, precision, recall, f1)

            wandb.log({"train_acc": accuracy,
                       "train_loss": loss,
                       "precision": precision,
                       "recall": recall,
                       "f1": f1})


print(args.ont)
dataset = DiamondDataset()
print(dataset)
exit()

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

counts = data.y.numpy().sum(axis=0)
max_count = max(counts)
weights = torch.tensor([math.log2(max_count / i) if i > 0 else math.log2(max_count + 10) for i in counts],
                                              dtype=torch.float).to(device)

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

model = GAT(dataset.num_features, dataset.num_classes)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# criterion = torch.nn.BCELoss(reduction='none')
criterion = FocalLoss()

ckp_dir = CONSTANTS.ROOT_DIR + 'checkpoints/{}/model_checkpoint/'.format(args.ont)
ckp_pth = ckp_dir + "current_checkpoint.pt"

if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
else:
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

current_epoch = 1
min_val_loss = np.Inf

wandb.config = {
    "learning_rate": args.lr,
    "epochs": current_epoch,
    "batch_size": args.train_batch
}

# dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
# data = dataset[0]
# print(data.y.dtype)
# exit()

train_model(current_epoch, min_val_loss,
            model=model, optimizer=optimizer,
            criterion=criterion, data=data)

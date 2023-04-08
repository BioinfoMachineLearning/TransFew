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
from Classes.Preprocess import Preprocess
from Graph.DiamondDataset import DiamondDataset
from Graph.Testp import Planetoid
from Classes.Diamond import Diamond
from Loss.CustomLoss import FocalLoss, FocalTverskyLoss
from models.model import GAT
from preprocessing.utils import load_ckp, pickle_load

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--ont", default='cc', type=str, help='Ontology under consideration')
parser.add_argument('--train_batch', type=int, default=1, help='Training batch size.')
# parser.add_argument('--valid_batch', type=int, default=10, help='Validation batch size.')
# parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'
device = 'cpu'


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

            loss = criterion(output[data.train_mask], data.y[data.train_mask])
            # loss = (loss * weights).mean()
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            loss = loss.data.item()
            accuracy = accuracy_score(data.y.cpu(), output.cpu() > 0.5)
            precision = precision_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            recall = recall_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            f1 = f1_score(data.y.cpu(), output.cpu() > 0.5, average="samples")

        ###################
        # Validate the model #
        ###################
        with torch.no_grad():
            model.eval()

            output = model(data.to(device))

            val_loss = criterion(output[data.valid_mask], data.y[data.valid_mask])
            # val_loss = (val_loss * weights).mean()
            val_loss = val_loss.mean()

            val_loss = val_loss.data.item()
            val_accuracy = accuracy_score(data.y.cpu(), output.cpu() > 0.5)
            val_precision = precision_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            val_recall = recall_score(data.y.cpu(), output.cpu() > 0.5, average="samples")
            val_f1 = f1_score(data.y.cpu(), output.cpu() > 0.5, average="samples")

        print('Epoch: {:04d}'.format(epoch),
              'train_loss: {:.4f}'.format(loss),
              'train_acc: {:.4f}'.format(accuracy),
              'precision: {:.4f}'.format(precision),
              'recall: {:.4f}'.format(recall),
              'f1: {:.4f}'.format(f1),
              'val_acc: {:.4f}'.format(val_accuracy),
              'val_loss: {:.4f}'.format(val_loss),
              'val_precision: {:.4f}'.format(val_precision),
              'val_recall: {:.4f}'.format(val_recall),
              'val_f1: {:.4f}'.format(val_f1),
              'time: {:.4f}s'.format(time.time() - t))

        # wandb.log({"train_acc": accuracy,
        #            "train_loss": loss,
        #            "precision": precision,
        #            "recall": recall,
        #            "f1": f1,
        #            "val_accuracy": val_accuracy,
        #            "val_loss": val_loss,
        #            "val_precision": val_precision,
        #            "val_recall": val_recall,
        #            "val_f1": val_f1,
        #            "time": time.time() - t})


print("++++++++++++++++++++++ Generating Data ++++++++++++++++++++++")
x = Preprocess()

print("++++++++++++++++++++++ Data generation finished ++++++++++++++++++++++")
exit()
dataset = DiamondDataset(ont=args.ont)

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

counts = data.y.numpy().sum(axis=0)
max_count = max(counts)
# weights = torch.tensor([math.log2(max_count / i) if i > 0 else math.log2(max_count + 10) for i in counts],
#                                               dtype=torch.float).to(device)

# weights = torch.tensor([max_count / i if i > 0 else max_count + 10 for i in counts], dtype=torch.float).to(device)

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

print('===========================================================================================================')

_templates = pickle_load(CONSTANTS.ROOT_DIR + "diamond/template")[args.ont]

templates = []
for node in data.nodes:
    templates.append(_templates[node])

templates = np.vstack(templates)

print(f'Shape of templates: {templates.shape}')

print('===========================================================================================================')

model = GAT(dataset.num_features, dataset.num_classes, templates)
model.to(device)
optimizer = optim.Adagrad(model.parameters(), lr=args.lr)  # , weight_decay=args.weight_decay)
# criterion = torch.nn.BCELoss(reduction='none')
criterion = FocalLoss()
# criterion = FocalTverskyLoss()

# ckp_dir = CONSTANTS.ROOT_DIR + 'checkpoints/{}/model_checkpoint/'.format(args.ont)
# ckp_pth = ckp_dir + "current_checkpoint.pt"

# if os.path.exists(ckp_pth):
#     print("Loading model checkpoint @ {}".format(ckp_pth))
#     model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
# else:
#     if not os.path.exists(ckp_dir):
#         os.makedirs(ckp_dir)

current_epoch = 1
min_val_loss = np.Inf

config = {
    "learning_rate": args.lr,
    "epochs": current_epoch,
    "batch_size": args.train_batch
}

# wandb.init(project="transfun2_{}".format(args.ont), entity='frimpz', config=config,
#            name="diamond_cc_classes_num_prot > 100_ only")
train_model(current_epoch, min_val_loss,
            model=model, optimizer=optimizer,
            criterion=criterion, data=data)

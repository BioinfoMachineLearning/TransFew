import argparse
import os
import numpy as np
import torch
import wandb
import CONSTANTS
import torch_geometric.transforms as T
from Utils import load_ckp, save_ckp
from models.model import LabelEncoder
import time
from torch_geometric.nn import GAE
import hparams
from num2words import num2words

os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--ont", default='bp', type=str, help='Ontology under consideration')
parser.add_argument("--load_weights", default=False, type=bool, help='Load weights from saved model')
parser.add_argument("--save_weights", default=False, type=bool, help='Save model weights')
parser.add_argument("--log_output", default=False, type=bool, help='Log output to weights and bias')
parser.add_argument("--features", default='biobert', type=str, help='Which features to use')
parser.add_argument("--gnn", default='GCN', type=str, help='Which GNN network to use')
parser.add_argument("--out_channels", default=1024, type=int, help='Final Hidden Dimension')


args = parser.parse_args()

torch.manual_seed(args.seed)


hyps = getattr(hparams, args.ont)
args.weight_decay = hyps['label_ae']['weight_decay']
args.lr = hyps['label_ae']['lr']

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cpu'
if args.cuda:
    device = 'cuda'

print("Ontology: {}, Learning rate: {},  Weight Decay: {}, Device: {}"\
      .format(args.ont, args.lr, args.weight_decay, device))



graph_path = CONSTANTS.ROOT_DIR + '{}/graph.pt'.format(args.ont)
data = torch.load(graph_path)

if args.features == 'x':
    if args.ont == 'cc':
        num_features = 2957
    elif args.ont == 'mf':
        num_features = 7224
    elif args.ont == 'bp':
        num_features = 21285
elif args.features == 'biobert':
    num_features = 768 
elif args.features == 'bert':
    num_features = 1024

variational = False


transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.2, num_test=0.0, is_undirected=False,
                                split_labels=True, add_negative_train_samples=True),
            ])
train_data, val_data, _ = transform(data)


def check_data_integrity(data, train_data):
    # wasn't sure about the Random split and wanted to confirm whether
    # the positive is not in the negative.

    a1, b1 = data.edge_index
    t1 = list(zip(a1.tolist(), b1.tolist()))
    t1 = sorted(t1, key=lambda element: (element[0], element[1]))


    a1, b1 = train_data.pos_edge_label_index
    t2 = list(zip(a1.tolist(), b1.tolist()))
    t2 = sorted(t2, key=lambda element: (element[0], element[1]))


    a1, b1 = train_data.neg_edge_label_index
    t3 = list(zip(a1.tolist(), b1.tolist()))
    t3 = sorted(t3, key=lambda element: (element[0], element[1]))


    assert len(t2) == len(set(t1).intersection(set(t2)))

    assert len(set(t1).intersection(set(t3))) == 0


# check_data_integrity(data=data, train_data=train_data)


def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data[args.features], data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    if variational:
       print("Using variational loss")
       loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss), auc, ap


@torch.no_grad()
def validate(data, model):
    model.eval()
    z = model.encode(data[args.features], data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return float(loss), auc, ap


def train_model(start_epoch, min_val_loss, train_data, 
                val_data, model, optimizer):
    
    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))

        start = time.time()

        loss, auc, ap = train(data=train_data, model=model, optimizer=optimizer)
        val_loss, val_auc, val_ap = validate(val_data, model=model)

        epoch_time = time.time() - start
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, time: {epoch_time:.4f}')
        

        if args.log_output:
            wandb.log({"Epoch": epoch,
                    "train_loss": loss,
                    "train_auc": auc,
                    "train_ap": ap,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "time": epoch_time
                    })
            
        checkpoint = {
                    'epoch': epoch,
                    'valid_loss_min': val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': None
                }
        
        if args.save_weights:
            # save checkpoint
            save_ckp(checkpoint, False, ckp_dir)
            
            if val_loss <= min_val_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'. \
                        format(min_val_loss, val_loss))
            
                # save checkpoint as best model
                save_ckp(checkpoint, True, ckp_dir)
                min_val_loss = val_loss


model = GAE(LabelEncoder(num_features, args.out_channels, features=args.features, gnn=args.gnn))
model = model.to(device)

print(num2words(count_params(model)))

exit()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/label/{}'.format(args.ont, args.gnn)
ckp_pth = ckp_dir + "current_checkpoint.pt"

if args.load_weights and os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, lr_scheduler, current_epoch, min_val_loss = load_ckp(checkpoint_dir=ckp_dir, model=model, optimizer=optimizer, lr_scheduler=None, best_model=True)
else:
    current_epoch = 0
    min_val_loss = np.Inf


config = {
    "learning_rate": args.lr,
    "epochs": current_epoch, # saved previous epoch
    "weight_decay": args.weight_decay
}

if args.log_output:
    wandb.init(project="LabelEmbedding", entity='frimpz', config=config, name="{}_label_{}_{}_{}".format(args.ont, args.lr, args.weight_decay, args.gnn))

train_model(start_epoch=current_epoch, min_val_loss=min_val_loss,
            train_data=train_data, val_data=val_data,
            model=model, optimizer=optimizer)
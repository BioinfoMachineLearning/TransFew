import math, os, time
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
import wandb
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, MultiStepLR
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.loader import DataLoader
from Dataset.Dataset import TransFunDataset
import CONSTANTS
from Dataset.FastDataset import FastTransFunDataset
from models.model import TFun, TFun_submodel
from Utils import load_ckp, pickle_load, read_cafa5_scores, save_ckp
import hparams as hparams
from num2words import num2words
warnings.filterwarnings("ignore", category=UserWarning)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.') # 0.0001
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).') #5e-4
parser.add_argument("--ont", default='cc', type=str, help='Ontology under consideration')
parser.add_argument('--train_batch', type=int, default=64, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=64, help='Validation batch size.')
parser.add_argument('--submodel', type=str, default='full', help='Sub model to train')
parser.add_argument("--load_weights", default=False, type=bool, help='Load weights from saved model')
parser.add_argument("--save_weights", default=False, type=bool, help='Save model weights')
parser.add_argument("--log_output", default=False, type=bool, help='Log output to weights and bias')
parser.add_argument('--label_features', type=str, default='linear', help='Sub model to train')


torch.manual_seed(17)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hyps = getattr(hparams, args.ont)
args.epochs = hyps[args.submodel]['epochs']
if args.submodel == "full":
    args.lr = hyps[args.submodel]['lr'][args.label_features]
else:
    args.lr = hyps[args.submodel]['lr']
args.weight_decay = hyps[args.submodel]['weight_decay']
args.train_batch = hyps[args.submodel]['batch_size']
args.valid_batch = hyps[args.submodel]['batch_size']


if args.cuda:
    device = 'cuda:0'
#device = 'cpu'


threshold = {'mf': 30, 'cc' : 30, 'bp': 100}
_term_indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/term_indicies".format(args.ont))



if args.submodel == 'full': 
    term_indicies =  torch.tensor(_term_indicies[0])
    sub_indicies = torch.tensor(_term_indicies[threshold[args.ont]])
else:
    term_indicies = torch.tensor(_term_indicies[threshold[args.ont]])
    sub_indicies = term_indicies


def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_scores(labels, preds):

    conf = preds/labels

    true_positives = torch.sum(conf == 1).item()
    false_positives = torch.sum(conf == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(conf)).item()
    false_negatives = torch.sum(conf == 0).item()

    accuracy = (true_positives + true_negatives) / (1.0 * (true_positives + true_negatives + false_positives + false_negatives))
    recall = true_positives / (1.0 * (true_positives + false_negatives))
    precision = true_positives / (1.0 * (true_positives + false_positives))
    fscore = 2 * precision * recall / (precision + recall)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten().cpu(), preds.flatten().cpu())
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, fscore , roc_auc



def train_model(start_epoch, min_val_loss, train_data, val_data, model, optimizer, lr_scheduler, criterion, class_weights):

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))

        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1, epoch_roc_auc =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1, val_roc_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            num_batches = 0
            for _epoch, _data in enumerate(train_data):

                if args.submodel == 'full':
                    features = _data[:5]
                    labels = _data[5]
                else:
                    features, labels = _data
                    features = features.to(device)
                
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(features)

                # loss = (criterion(output, labels)).mean()

                loss = (criterion(output, labels) * class_weights.to(device)).mean()

                loss.backward()
                optimizer.step()


                epoch_loss += loss.data.item()
                out_cpu_5 = output > 0.5


                a, p, r, f, roc  = compute_scores(labels, out_cpu_5)

                epoch_accuracy += a
                epoch_precision += p
                epoch_recall += r
                epoch_f1 += f
                epoch_roc_auc += roc
                

                # print(f1_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples"))
                '''epoch_accuracy += accuracy_score(y_true=labels.cpu(), y_pred=out_cpu_5)
                epoch_precision += precision_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                epoch_recall += recall_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                epoch_f1 += f1_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                epoch_roc_auc += compute_roc(labels.cpu().detach().numpy(), output.cpu().detach().numpy())'''
                
                num_batches = num_batches + 1

            epoch_accuracy = epoch_accuracy / num_batches
            epoch_precision = epoch_precision / num_batches
            epoch_recall = epoch_recall / num_batches
            epoch_f1 = epoch_f1 / num_batches
            epoch_roc_auc = epoch_roc_auc / num_batches


        ###################
        # Validate the model #
        ###################
        with torch.no_grad():
            model.eval()

            num_batches = 0
            for _epoch, _data in enumerate(val_data):

                if args.submodel == 'full':
                    features = _data[:5]
                    labels = _data[5]
                else:
                    features, labels = _data
                    features = features.to(device)
                
                labels = labels.to(device)
                
                output = model(features)


                out_cpu_5 = output > 0.5
                val_loss += (criterion(output, labels)* class_weights.to(device)).mean().data.item()

                '''val_accuracy += accuracy_score(y_true=labels.cpu(), y_pred=out_cpu_5)
                val_precision += precision_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                val_recall += recall_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                val_f1 += f1_score(y_true=labels.cpu(), y_pred=out_cpu_5, average="samples")
                val_roc_auc += compute_roc(labels.cpu().detach().numpy(), output.cpu().detach().numpy())'''

                a, p, r, f, roc  = compute_scores(labels, out_cpu_5)

                val_accuracy += a
                val_precision += p
                val_recall += r
                val_f1 += f
                val_roc_auc += roc

                num_batches = num_batches +1


            val_accuracy = val_accuracy / num_batches
            val_precision = val_precision / num_batches
            val_recall = val_recall / num_batches
            val_f1 = val_f1 / num_batches
            val_roc_auc = val_roc_auc / num_batches


        lr_scheduler.step()

        print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.4f}'.format(epoch_loss),
                  'train_acc: {:.4f}'.format(epoch_accuracy),
                  'precision: {:.4f}'.format(epoch_precision),
                  'recall: {:.4f}'.format(epoch_recall),
                  'f1: {:.4f}'.format(epoch_f1),
                  'roc_auc: {:.4f}'.format(epoch_roc_auc),
                  'val_acc: {:.4f}'.format(val_accuracy),
                  'val_loss: {:.4f}'.format(val_loss),
                  'val_precision: {:.4f}'.format(val_precision),
                  'val_recall: {:.4f}'.format(val_recall),
                  'val_f1: {:.4f}'.format(val_f1),
                  'val_roc_auc: {:.4f}'.format(val_roc_auc),
                  'time: {:.4f}s'.format(time.time() - t)
                  )


        if args.log_output:
            wandb.log({"Epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_acc": epoch_accuracy,
                    "precision": epoch_precision,
                    "recall": epoch_recall,
                    "f1": epoch_f1,
                    "roc_auc": epoch_roc_auc,
                    "val_acc": val_accuracy,
                    "val_loss": val_loss,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_roc_auc" : val_roc_auc,
                    "time": time.time() - t
                    })
            
            checkpoint = {
                    'epoch': epoch,
                    'valid_loss_min': val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
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


pth = CONSTANTS.ROOT_DIR + "{}/{}_data"
train_dataset = FastTransFunDataset(data_pth=pth.format(args.ont, 'train'), term_indicies=term_indicies, submodel=args.submodel, weights=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
class_weights = train_dataset.get_class_weights().to(device)

val_dataset = FastTransFunDataset(data_pth=pth.format(args.ont, 'validation'), term_indicies=term_indicies, submodel=args.submodel)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.valid_batch, shuffle=True)


kwargs = {
    'device': device,
    'ont': args.ont,
    'indicies': term_indicies,
    'sub_indicies': sub_indicies,
    'sub_model': args.submodel,
    'load_weights': args.load_weights,
    'label_features': args.label_features
}


if args.submodel == 'full':
    model = TFun(**kwargs)
    for name, param in model.named_parameters():
        if name.startswith("interpro"):
            param.requires_grad = False
        if name.startswith("msa_mlp"):
            param.requires_grad = False
        if name.startswith("diamond_mlp"):
            param.requires_grad = False
        if name.startswith("esm_mlp"):
            param.requires_grad = False
        if name.startswith("string_mlp"):
            param.requires_grad = False
        if name.startswith("label_embedding"):
            param.requires_grad = False
    ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}_{}/'.format(args.ont, args.submodel, kwargs['label_features'])
    ckp_pth = ckp_dir + "current_checkpoint.pt"
else:
    model = TFun_submodel(**kwargs)
    ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}/'.format(args.ont, args.submodel)
    ckp_pth = ckp_dir + "current_checkpoint.pt"


print("Ontology: {}, \n Learning rate: {}, \n Submodel: {}, \n Batch size: {}, \n Weight Decay: {}, \n Device: {}, \
      Label Embedding: {}, \n Number of Parameters: {}, \n Number of terms: {}"\
      .format(args.ont, args.lr, args.submodel, args.train_batch, args.weight_decay, device, args.label_features, num2words(count_params(model)), term_indicies.shape))

# print(model)
model.to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.BCELoss(reduction='none')
lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)


if args.load_weights and os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, lr_scheduler, current_epoch, min_val_loss = load_ckp(checkpoint_dir=ckp_dir, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, best_model=False)
else:
    current_epoch = 0
    min_val_loss = np.Inf

current_epoch = current_epoch#  + 1
config = {
    "learning_rate": args.lr,
    "epochs": current_epoch, # saved previous epoch
    "batch_size": args.train_batch,
    "valid_size": args.valid_batch,
    "weight_decay": args.weight_decay
}

if args.log_output:
    wandb.init(project="TransZero", entity='frimpz', config=config, name="{}_{}_{}".format(args.ont, args.submodel, args.label_features))


train_model(start_epoch=current_epoch, min_val_loss=min_val_loss, 
            train_data=trainloader, val_data=valloader, model=model, 
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            criterion=criterion, class_weights=class_weights)
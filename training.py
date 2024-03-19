import math, os, time
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
import wandb
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.loader import DataLoader
import CONSTANTS
from Dataset.MyDataset import TransFewDataset
from models.model import TFun, TFun_submodel
from Loss.Loss import HierarchicalLoss, DiceLoss
from Utils import load_ckp, pickle_load, read_cafa5_scores, save_ckp
import hparams as hparams
from torch.autograd import Variable
from num2words import num2words
warnings.filterwarnings("ignore", category=UserWarning)
import collections
import torch.nn.functional as F

'''os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "online"'''

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.') # 0.0001
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).') #5e-4
parser.add_argument("--ont", default='mf', type=str, help='Ontology under consideration')
parser.add_argument('--train_batch', type=int, default=64, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=64, help='Validation batch size.')
parser.add_argument('--submodel', type=str, default='full', help='Sub model to train')
parser.add_argument("--group", default='freq', type=str, help='Frequent or Rare model')
parser.add_argument("--load_weights", default=False, type=bool, help='Load weights from saved model')
parser.add_argument("--save_weights", default=False, type=bool, help='Save model weights')
parser.add_argument("--log_output", default=False, type=bool, help='Log output to weights and bias')
parser.add_argument('--label_features', type=str, default='gcn', help='Sub model to train')
parser.add_argument('--entropy_loss', type=str, default=0.1, help='Entropy_loss')

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
# device = 'cpu'

threshold = {'mf': 30, 'cc' : 30, 'bp': 30}
_term_indicies = pickle_load(CONSTANTS.ROOT_DIR + "{}/term_indicies".format(args.ont))

if args.ont == 'bp':
    full_term_indicies, mid_term_indicies,  freq_term_indicies =  _term_indicies[0], _term_indicies[5], _term_indicies[30]
    rare_term_indicies_2 = torch.tensor([i for i in full_term_indicies if not i in set(mid_term_indicies)]).to(device)
    rare_term_indicies = torch.tensor([i for i in mid_term_indicies if not i in set(freq_term_indicies)]).to(device)
    full_term_indicies, freq_term_indicies =  torch.tensor(_term_indicies[0]).to(device), torch.tensor(freq_term_indicies).to(device)
else:
    full_term_indicies =  _term_indicies[0]
    freq_term_indicies = _term_indicies[threshold[args.ont]]
    rare_term_indicies = torch.tensor([i for i in full_term_indicies if not i in set(freq_term_indicies)]).to(device)
    full_term_indicies =  torch.tensor(full_term_indicies).to(device)
    freq_term_indicies = torch.tensor(freq_term_indicies).to(device)
    rare_term_indicies_2 = None


def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_weights(labels, alpha):
    label_frequencies =  torch.sum(labels, dim=0)
    total_count = torch.sum(label_frequencies, dim=0)
    # take care of zeros
    label_frequencies =  torch.add(label_frequencies, 1e-2)
    weights = torch.div(total_count, label_frequencies)
    weights = torch.pow(torch.log2(weights), alpha)
    return weights



def compute_scores(labels, preds):
    conf = preds/labels
    true_positives = torch.sum(conf == 1).item() + 1e-5
    false_positives = torch.sum(conf == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(conf)).item()
    false_negatives = torch.sum(conf == 0).item()

    accuracy = (true_positives + true_negatives) / (1.0 * (true_positives + true_negatives + false_positives + false_negatives))
    recall = true_positives / (1.0 * (true_positives + false_negatives))
    precision = true_positives / (1.0 * (true_positives + false_positives))
    fscore = 2 * precision * recall / (precision + recall)
    # print("TP:{}, FP:{}, TN:{}, FN:{}".format(true_positives, false_positives, true_negatives, false_negatives))

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten().cpu(), preds.flatten().cpu())
    roc_auc = auc(fpr, tpr)
    return accuracy, precision, recall, fscore , roc_auc

from torch.nn import Sigmoid
sigmoid = Sigmoid()

def train_model(start_epoch, min_val_loss, train_data, val_data, model, optimizer, lr_scheduler, criterion):

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))

        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1, epoch_roc_auc =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1, val_roc_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        epoch_accuracy_r, epoch_precision_r, epoch_recall_r, epoch_f1_r , epoch_roc_auc_r = 0.0, 0.0, 0.0, 0.0, 0.0
        val_accuracy_r, val_precision_r, val_recall_r, val_f1_r , val_roc_auc_r = 0.0, 0.0, 0.0, 0.0, 0.0
        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            num_batches = 0
            for _epoch, _data in enumerate(train_data):

                if args.submodel == 'full':
                    features = _data[:3]
                    labels = _data[3].to(device)

                    freq_labels = torch.index_select(labels, 1, freq_term_indicies)
                    rare_labels = torch.index_select(labels, 1, rare_term_indicies)
                    labels = torch.index_select(labels.to(device), 1, full_term_indicies)

                    class_weights = get_weights(labels, hyps[args.submodel]['weight_factor'])

                    # class_weights = class_weights * labels
                    # class_weights[class_weights==0] = 2

                    # print(torch.max(class_weights), torch.min(class_weights))
                    
                    optimizer.zero_grad()

                    output, att = model(features)


                    output_freq = torch.index_select(output, 1, freq_term_indicies)
                    output_rare  = torch.index_select(output, 1, rare_term_indicies)
                    output = torch.index_select(output, 1, full_term_indicies)
                    # att = torch.index_select(att.squeeze(1), 1, full_term_indicies)

                    entropy_loss = -torch.sum(output * torch.log(output), dim=1).mean()
                    loss = (criterion(output, labels) * class_weights).mean()
                    # attn_loss = (criterion(sigmoid(att), labels) * class_weights).mean()

                    # klloss = (kl_loss(att, freq_labels)).mean()

                    #print(_att)

                    # print(torch.topk(_att, 7, dim=1))


                    # print(klloss, loss, rare_loss, rare_loss)



                    # loss =  rare_loss+freq_loss


                    # loss = loss + args.entropy_loss * entropy_loss
                    # print(loss, entropy_loss)
        
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.data.item()

                    a, p, r, f, roc  = compute_scores(freq_labels, output_freq > 0.5)
                    a_r, p_r, r_r, f_r, roc_r  = compute_scores(rare_labels, output_rare > .5)

                    epoch_accuracy += a
                    epoch_precision += p
                    epoch_recall += r
                    epoch_f1 += f
                    epoch_roc_auc += roc

                    epoch_accuracy_r += a_r
                    epoch_precision_r += p_r
                    epoch_recall_r += r_r
                    epoch_f1_r += f_r
                    epoch_roc_auc_r += roc_r
                    
                else:
                    features, labels = _data
                    features = features.to(device)

                    if args.group == 'freq':
                        labels = torch.index_select(labels.to(device), 1, freq_term_indicies)
                    elif args.group == 'rare':
                        labels = torch.index_select(labels.to(device), 1, rare_term_indicies)

                    class_weights = get_weights(labels, hyps[args.submodel]['weight_factor'])

                    # print(torch.min(class_weights), torch.max(class_weights))

                    if args.group == 'rare':
                        class_weights = class_weights * labels
                        class_weights[class_weights==0] = 5.0

                    #print(torch.min(class_weights), torch.max(class_weights))

                    #print(torch.min(class_weights), torch.max(class_weights))#, torch.mean(class_weights), torch.median(class_weights), torch.mode(class_weights))

                    optimizer.zero_grad()
                    output = model(features)
                    loss = (criterion(output, labels) * class_weights).mean()
        

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.data.item()

                    a, p, r, f, roc  = compute_scores(labels, output > .5)

                    if args.group == 'freq':
                        epoch_accuracy += a
                        epoch_precision += p
                        epoch_recall += r
                        epoch_f1 += f
                        epoch_roc_auc += roc
                    elif args.group == 'rare':
                        epoch_accuracy_r += a
                        epoch_precision_r += p
                        epoch_recall_r += r
                        epoch_f1_r += f
                        epoch_roc_auc_r += roc
                
                num_batches = num_batches + 1

            epoch_accuracy = epoch_accuracy / num_batches
            epoch_precision = epoch_precision / num_batches
            epoch_recall = epoch_recall / num_batches
            epoch_f1 = epoch_f1 / num_batches
            epoch_roc_auc = epoch_roc_auc / num_batches

            epoch_accuracy_r = epoch_accuracy_r / num_batches
            epoch_precision_r = epoch_precision_r / num_batches
            epoch_recall_r = epoch_recall_r / num_batches
            epoch_f1_r = epoch_f1_r / num_batches
            epoch_roc_auc_r = epoch_roc_auc_r / num_batches


        ###################
        # Validate the model #
        ###################
        with torch.no_grad():
            model.eval()

            num_batches = 0
            for _epoch, _data in enumerate(val_data):

                if args.submodel == 'full':
                    features = _data[:3]
                    labels = _data[3].to(device)

                    freq_labels = torch.index_select(labels, 1, freq_term_indicies)
                    rare_labels = torch.index_select(labels, 1, rare_term_indicies)
                    labels = torch.index_select(labels.to(device), 1, full_term_indicies)

                    output, _ = model(features)
                    output_freq = torch.index_select(output, 1, freq_term_indicies)
                    output_rare = torch.index_select(output, 1, rare_term_indicies)
                    output = torch.index_select(output, 1, full_term_indicies)

                    loss = (criterion(output, labels)).mean()
                    entropy_loss = -torch.sum(output * torch.log(output), dim=1).mean()

                    loss = loss + args.entropy_loss * entropy_loss

                    val_loss += loss.data.item()
                    a, p, r, f, roc  = compute_scores(freq_labels, output_freq > 0.5)
                    a_r, p_r, r_r, f_r, roc_r  = compute_scores(rare_labels, output_rare > .5)

                    val_accuracy += a
                    val_precision += p
                    val_recall += r
                    val_f1 += f
                    val_roc_auc += roc

                    val_accuracy_r += a_r
                    val_precision_r += p_r
                    val_recall_r += r_r
                    val_f1_r += f_r
                    val_roc_auc_r += roc_r

                    
                else:
                    features, labels = _data
                    features = features.to(device)

                    if args.group == 'freq':
                        labels = torch.index_select(labels.to(device), 1, freq_term_indicies)
                    elif args.group == 'rare':
                        labels = torch.index_select(labels.to(device), 1, rare_term_indicies)
                    output = model(features)

                    loss = (criterion(output, labels)).mean()
                    entropy_loss = -torch.sum(output * torch.log(output), dim=1).mean()

                    val_loss += loss.data.item()
                    a, p, r, f, roc  = compute_scores(labels, output > .5)

                    if args.group == 'freq':
                        val_accuracy += a
                        val_precision += p
                        val_recall += r
                        val_f1 += f
                        val_roc_auc += roc
                    elif args.group == 'rare':
                        val_accuracy_r += a
                        val_precision_r += p
                        val_recall_r += r
                        val_f1_r += f
                        val_roc_auc_r += roc

                num_batches = num_batches + 1


            val_accuracy = val_accuracy / num_batches
            val_precision = val_precision / num_batches
            val_recall = val_recall / num_batches
            val_f1 = val_f1 / num_batches
            val_roc_auc = val_roc_auc / num_batches

            val_accuracy_r = val_accuracy_r / num_batches
            val_precision_r = val_precision_r / num_batches
            val_recall_r = val_recall_r / num_batches
            val_f1_r = val_f1_r / num_batches
            val_roc_auc_r = val_roc_auc_r / num_batches


        lr_scheduler.step()

        print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.4f}'.format(epoch_loss),
                  'train_freq_accuracy: {:.4f}'.format(epoch_accuracy),
                  'train_rare_accuracy: {:.4f}'.format(epoch_accuracy_r),
                  'precision_freq: {:.4f}'.format(epoch_precision),
                  'precision_rare: {:.4f}'.format(epoch_precision_r),
                  'recall_freq: {:.4f}'.format(epoch_recall),
                  'recall_rare: {:.4f}'.format(epoch_recall_r),
                  'f1_freq: {:.4f}'.format(epoch_f1),
                  'f1_rare: {:.4f}'.format(epoch_f1_r),
                  'train_roc_auc_freq: {:.4f}'.format(epoch_roc_auc),
                  'train_roc_auc_rare: {:.4f}'.format(epoch_roc_auc_r),

                  'val_acc: {:.4f}'.format(val_accuracy),
                  'val_acc_freq: {:.4f}'.format(val_accuracy_r),
                  'val_loss: {:.4f}'.format(val_loss),
                  'val_precision_freq: {:.4f}'.format(val_precision),
                  'val_precision_rare: {:.4f}'.format(val_precision_r),
                  'val_recall_freq: {:.4f}'.format(val_recall),
                  'val_recall_rare: {:.4f}'.format(val_recall_r),
                  'val_f1_freq: {:.4f}'.format(val_f1),
                  'val_f1_rare: {:.4f}'.format(val_f1_r),
                  'val_roc_auc_freq: {:.4f}'.format(val_roc_auc),
                  'val_roc_auc_rare: {:.4f}'.format(val_roc_auc_r),
                  'time: {:.4f}s'.format(time.time() - t))

        # Weights and bias. set log_output=False
        if args.log_output:
            wandb.log({"Epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_freq_accuracy": epoch_accuracy,
                    "train_rare_accuracy": epoch_accuracy_r,
                    "precision_freq": epoch_precision,
                    "precision_rare": epoch_precision_r,
                    "recall_freq": epoch_recall,
                    "recall_rare": epoch_recall_r,
                    "f1_freq": epoch_f1, 
                    "f1_rare": epoch_f1_r,
                    "train_roc_auc_freq": epoch_roc_auc, 
                    "train_roc_auc_rare": epoch_roc_auc_r,
                    "val_loss": val_loss,
                    "val_acc_freq": val_accuracy, 
                    "val_acc_rare": val_accuracy_r,
                    "val_precision_freq": val_precision, 
                    "val_precision_rare": val_precision_r,
                    "val_recall_freq": val_recall, 
                    "val_recall_rare": val_recall_r,
                    "val_f1_freq": val_f1, 
                    "val_f1_rare": val_f1_r,
                    "val_roc_auc_freq": val_roc_auc, 
                    "val_roc_auc_rare": val_roc_auc_r,
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
train_dataset = TransFewDataset(data_pth=pth.format(args.ont, 'train'), submodel=args.submodel)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)

val_dataset = TransFewDataset(data_pth=pth.format(args.ont, 'validation'), submodel=args.submodel)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.valid_batch, shuffle=True)


kwargs = {
    'device': device,
    'ont': args.ont,
    'full_indicies': full_term_indicies,
    'freq_indicies': freq_term_indicies,
    'rare_indicies': rare_term_indicies,
    'rare_indicies_2': rare_term_indicies_2,
    'sub_model': args.submodel,
    'load_weights': args.load_weights,
    'label_features': args.label_features,
    'group': args.group
}


if args.submodel == 'full':
    model = TFun(**kwargs)
    for name, param in model.named_parameters():
        '''if name.startswith("interpro"):
            param.requires_grad = False
        if name.startswith("msa_mlp"):
            param.requires_grad = False
        if name.startswith("diamond_mlp"):
            param.requires_grad = False
        if name.startswith("esm_mlp"):
            param.requires_grad = False
        if name.startswith("string_mlp"):
            param.requires_grad = False'''
        if name.startswith("label_embedding"):
            param.requires_grad = False
    ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}_{}/'.format(args.ont, args.submodel, kwargs['label_features'])
    ckp_pth = ckp_dir + "current_checkpoint.pt"
else:
    model = TFun_submodel(**kwargs)
    ckp_dir = CONSTANTS.ROOT_DIR + '{}/models/{}_{}_2/'.format(args.ont, args.submodel,args.group)
    ckp_pth = ckp_dir + "current_checkpoint.pt"


print("Ontology: {}, \n Learning rate: {}, Leraning rate scheduler: {}, \n Submodel: {}, Group: {}, \n Batch size: {}, \n Weight Decay: {}  Class weight factor: {}, \n Device: {}, \
      Label Embedding: {}, \n Number of Parameters: {}, \n Number of terms: {} Freq terms: {}, Rare terms: {}"\
      .format(args.ont, args.lr, hyps[args.submodel]['lr_scheduler'], args.submodel, args.group, args.train_batch, args.weight_decay, hyps[args.submodel]['weight_factor'], device, args.label_features, num2words(count_params(model)), full_term_indicies.shape, freq_term_indicies.shape, rare_term_indicies.shape))


# lamb = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
for name, param in model.named_parameters():
   if param.requires_grad:
       print(name)



print(model)
model.to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


criterion = torch.nn.BCELoss(reduction='none')
kl_loss = torch.nn.KLDivLoss(reduction="none")
# criterion1 = DiceLoss()
lr_scheduler = CosineAnnealingLR(optimizer, hyps[args.submodel]['lr_scheduler'])


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
    wandb.init(project="TransZero", entity='frimpz', config=config, name="{}_{}_{}_{}_{}".format(args.ont, args.submodel, args.label_features, args.group, args.entropy_loss))


train_model(start_epoch=current_epoch, min_val_loss=min_val_loss, 
            train_data=trainloader, val_data=valloader, model=model, 
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            criterion=criterion)
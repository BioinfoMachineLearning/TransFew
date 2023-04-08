import torch.nn
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear, Sigmoid
from torchsummary import summary

from models import net_utils


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, templates):
        super(GAT, self).__init__()

        self.hid = 1024
        self.in_head = 2
        self.out_head = 1

        self.conv1 = GATv2Conv(num_features, self.hid, heads=self.in_head, dropout=0.1,
                               edge_dim=1, concat=False)
        self.conv2 = GATv2Conv(self.hid, num_classes, heads=self.out_head,
                               concat=True, dropout=0.1, edge_dim=1)
        self.conv3 = GATv2Conv(num_features, num_classes, heads=self.in_head, dropout=0.1,
                               edge_dim=1, concat=False)
        self.conv4 = self.seq_fc1 = net_utils.FC(templates.shape[1], num_classes,
                                                 act_fun='relu', bnorm=False)

        ##################################################################################
        # (in_features, out_features, bnorm, act_fun, dropout)
        self.des = [(templates.shape[1], 1000, True, 'relu', 0.1),
                    (1000, num_classes, False, None, 0.1)]

        self.mlp = net_utils.MLP(self.des).to(torch.float)

        self.bn1 = net_utils.BNormActFun(self.hid, bnorm=True, act_fun="tanh")

        self.templates = torch.from_numpy(templates).float()
        self.num_layers = 2
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = Sigmoid()

    def forward_once(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # x.to(torch.device('cuda'))
        # self.mlp.to((torch.device('cuda')))
        #
        # print(self.mlp)
        #
        # print(summary(self.mlp, (x.shape[1], )))

        # print(self.mlp)

        # p = self.conv4(self.templates.float())

        x = self.mlp(self.templates)

        return x
        # # x = self.dropout(x)
        # x = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x)
        # x = self.conv2(x, edge_index, edge_attr)
        # print(x.shape)
        # exit()
        #
        # # x = F.relu(x)
        # x = self.dropout(x)
        #
        # exit()
        #
        # return x

    def forward(self, data):
        x = self.forward_once(data)
        x = self.sigmoid(x)
        # passes = []
        # for layer in range(self.num_layers):
        #     passes.append(self.forward_once(data))
        #
        # x = torch.cat(passes, 1)
        # x = self.fc3(x)
        # x = self.sigmoid(x)

        return x

# works well
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels = kwargs['hidden']
        edge_features = kwargs['edge_features']
        num_classes = kwargs['num_classes']
        num_egnn_layers = kwargs['egnn_layers']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']
        self.device = kwargs['device']

        self.fc1 = nn.Linear(input_features_size, num_classes, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(num_classes, int(num_classes / 2), bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(int(num_classes / 2), int(num_classes / 4), bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.final = net_utils.FC(num_classes + int(num_classes / 2) + int(num_classes / 4), num_classes, relu=False, bnorm=False)
        self.sig = Sigmoid()

    def forward(self, data):

        x_res, x_emb_seq, edge_index, edge_index_dist, x_batch, x_pos = data['atoms'].embedding_features_per_residue, \
            data['atoms'].embedding_features_per_sequence, \
            data[self.edge_type].edge_index, \
            data['dist_10'].edge_index, \
            data['atoms'].batch, \
            data['atoms'].pos

        x = self.fc1(x_emb_seq)
        x1 = self.relu1(x)

        x2 = self.fc2(x1)
        x2 = self.relu2(x2)

        x3 = self.fc3(x2)
        x3 = self.relu3(x3)

        x = torch.cat([x1, x2, x3], 1)
        x = self.final(x)
        x = self.sig(x)

        return x
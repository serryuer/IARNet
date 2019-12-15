import os

import torch
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from torch.nn import functional as F
from torch_geometric.data import DataListLoader
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool, DataParallel
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from transformers import BertModel

from data_utils.FakedditGEARDataset import FakedditGEARDataset


class GEARConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim=100, concat=True, bias=True, **kwargs):
        super(GEARConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.weight_0 = Parameter(torch.Tensor(2 * out_channels, hidden_dim))
        self.weight_1 = Parameter(torch.Tensor(1, hidden_dim))
        self.att = Parameter(torch.Tensor(1, 2 * out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # x_j = x_j.view(self.heads, self.out_channels)
        # x_i = x_i.view(self.heads, self.out_channels)
        # alpha : (1, hidden_dim)  * (hidden_dim) = 1
        alpha = torch.matmul(self.weight_1, F.relu(torch.matmul(torch.cat([x_i, x_j], dim=-1), self.weight_0)))
        alpha = softmax(alpha, edge_index_i, size_i)


class GEAR(nn.Module):
    def __init__(self,
                 num_class=2,
                 bert_path='/home/yujunshuai/model/chinese_L-12_H-768_A-12'):
        super(GEAR, self).__init__()

        self.num_class = num_class

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=bert_path
        )

        self.conv1 = GATConv(768, 768)
        self.conv2 = GATConv(768, 512)

        self.pool = global_mean_pool

        self.classifier = nn.Linear(512, self.num_class)

    def forward(self, data):
        # node_features : (node_count, 2, sequence_length)
        node_features, edge_index, batch, labels = data.x, data.edge_index, data.batch, data.y
        input_ids = node_features[:, 0, :]
        attention_mask = node_features[:, 1, :]
        node_features = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)[1]

        node_features = F.relu(self.conv1(node_features, edge_index))
        node_features = F.relu(self.conv2(node_features, edge_index))

        pooled_features = self.pool(node_features, batch)

        logits = self.classifier(pooled_features)
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    BERT_PATH = '/home/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'
    dataset = FakedditGEARDataset('/home/yujunshuai/data/Fakeddit/fakeddit_v1.0/',
                                  bert_path=BERT_PATH,
                                  max_sequence_length=256,
                                  num_class=2)

    train_loader = DataListLoader(dataset, batch_size=10, shuffle=True)
    model = DataParallel(GEAR(num_class=2, bert_path=BERT_PATH))

    model = model.cuda()

    for data in train_loader:
        outputs = model(data)
        print(outputs)

from typing import List

import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
import numpy as np

from torch_geometric.nn import GATConv, global_mean_pool
from transformers import BertModel


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_channels=in_size, out_channels=out_size, heads=layer_num_heads,
                                           dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, h, gs):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(h, g)

        return h


class HANForFakedditClassification(torch.nn.Module):
    def __init__(self, num_class, dropout, pretrained_weight, hidden_size=512, use_image=False):
        super(HANForFakedditClassification, self).__init__()

        self.num_class = num_class

        self.weight = pretrained_weight
        self.vocab_size = self.weight.shape[0]
        self.embed_size = self.weight.shape[1]
        self.hidden_size = hidden_size
        self.word_embedding = torch.nn.Embedding(self.vocab_size, self.embed_size)
        self.blstm = torch.nn.LSTM(input_size=self.embed_size,
                                   hidden_size=self.embed_size,
                                   bias=True,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=0.3)

        self.num_heads = [8, 8]
        self.han = HAN(2, self.embed_size, self.hidden_size, self.num_class, self.num_heads, dropout)

        self.use_image = use_image
        if self.use_image:
            self.vgg = torchvision.models.vgg19_bn(pretrained=True).features
            self.img_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(0.4),
                torch.nn.Linear(4096, self.hidden_size)
            )
            self.classifier = nn.Linear(self.hidden_size * (self.num_heads[-1] + 1), self.num_class)
        else:
            self.classifier = nn.Linear(self.hidden_size * self.num_heads[-1], self.num_class)

        self.__init_weights__()

    def __init_weights__(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        if self.use_image:
            torch.nn.init.xavier_uniform_(self.img_lin[0].weight)
            torch.nn.init.xavier_uniform_(self.img_lin[3].weight)

    def forward(self, input):
        node_features, comment_to_data_edge_index, data_to_comment_edge_index, batch, labels = \
            input.x, input.comment_to_data_edge_index, input.data_to_comment_edge_index, input.batch, input.y

        text_embedding = self.word_embedding(node_features[:, 0, :])
        lengths = node_features[:, 1, :].sum(dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            text_embedding, lengths, enforce_sorted=False, batch_first=True
        )

        _, [h_n, _] = self.blstm(packed)
        node_features = h_n.mean(dim=0)

        node_features = self.han([comment_to_data_edge_index, data_to_comment_edge_index], node_features)
        mean_node = global_mean_pool(node_features, batch)
        if self.use_image:
            vgg_output = self.vgg(torch.stack(torch.split(input.img_features, 3, dim=0), dim=0))

            vgg_output = vgg_output.view(vgg_output.size(0), -1)
            image_encode = self.img_lin(vgg_output)
            mean_node = torch.cat([mean_node, image_encode], -1)
        logits = self.classifier(mean_node)
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


class HANForWeiboClassification(torch.nn.Module):
    def __init__(self, num_class, dropout, pretrained_weight, hidden_size=512, use_image=False,
                 edge_mask=[1, 1, 1, 1, 1], layer=1, use_bert=False, bert_path=None, finetune_bert=False):
        super(HANForWeiboClassification, self).__init__()

        self.num_class = num_class

        self.weight = pretrained_weight
        self.hidden_size = hidden_size
        self.use_bert = use_bert
        if self.use_bert:
            self.bert = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_path
            )
            self.embed_size = 768
            self.finetune_bert = finetune_bert
        else:
            self.vocab_size = self.weight.shape[0]
            self.embed_size = self.weight.shape[1]
            self.word_embedding = torch.nn.Embedding(self.vocab_size, self.embed_size)
            self.word_embedding.weight.data.copy_(self.weight)

        self.blstm = torch.nn.LSTM(input_size=self.embed_size,
                                   hidden_size=self.hidden_size,
                                   bias=True,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)
        self.profile_feature_size = 256

        self.comment_profile_embedding = torch.nn.Parameter(torch.Tensor(self.profile_feature_size, self.hidden_size))
        self.data_profile_embedding = torch.nn.Parameter(torch.Tensor(self.profile_feature_size, self.hidden_size))

        self.num_heads = [4] * np.sum(edge_mask)
        self.edge_types = np.sum(edge_mask)
        self.edge_mask = edge_mask
        self.han = nn.ModuleList()

        self.han.append(
            HAN(self.edge_types, self.hidden_size, self.hidden_size, self.num_class, self.num_heads, dropout))
        for l in range(1, layer):
            self.han.append(
                HAN(self.edge_types, self.hidden_size * self.num_heads[- 1], self.hidden_size, self.num_class,
                    self.num_heads, dropout))

        self.use_image = use_image
        if self.use_image:
            self.vgg = torchvision.models.vgg19_bn(pretrained=True).features
            self.img_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(0.4),
                torch.nn.Linear(4096, self.hidden_size)
            )
            self.classifier = nn.Linear(self.hidden_size * (self.num_heads[-1] + 1), self.num_class)
        else:
            self.classifier = nn.Linear(self.hidden_size * self.num_heads[-1], self.num_class)

        self.__init_weights__()

    def __init_weights__(self):
        torch.nn.init.xavier_uniform_(self.comment_profile_embedding)
        torch.nn.init.xavier_uniform_(self.data_profile_embedding)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        if self.use_image:
            torch.nn.init.xavier_uniform_(self.img_lin[0].weight)
            torch.nn.init.xavier_uniform_(self.img_lin[3].weight)

    def forward(self, input):
        node_features, batch, labels = input.x, input.batch, input.y
        # edge_index_comment_profile_to_comment = input.edge_index_comment_profile_to_comment
        # edge_index_comment_to_data = input.edge_index_comment_to_data
        # edge_index_data_to_comment = input.edge_index_data_to_comment
        # edge_index_data_profile_to_data = input.edge_index_data_profile_to_data
        # edge_index_comment_to_comment = input.edge_index_comment_to_comment
        edge_indexes = [input.edge_index_comment_to_data, input.edge_index_data_to_comment,
                        input.edge_index_comment_profile_to_comment, input.edge_index_comment_to_comment,
                        input.edge_index_data_profile_to_data]
        node_type = input.node_type

        encode_node_features = []
        for i in range(node_features.shape[0]):
            node_feature = node_features[i]
            if node_type[i].item() == 0 or node_type[i].item() == 2:

                if self.use_bert:
                    if self.finetune_bert:
                        text_embedding = self.bert(input_ids=node_feature[0].unsqueeze(0),
                                                   attention_mask=node_feature[1].unsqueeze(0))[0]
                    else:
                        with torch.no_grad():
                            text_embedding = self.bert(input_ids=node_feature[0].unsqueeze(0),
                                                       attention_mask=node_feature[1].unsqueeze(0))[0]
                else:
                    # node_features[0] : max_seq_len
                    # text_embedding : max_seq_len, embed_size
                    text_embedding = self.word_embedding(node_feature[0]).unsqueeze(0)
                lengths: List[int] = [int(node_feature[1].sum().item())]
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    text_embedding, lengths, enforce_sorted=False, batch_first=True
                )

                _, [h_n, _] = self.blstm(packed)
                encode_feature = h_n.mean(dim=0).squeeze(0)

            elif node_type[i].item() == 1:
                encode_feature = torch.matmul(torch.t(node_feature[0]).float(), self.data_profile_embedding)
            else:
                encode_feature = torch.matmul(torch.t(node_feature[0]).float(), self.comment_profile_embedding)
            encode_node_features.append(encode_feature)

        node_features = torch.stack(encode_node_features, dim=0)

        for layer in self.han:
            node_features = layer([edge_indexes[i] for i in range(len(edge_indexes)) if self.edge_mask[i] == 1],
                                  node_features)

        mean_node = global_mean_pool(node_features, batch)

        if self.use_image:
            vgg_output = self.vgg(torch.stack(torch.split(input.img_features, 3, dim=0), dim=0))

            vgg_output = vgg_output.view(vgg_output.size(0), -1)
            image_encode = self.img_lin(vgg_output)
            mean_node = torch.cat([mean_node, image_encode], -1)

        logits = self.classifier(mean_node)

        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


class BertForWeiboClassification(torch.nn.Module):
    def __init__(self, num_class, dropout, bert_path=None, finetune_bert=False, concat=False):
        super(BertForWeiboClassification, self).__init__()

        self.num_class = num_class

        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=bert_path
        )
        self.embed_size = 768
        self.finetune_bert = finetune_bert
        self.concat = concat

        self.classifier = nn.Linear(768, self.num_class)

        self.max_sequence_length = 512

    def forward(self, input):
        node_features, node_type, batch, labels = input.x, input.node_type, input.batch, input.y
        batch_size = input.num_graphs
        classify_features = []
        for j in range(batch_size):
            split_node_features = torch.stack([node_features[i] for i in range(batch.shape[0]) if
                                               batch[i] == j and (node_type[i] == 0 or node_type[i] == 2)], dim=0)
            input_ids = split_node_features[:, 0, :]
            attention_mask = split_node_features[:, 1, :]
            if self.concat:
                input_ids = input_ids.masked_select(attention_mask.byte())
                if input_ids.shape[0] > self.max_sequence_length:
                    input_ids = input_ids[:self.max_sequence_length]
                    attention_mask = torch.ones_like(input_ids).long()
                else:
                    attention_mask = torch.ones_like(input_ids).long()
                    padding_ids = torch.zeros(self.max_sequence_length - input_ids.shape[0],
                                              device=attention_mask.device).long()
                    input_ids = torch.cat([input_ids, padding_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, padding_ids], dim=0)
                if self.finetune_bert:
                    split_node_features = self.bert(input_ids=input_ids.unsqueeze(0),
                                                    attention_mask=attention_mask.unsqueeze(0))[1]
                else:
                    with torch.no_grad():
                        split_node_features = self.bert(input_ids=input_ids.unsqueeze(0),
                                                        attention_mask=attention_mask.unsqueeze(0))[1]
            else:
                if self.finetune_bert:
                    split_node_features = self.bert(input_ids=input_ids,
                                                    attention_mask=attention_mask)[1]
                else:
                    with torch.no_grad():
                        split_node_features = self.bert(input_ids=input_ids,
                                                        attention_mask=attention_mask)[1]
            classify_features.append(split_node_features.mean(dim=0))

        classify_features = torch.stack(classify_features, dim=0)

        logits = self.classifier(classify_features)

        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs

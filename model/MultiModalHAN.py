from typing import List

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch_geometric import nn

profile_feature_size = 256
embed_size = 300
vocab_size = 195203
hidden_size = 512


class MultimodalAttention(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(MultimodalAttention, self).__init__()

        self.project = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)


class MultiModalHanLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(MultiModalHanLayer, self).__init__()
        self.comment_profile_to_comment_gcn = nn.GATConv(embed_size, embed_size)
        self.data_profile_to_data_gcn = nn.GATConv(embed_size, embed_size)
        self.comment_to_data_gcn = nn.GATConv(embed_size, embed_size)
        self.multimodalAttention = MultimodalAttention(embed_size, embed_size)

    def forward(self, encode_node_features,
                edge_index_comment_profile_to_comment,
                edge_index_data_profile_to_data,
                edge_index_comment_to_data):
        node_features_1 = self.comment_profile_to_comment_gcn(encode_node_features,
                                                              edge_index_comment_profile_to_comment)
        node_features_2 = self.data_profile_to_data_gcn(encode_node_features,
                                                        edge_index_data_profile_to_data)
        node_features_3 = self.comment_to_data_gcn(encode_node_features,
                                                   edge_index_comment_to_data)
        encode_node_features = torch.stack([node_features_1, node_features_2, node_features_3], dim=1)
        encode_node_features = self.multimodalAttention(encode_node_features)
        return encode_node_features


class MultiModalHAN(torch.nn.Module):
    def __init__(self,
                 pretrained_weight,
                 num_class=2,
                 num_gcn_layers=2,
                 bert_trainable=False,
                 vgg_trainable=True):
        super(MultiModalHAN, self).__init__()
        self.num_class = num_class
        self.weight = pretrained_weight
        self.word_embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.blstm = torch.nn.LSTM(input_size=embed_size,
                                   hidden_size=embed_size,
                                   bias=True,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=0.3)
        self.vgg_trainable = vgg_trainable
        self.num_gcn_layers = num_gcn_layers

        self.comment_profile_embedding = torch.nn.Parameter(torch.Tensor(profile_feature_size, embed_size))
        self.data_profile_embedding = torch.nn.Parameter(torch.Tensor(profile_feature_size, embed_size))

        self.gcn_layer1 = MultiModalHanLayer(embed_size)
        self.gcn_layer2 = MultiModalHanLayer(embed_size)

        self.vgg = torchvision.models.vgg19_bn(pretrained=True).features
        self.img_lin = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(4096, embed_size)
        )

        self._init_parameters_()

        self.blstm.flatten_parameters()

    def _init_parameters_(self):
        self.word_embedding.weight.data.copy_(self.weight)
        torch.nn.init.xavier_uniform_(self.comment_profile_embedding)
        torch.nn.init.xavier_uniform_(self.data_profile_embedding)
        torch.nn.init.xavier_uniform_(self.img_lin[0].weight)
        torch.nn.init.xavier_uniform_(self.img_lin[3].weight)

    def forward(self, data):
        node_features = data.x
        node_type = data.node_type
        edge_index_comment_profile_to_comment = data.edge_index_comment_profile_to_comment
        edge_index_comment_to_data = data.edge_index_comment_to_data
        edge_index_data_profile_to_data = data.edge_index_data_profile_to_data
        img_features = torch.stack(torch.split(data.img_features, 3, dim=0), dim=0)

        encode_node_features = []
        for i in range(node_features.shape[0]):
            node_feature = node_features[i]
            if node_type[i].item() == 0 or node_type[i].item() == 2:
                # node_features[0] : max_seq_len
                # text_embedding : max_seq_len, embed_size
                text_embedding = self.word_embedding(node_feature[0]).unsqueeze(0)
                lengths: List[int] = [int(node_feature[1].sum().item())]
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    text_embedding, lengths, enforce_sorted=False, batch_first=True
                )

                lstm_out, [h_n, c_n] = self.blstm(packed)
                encode_feature = h_n.mean(dim=0).squeeze(0)

            elif node_type[i].item() == 1:
                encode_feature = torch.matmul(torch.t(node_feature[0]).float(), self.data_profile_embedding)
            else:
                encode_feature = torch.matmul(torch.t(node_feature[0]).float(), self.comment_profile_embedding)
            encode_node_features.append(encode_feature)

        encode_node_features = torch.stack(encode_node_features, dim=0)

        encode_node_features = self.gcn_layer1(encode_node_features, edge_index_comment_profile_to_comment,
                                               edge_index_data_profile_to_data, edge_index_comment_to_data)
        encode_node_features = self.gcn_layer2(encode_node_features, edge_index_comment_profile_to_comment,
                                               edge_index_data_profile_to_data, edge_index_comment_to_data)

        # if self.vgg_trainable:
        #     vgg_output = self.vgg(img_features)
        # else:
        #     with torch.no_grad():
        #         vgg_output = self.vgg(img_features)
        #
        # vgg_output = vgg_output.view(vgg_output.size(0), -1)

        # image_encode = self.img_lin(vgg_output)

        from torch_geometric.nn import global_mean_pool
        mean_node = global_mean_pool(encode_node_features, data.batch)

        # return torch.cat([mean_node, image_encode], -1)
        return mean_node


class MultiModalHANClassification(torch.nn.Module):
    def __init__(self, num_class, pretrained_weight):
        super(MultiModalHANClassification, self).__init__()
        self.num_class = num_class
        self.multiModelHAN = MultiModalHAN(num_class=num_class, pretrained_weight=pretrained_weight)
        self.classifier = torch.nn.Linear(embed_size, num_class)

    def forward(self, input):
        outputs = self.multiModelHAN(input)
        logits = self.classifier(outputs)
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        labels = input.y
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


if __name__ == '__main__':
    pass

    # BERT_PATH = '/sdd/yujunshuai/model/chinese_L-12_H-768_A-12'
    # dataset = WeiboGraphDataset('/sdd/yujunshuai/data/weibo/', )
    #
    # train_loader = DataListLoader(dataset, batch_size=4, shuffle=True)
    # model = DataParallel(MultiModalHANClassification(num_class=2, bert_path=BERT_PATH))
    #
    # model = model.cuda()
    #
    # for data in train_loader:
    #     outputs = model(data)
    #     print(outputs)

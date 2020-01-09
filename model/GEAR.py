import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import BertModel

from model.HAN import SemanticAttention


class GEAR(nn.Module):
    def __init__(self,
                 num_class=2,
                 hidden_size=512,
                 bert_path='/home/yujunshuai/model/chinese_L-12_H-768_A-12',
                 use_bert=False,
                 bert_trainable=False,
                 pretraining_weight=None):
        super(GEAR, self).__init__()

        self.num_class = num_class
        self.hidden_size = hidden_size
        self.use_bert = use_bert
        self.bert_trainable = bert_trainable

        if use_bert:
            self.bert = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_path
            )
            self.embed_size = 768
        else:
            self.weight = pretraining_weight
            self.vocab_size = self.weight.shape[0]
            self.embed_size = self.weight.shape[1]
            self.word_embedding = torch.nn.Embedding(self.vocab_size, self.embed_size)
            self.word_embedding.weight.data.copy_(self.weight)
            self.blstm = torch.nn.LSTM(input_size=self.embed_size,
                                       hidden_size=self.embed_size,
                                       bias=True,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True,
                                       dropout=0.3)

        num_head = [8, 8, 8]
        self.conv1 = GATConv(self.embed_size, self.hidden_size, heads=num_head[0])
        self.conv2 = GATConv(self.hidden_size, self.hidden_size, heads=num_head[0])
        self.conv3 = GATConv(self.hidden_size, self.hidden_size, heads=num_head[0])

        self.pool = global_mean_pool

        self.classifier = nn.Linear(self.hidden_size * num_head[-1], self.num_class)

        self.semantic_attention = SemanticAttention(in_size=hidden_size * num_head[-1])

        self.__init_weights__()

    def __init_weights__(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, data):
        # node_features : (node_count, 2, sequence_length)
        node_features, edge_index, batch, labels = data.x, data.edge_index, data.batch, data.y
        input_ids = node_features[:, 0, :]
        attention_mask = node_features[:, 1, :]
        if self.use_bert:
            if self.bert_trainable:
                node_features = self.bert(input_ids=input_ids,
                                          attention_mask=attention_mask)[1]
            else:
                with torch.no_grad():
                    node_features = self.bert(input_ids=input_ids,
                                              attention_mask=attention_mask)[1]
        else:
            text_embedding = self.word_embedding(input_ids)
            lengths = attention_mask.sum(dim=-1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                text_embedding, lengths, enforce_sorted=False, batch_first=True
            )
            _, [h_n, _] = self.blstm(packed)
            node_features = h_n.mean(dim=0)

        semantic_embeddings = []
        semantic_embeddings.append(F.relu(self.conv1(node_features, edge_index).flatten(1)))
        semantic_embeddings.append(F.relu(self.conv2(node_features, edge_index).flatten(1)))
        semantic_embeddings.append(F.relu(self.conv3(node_features, edge_index).flatten(1)))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        semantic_embeddings = self.semantic_attention(semantic_embeddings)

        pooled_features = self.pool(semantic_embeddings, batch)

        logits = self.classifier(pooled_features)
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs


if __name__ == '__main__':
    pass

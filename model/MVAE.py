from typing import List

import torchvision

import torch
from torch import nn
from transformers import BertModel

text_embed_size = 512
text_features_size = 256
image_features_size = 256
fusion_size = 512


class MVAE(nn.Module):
    def __init__(self, bert_path, bert_trainable, vgg_trainable):
        super(MVAE, self).__init__()

        self.bert_trainable = bert_trainable
        self.bert_path = bert_path
        self.vgg_trainable = vgg_trainable

        # Encoder
        # Text
        self.en_text_embedding = BertModel(bert_path)
        self.en_blstm_1 = nn.LSTM(768, text_embed_size, bidirectional=True, dropout=0.3, )
        self.en_blstm_2 = nn.LSTM(text_embed_size, text_embed_size, bidirectional=True, dropout=0.3)
        self.en_text_fc = nn.Linear(text_embed_size, text_features_size)
        # Image
        self.en_image_embedding = torchvision.models.vgg19_bn(pretrained=True).features
        self.en_img_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, image_features_size),
        )

        self.fusion_fc = nn.Linear(text_features_size + image_features_size, fusion_size)
        self.fusion_mean_fc = nn.Linear(fusion_size, fusion_size)
        self.fusion_var_fc = nn.Linear(fusion_size, fusion_size)

        # Decoder
        # Text
        self.de_text_fc = nn.Linear(fusion_size, text_embed_size)
        self.de_blstm_1 = nn.LSTM(text_embed_size, text_embed_size, bidirectional=True, dropout=0.3)
        self.de_blstm_2 = nn.LSTM(text_embed_size, text_embed_size, bidirectional=True, dropout=0.3)
        self.de_word_decoding = nn.Linear(text_embed_size, self.en_text_embedding.config.vocab_size)
        # Image
        self.de_image_fc = nn.Sequential(
            nn.Linear(fusion_size, image_features_size),
            nn.Linear(image_features_size, 512 * 7 * 7)
        )

    def forward(self, input):
        input_ids, attention_masks, img_tensors, labels = input[0], input[1], input[2], input[3]
        if self.bert_trainable:
            text_embedding = self.bert(input_ids, attention_masks)[0]
        else:
            with torch.no_grad():
                text_embedding = self.bert(input_ids, attention_masks)[0]
        lengths: List[int] = [int(attention_masks[i].sum().item()) for i in range(attention_masks.shape[0])]
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            text_embedding, lengths, enforce_sorted=False, batch_first=True
        )

        lstm_out, [h_n, c_n] = self.lstm(packed)
        h_n = h_n.mean(dim=0)




        if self.vgg_trainable:
            img_features = self.vgg(img_tensors)
        else:
            with torch.no_grad():
                img_features = self.vgg(img_tensors)


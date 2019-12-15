from torch import nn
from transformers import BertForSequenceClassification


class TomBert(nn.Module):
    def __init__(self, bert_path):
        super(TomBert, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_path)


    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data[0], data[1], data[2], data[3]

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            labels=labels)
        return outputs

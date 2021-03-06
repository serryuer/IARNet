import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class FakedditDataset(Dataset):
    def __init__(self, root, bert_path, max_sequence_length, num_class, isVal=False, isTest=False):
        self.isVal = isVal
        self.isTest = isTest
        self.root = root
        self.bert_path = bert_path
        self.max_sequence_length = max_sequence_length
        self.num_class = num_class

        if self.isVal:
            self.data = pd.read_csv(os.path.join(root, 'validate.tsv'), sep='\t')
        elif self.isTest:
            self.data = pd.read_csv(os.path.join(root, 'test.tsv'), sep='\t')
        else:
            self.data = pd.read_csv(os.path.join(root, 'train.tsv'), sep='\t')

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
                self,
                tokens,
                input_ids,
                input_mask,
                segment_ids,
        ):
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids

    def convert_sentence_to_features(self, sentence):
        max_sequence_length = self.max_sequence_length + 2
        tokenize_result = self.tokenizer.tokenize(sentence)

        # truncate sequences pair
        while len(tokenize_result) > self.max_sequence_length:
            tokenize_result.pop()

        tokens = []
        tokens.append('[CLS]')
        segment_ids = []
        segment_ids.append(0)

        # add tokens in tokens_1
        for token in tokenize_result:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length
        assert len(segment_ids) == max_sequence_length

        return FakedditDataset.BertInputFeatures(
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )

    def __getitem__(self, item):
        line = self.data.loc[item]
        title = str(line['clean_title'])
        features = self.convert_sentence_to_features(title)
        input_ids = torch.LongTensor(features.input_ids)
        input_mask = torch.LongTensor(features.input_mask)
        segment_ids = torch.LongTensor(features.segment_ids)

        # if self.isTest:
        #     return input_ids, input_mask, segment_ids

        if self.num_class == 2:
            label = int(line['2_way_label'])
        elif self.num_class == 3:
            label = int(line['3_way_label'])
        else:
            label = int(line['5_way_label'])

        label = torch.tensor([label])
        return input_ids, input_mask, segment_ids, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = FakedditDataset('/home/yujunshuai/data/Fakeddit/fakeddit_v1.0')
    print(f'dataset length : {len(dataset)}')
    print(dataset[0])

import os
from typing import List, Dict

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils.ltp_parsing import LtpParsing
from data_utils.spacy_parsing import SpacyParsing


class FakedditGraphDataset(InMemoryDataset):
    def __init__(self, root, num_class=2, transform=None, pre_transform=None, isVal=False, isTest=False,
                 max_sequence_length=256,
                 bert_path='/home/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'):
        self.isTest = isTest
        self.isVal = isVal
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.root = root
        self.num_class = num_class

        super(FakedditGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.isTest:
            return ['gear_graph_test.dataset']
        elif self.isVal:
            return ['gear_graph_val.dataset']
        else:
            return ['gear_graph_train.dataset']

    def download(self):
        pass

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

        tokens = ['[CLS]']
        for token in tokenize_result:
            tokens.append(token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length

        return [input_ids, input_mask]

    def process(self):
        # Read data_utils into huge `Data` list.
        data_list = []

        if self.isTest:
            data = pd.read_csv(os.path.join(self.root, 'test_with_image_comments.csv'),
                               sep='\t')
        elif self.isVal:
            data = pd.read_csv(os.path.join(self.root, 'validate_with_image_comments.csv'),
                               sep='\t')
        else:
            data = pd.read_csv(os.path.join(self.root, 'train_with_image_comments.csv'),
                               sep='\t')


        for index, row in tqdm(data.iterrows()):
            clean_title = str(row['clean_title'])

            comments = list(str(row['comments']).split('\t'))[:5]

            title_feature = self.convert_sentence_to_features(clean_title)
            comments_features = [self.convert_sentence_to_features(clean_title + str(comment)) for comment in comments]

            node_features = []
            source_nodes = []
            target_nodes = []

            node_features.append(title_feature)
            node_features.extend(comments_features)

            node_count = len(comments_features) + 1
            for i in range(1, node_count):
                source_nodes.append(0)
                target_nodes.append(i)

            node_features = torch.LongTensor(node_features)
            edge_index = torch.LongTensor([source_nodes, target_nodes])

            if self.num_class == 2:
                label = row['2_way_label']
            elif self.num_class == 3:
                label = row['3_way_label']
            elif self.num_class == 5:
                label = row['5_way_label']

            data = Data(x=node_features, edge_index=edge_index, y=torch.LongTensor([label]))

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = FakedditGEARDataset('~/data/Fakeddit/fakeddit_v1.0', isTest=True)
    print(len(dataset))

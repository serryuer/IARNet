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


class FakedditSRLDataset(InMemoryDataset):
    def __init__(self, root, language='zh', transform=None, pre_transform=None, isVal=False, isTest=False,
                 max_sequence_length=160, bert_model_or_path='/home/yujunshuai/model/chinese_wwm_ext_pytorch'):
        self.isTest = isTest
        self.isVal = isVal
        if language == 'zh':
            self.dependency_parsing = LtpParsing()
        else:
            self.dependency_parsing = SpacyParsing()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)

        super(FakedditSRLDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.isTest:
            return ['test.dataset']
        elif self.isVal:
            return ['validate.dataset']
        else:
            return ['train.dataset']

    def download(self):
        pass



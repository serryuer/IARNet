import json
import os
import random
import re
from itertools import repeat

import jieba
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset
from torchvision import transforms as T
from tqdm import tqdm
from random import random
from random import randint
import numpy as np
import torch
from transformers import BertTokenizer
import preprocessor as p


class ValueFeatureOneHotEncoder(object):
    def __init__(self, segment_schema):
        self.segment_schema = segment_schema

    def transform(self, value):
        value = value.item()
        one_hot_value = np.array([0] * (len(self.segment_schema) + 1))
        for i in reversed(range(len(self.segment_schema))):
            if value >= self.segment_schema[i]:
                one_hot_value[i] = 1
                return one_hot_value
        one_hot_value[len(self.segment_schema)] = 1
        return one_hot_value


class TWGraphDataset(InMemoryDataset):
    def __init__(self, root, w2id, step, tw_name,
                 transform=None,
                 pre_transform=None,
                 data_max_sequence_length=156,
                 comment_max_sequence_length=156,
                 feature_dim=156,
                 max_comment_num=10,
                 restart_prob=0.5,
                 delay=1000,
                 tokenizer=None):
        self.tw_name = tw_name
        self.w2id = w2id
        self.step = step
        self.feature_dim = feature_dim
        self.data_max_sequence_length = data_max_sequence_length
        self.comment_max_sequence_length = comment_max_sequence_length
        self.max_comment_num = max_comment_num
        self.root = root
        self.restart_prob = restart_prob
        self.delay = delay
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.use_bert = True
        else:
            self.use_bert = False
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.transforms = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

        super(TWGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.use_bert:
            return [f'{self.tw_name}_bert_{self.restart_prob}_{self.delay}_{self.step}.dataset']
        else:
            return [f'{self.tw_name}_glove_{self.restart_prob}_{self.delay}_{self.step}.dataset']

    def download(self):
        pass

    class Comment(object):
        def __init__(self, comment_text, comment_feature):
            self.comment_text = comment_text
            self.comment_feature = comment_feature

    class CommentNode(object):
        def __init__(self, comment):
            self.comment = comment
            self.children = []

    def convert_sentence_to_features(self, sentence, max_sequence_length):
        if self.tokenizer is not None:
            words = self.tokenizer.tokenize(sentence)
            tokens = ['[CLS]']
            tokens.extend(words)
            tokens = tokens[:max_sequence_length - 1]
            tokens.append('[SEP]')
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            words = sentence.split(' ')
            words = words[:max_sequence_length]
            input_ids = []
            for word in words:
                if word in self.w2id:
                    input_ids.append(self.w2id[word])
                else:
                    input_ids.append(self.unkown_idx)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length

        return [input_ids, input_mask]

    def get_padding_features(self, features, max_sequence_length):
        input_mask = [1] * len(features)
        input_ids = features
        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length

        return [input_ids, input_mask]

    def get_data_features(self, data):
        encoder = {}
        encoder['retweet_count'] = ValueFeatureOneHotEncoder(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
             900, 1000])
        encoder['favorite_count'] = ValueFeatureOneHotEncoder(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
             900, 1000])
        all_feature_one_hot = []
        if data.get('geo', None) == None:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('coordinates', None) == None:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('place', None) == None:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('contributors', None) == None:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('is_quote_status', False) == True:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('favorited', False) == True:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if data.get('retweeted', False) == True:
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])

        for item in encoder.items():
            all_feature_one_hot.extend(item[1].transform(np.array([[int(data[item[0]])]])).tolist())

        return np.array(self.get_padding_features(all_feature_one_hot, self.feature_dim))

    def get_comment_features(self, comment):
        encoder = {}
        encoder['followers_count'] = ValueFeatureOneHotEncoder(
            [0, 5, 10, 20, 40, 75, 100, 200, 300, 400, 500, 700, 900, 1000, 2000, 3000, 5000, 10000, 20000, 30000,
             50000])
        encoder['friends_count'] = ValueFeatureOneHotEncoder([0, 50, 100, 150, 200, 300, 400, 1000, 3000])
        encoder['listed_count'] = ValueFeatureOneHotEncoder(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 300, 500, 1000, 10000])
        encoder['favourites_count'] = ValueFeatureOneHotEncoder(
            [0, 10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000, 50000, 100000, 500000, 1000000,
             5000000, 10000000, 100000000])
        encoder['statuses_count'] = ValueFeatureOneHotEncoder(
            [0, 10, 50, 100, 200, 300, 400, 500, 700, 900, 1000, 2000, 3000, 5000, 10000])
        all_feature_one_hot = []
        for item in encoder.items():
            all_feature_one_hot.extend(item[1].transform(np.array([[int(comment[item[0]])]])).tolist())
        if comment.get('protected', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('utc_offset', None):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('time_zone', None):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('geo_enabled', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('verified', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('contributors_enabled', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('is_translator', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('is_translation_enabled', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('profile_background_tile', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('profile_use_background_image', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('has_extended_profile', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('default_profile', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('default_profile_image', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('following', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('follow_request_sent', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        if comment.get('notifications', False):
            all_feature_one_hot.extend([0, 1])
        else:
            all_feature_one_hot.extend([1, 0])
        return np.array(self.get_padding_features(all_feature_one_hot, self.feature_dim))

    def preprocessing_text(self, text):
        text = p.clean(text).strip()
        if text == '':
            return None
        return text

    def process(self):
        # Read data_utils into huge `Data` list.
        data_list = []
        pbar = tqdm(total=1491)
        with open(os.path.join(self.root, 'source_tweets.txt')) as source_content:
            with open(os.path.join(self.root, 'label.txt')) as all_samples:
                sample = all_samples.readline()
                while sample != '':
                    pbar.update(1)
                    try:
                        sample_file = sample[sample.find(':') + 1:-1]
                        sample_label = sample[:sample.find(':')]
                        if sample_label == 'unverified':
                            sample_label = 0
                        elif sample_label == 'non-rumor':
                            sample_label = 1
                        elif sample_label == 'true':
                            sample_label = 2
                        elif sample_label == 'false':
                            sample_label = 3
                        # node type:
                        # 0: root / 1: root feature / 2:comment / 3:comment feature
                        node_type = []
                        source_nodes_comment_profile_to_comment = []
                        target_nodes_comment_profile_to_comment = []
                        source_nodes_comment_to_data = []
                        target_nodes_comment_to_data = []
                        source_nodes_data_to_comment = []
                        target_nodes_data_to_comment = []
                        source_nodes_data_profile_to_data = []
                        target_nodes_data_profile_to_data = []
                        source_nodes_comment_to_comment = []
                        target_nodes_comment_to_comment = []
                        comment_text = []
                        comment_features = []
                        not_top_level_comment_idx = []
                        comment_ids = []
                        comment_parent_ids = []
                        source_id = ''
                        comment_tree = {}
                        with open(os.path.join(self.root, f'tree/{sample_file}.txt')) as sample_reader:
                            line = sample_reader.readline()
                            while line != '':
                                parent_str = line[1:line.find(']')]
                                child_str = line[line.rfind('[') + 1:-1]
                                parent = [s[1:-1] for s in parent_str.split(', ')]
                                child = [s[1:-1] for s in child_str.split(', ')]
                                parent_user_id = parent[0]
                                parent_tw_id = parent[1]
                                parent_delay = float(parent[2])
                                child_user_id = child[0]
                                child_tw_id = child[1]
                                child_delay = float(child[2][:-1])
                                if parent[0] == 'ROOT':
                                    root_node_features = self.convert_sentence_to_features(
                                        source_content.readline().split('\t')[-1],
                                        self.data_max_sequence_length)
                                    node_type.append(0)
                                    root_node_features
                                    source_id = child_tw_id
                                    line = sample_reader.readline()
                                    continue
                                if child_delay > self.delay * 60:
                                    line = sample_reader.readline()
                                    continue
                                if not os.path.exists(os.path.join(self.root, f'message/{child_tw_id}.json')):
                                    line = sample_reader.readline()
                                    continue
                                with open(os.path.join(self.root, f'message/{child_tw_id}.json'), 'r',
                                          encoding='utf-8') as f:
                                    tw = json.load(f)
                                    text = self.preprocessing_text(tw['text'])
                                    if text is None or len(text) <= 2:
                                        line = sample_reader.readline()
                                        continue
                                    comment_text.append(
                                        self.convert_sentence_to_features(text, self.comment_max_sequence_length))
                                    comment_features.append(self.get_comment_features(tw['user']))
                                    if tw['in_reply_to_status_id_str'] != source_id:
                                        not_top_level_comment_idx.append(len(comment_features) - 1)
                                    comment_ids.append(tw['id'])
                                    comment_parent_ids.append(tw['in_reply_to_status_id_str'])

                                line = sample_reader.readline()

                        for idx, comment_feature in enumerate(comment_features):
                            if idx not in not_top_level_comment_idx:
                                comment_tree[str(idx)] = TWGraphDataset.CommentNode(
                                    TWGraphDataset.Comment(comment_text[idx], comment_feature))
                        ready_to_connect_comment_idx = not_top_level_comment_idx
                        current_level_node = {}
                        last_level_node = comment_tree
                        while len(ready_to_connect_comment_idx) != 0:
                            for idx in ready_to_connect_comment_idx:
                                parent_id = comment_parent_ids[idx]
                                for key in last_level_node:
                                    if comment_ids[int(key)] == parent_id:
                                        node = TWGraphDataset.CommentNode(
                                            TWGraphDataset.Comment(comment_text[idx], comment_features[idx]))
                                        last_level_node[key].children.append(node)
                                        current_level_node[idx] = node
                                        not_top_level_comment_idx.remove(idx)
                                        break
                            if len(current_level_node) == 0:
                                for idx in ready_to_connect_comment_idx:
                                    comment_tree[str(idx)] = TWGraphDataset.CommentNode(
                                        TWGraphDataset.Comment(comment_text[idx], comment_features[idx]))
                                break
                            last_level_node = current_level_node
                            current_level_node = {}
                            ready_to_connect_comment_idx = not_top_level_comment_idx

                        current_node_group = list(comment_tree.values())
                        node_features = [root_node_features, ]
                        is_top = True
                        parent_comment_node_idx = -1
                        while (len(node_type) - 1) / 2 <= self.max_comment_num:
                            if len(current_node_group) == 0:
                                break
                            idx = randint(0, len(current_node_group) - 1)
                            node_features.append(current_node_group[idx].comment.comment_text)
                            node_type.append(2)
                            node_features.append(current_node_group[idx].comment.comment_feature)
                            node_type.append(3)
                            source_nodes_comment_profile_to_comment.append(len(node_type) - 1)
                            target_nodes_comment_profile_to_comment.append(len(node_type) - 2)
                            source_nodes_comment_to_data.append(len(node_type) - 2)
                            target_nodes_comment_to_data.append(0)
                            source_nodes_data_to_comment.append(0)
                            target_nodes_data_to_comment.append(len(node_type) - 2)
                            if not is_top:
                                source_nodes_comment_to_comment.append(len(node_type) - 2)
                                target_nodes_comment_to_comment.append(parent_comment_node_idx)
                                target_nodes_comment_to_comment.append(len(node_type) - 2)
                                source_nodes_comment_to_comment.append(parent_comment_node_idx)
                            prob = random()
                            if prob > self.restart_prob:
                                current_node_group = list(comment_tree.values())
                                is_top = True
                            else:
                                current_node_group = current_node_group[idx].children
                                if len(current_node_group) == 0:
                                    current_node_group = list(comment_tree.values())
                                    is_top = True
                                else:
                                    is_top = False
                                    parent_comment_node_idx = len(node_type) - 2

                        node_features = torch.LongTensor(node_features)
                        data = Data(x=node_features,
                                    node_type=torch.LongTensor(node_type),
                                    edge_index_comment_profile_to_comment=
                                    torch.LongTensor([source_nodes_comment_profile_to_comment,
                                                      target_nodes_comment_profile_to_comment]),
                                    edge_index_comment_to_data=
                                    torch.LongTensor([source_nodes_comment_to_data,
                                                      target_nodes_comment_to_data]),
                                    edge_index_data_to_comment=
                                    torch.LongTensor([source_nodes_data_to_comment,
                                                      target_nodes_data_to_comment]),
                                    edge_index_data_profile_to_data=
                                    torch.LongTensor([source_nodes_data_profile_to_data,
                                                      target_nodes_data_profile_to_data]),
                                    edge_index_comment_to_comment=
                                    torch.LongTensor([source_nodes_comment_to_comment,
                                                      target_nodes_comment_to_comment]),
                                    y=torch.LongTensor([sample_label]))
                        data_list.append(data)
                    except BaseException:
                        print(f'parse error for {sample_file}')
                        sample = all_samples.readline()
                        continue
                    sample = all_samples.readline()

        print(f"data length : {len(data_list)}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # return data, slices

    def get_back(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = \
                slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        if data.node_type.view(-1).shape[0] > self.max_comment_num * 2 + 2:
            comment_num = int((data.node_type.view(-1).shape[0] - 2) / 2)
            selected_comment = random.sample(range(2, comment_num + 2), self.max_comment_num)
            selected_node = [0, 1]
            selected_node.extend(selected_comment)
            selected_node.extend([i + comment_num for i in selected_comment])
            data.x = torch.index_select(input=data.x, index=torch.tensor(selected_node), dim=0)

            def reconstruect_edge(max_comment_num):
                source_nodes_comment_profile_to_comment = []
                target_nodes_comment_profile_to_comment = []
                source_nodes_comment_to_data = []
                target_nodes_comment_to_data = []
                source_nodes_data_to_comment = []
                target_nodes_data_to_comment = []

                for i in range(2, max_comment_num + 2):
                    source_nodes_comment_to_data.append(i)
                    target_nodes_comment_to_data.append(0)
                    source_nodes_data_to_comment.append(0)
                    target_nodes_data_to_comment.append(i)
                node_count = 2 + max_comment_num
                for i in range(0, max_comment_num):
                    source_nodes_comment_profile_to_comment.append(node_count + i)
                    target_nodes_comment_profile_to_comment.append(2 + i)
                return (torch.LongTensor([source_nodes_comment_to_data,
                                          target_nodes_comment_to_data]),
                        torch.LongTensor([source_nodes_comment_profile_to_comment,
                                          target_nodes_comment_profile_to_comment]),
                        torch.LongTensor([source_nodes_data_to_comment,
                                          target_nodes_data_to_comment]))

            data.edge_index_comment_to_data, data.edge_index_comment_profile_to_comment, data.edge_index_data_to_comment = \
                reconstruect_edge(self.max_comment_num)
            data.node_type = [0, 1]
            data.node_type.extend([2] * self.max_comment_num)
            data.node_type.extend([3] * self.max_comment_num)
            data.node_type = torch.LongTensor(data.node_type)
        return data


if __name__ == '__main__':
    from model.util import *

    BERT_PATH = '/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'
    weight = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vectors.pt')
    w2id = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vocab.pt')
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    dataset = TWGraphDataset('/sdd/yujunshuai/data/twitter_15_16/twitter15/', step=0, tw_name='tw15',
                             max_comment_num=10,
                             restart_prob=0.6, delay=1000,
                             tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0])

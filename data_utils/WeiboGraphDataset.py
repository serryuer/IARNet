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


class WeiboGraphDataset(InMemoryDataset):
    def __init__(self, root, w2id, step,
                 transform=None,
                 pre_transform=None,
                 data_max_sequence_length=256,
                 comment_max_sequence_length=256,
                 max_comment_num=10,
                 restart_prob=0.5,
                 delay=1000,
                 tokenizer=None):
        self.w2id = w2id
        self.unkown_idx = len(w2id)
        self.step = step
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

        super(WeiboGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.use_bert:
            return [f'weibo_bert_{self.max_comment_num}_{self.restart_prob}_{self.delay}_{self.step}.dataset']
        else:
            return [f'weibo_glove_{self.max_comment_num}_{self.restart_prob}_{self.delay}_{self.step}.dataset']

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

    def prepare_comment_features(self, data_list):
        comment_text = []
        comment_features = []
        not_top_level_comment_idx = []
        comment_ids = []
        comment_parent_ids = []
        for i in range(1, len(data_list)):
            comment = data_list[i]
            if comment['t'] - data_list[0]['t'] > self.delay * 60 * 60:
                continue
            text = self.preprocessing_text(comment['text'])
            if text is None or len(text) <= 2:
                continue
            comment_text.append(self.convert_sentence_to_features(text, self.comment_max_sequence_length))
            comment_features.append(self.get_comment_features(comment))
            if comment['parent'] != data_list[0]['mid']:
                not_top_level_comment_idx.append(len(comment_features) - 1)
            comment_ids.append(comment['mid'])
            comment_parent_ids.append(comment['parent'])
        # construct comment tree
        comment_tree = {}
        for idx, comment_feature in enumerate(comment_features):
            if idx not in not_top_level_comment_idx:
                comment_tree[str(idx)] = WeiboGraphDataset.CommentNode(
                    WeiboGraphDataset.Comment(comment_text[idx], comment_feature))
        ready_to_connect_comment_idx = not_top_level_comment_idx
        current_level_node = {}
        last_level_node = comment_tree
        while len(ready_to_connect_comment_idx) != 0:
            for idx in ready_to_connect_comment_idx:
                parent_id = comment_parent_ids[idx]
                for key in last_level_node:
                    if comment_ids[int(key)] == parent_id:
                        node = WeiboGraphDataset.CommentNode(
                            WeiboGraphDataset.Comment(comment_text[idx], comment_features[idx]))
                        last_level_node[key].children.append(node)
                        current_level_node[idx] = node
                        not_top_level_comment_idx.remove(idx)
                        break
            if len(current_level_node) == 0:
                for idx in ready_to_connect_comment_idx:
                    comment_tree[str(idx)] = WeiboGraphDataset.CommentNode(
                        WeiboGraphDataset.Comment(comment_text[idx], comment_features[idx]))
                break
            last_level_node = current_level_node
            current_level_node = {}
            ready_to_connect_comment_idx = not_top_level_comment_idx
        return list(comment_tree.values())

    def convert_sentence_to_features(self, sentence, max_sequence_length):
        if self.tokenizer is not None:
            words = self.tokenizer.tokenize(sentence)
            tokens = ['[CLS]']
            tokens.extend(words)
            tokens = tokens[:max_sequence_length - 1]
            tokens.append('[SEP]')
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            words = list(jieba.cut(sentence))
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
        encoder['reposts_count'] = ValueFeatureOneHotEncoder([0, 100, 200, 350, 500, 750, 1000, 10000, 500000, 1000000])
        encoder['bi_followers_count'] = ValueFeatureOneHotEncoder(
            [0, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000])
        encoder['city'] = ValueFeatureOneHotEncoder(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 31, 32, 34,
             35, 39, 51, 52, 1000])
        encoder['province'] = ValueFeatureOneHotEncoder(
            [11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 61,
             62, 64, 65, 71, 81, 82, 100, 400])
        encoder['friends_count'] = ValueFeatureOneHotEncoder([0, 100, 200, 300, 400, 500, 700, 900, 1100, 2000, 3000])
        encoder['attitudes_count'] = ValueFeatureOneHotEncoder([0, 1, 2, 3, 10, 50, 100, 300, 500, 1000])
        encoder['followers_count'] = ValueFeatureOneHotEncoder(
            [0, 10, 1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 100000000])
        encoder['verified_type'] = ValueFeatureOneHotEncoder([-1, 0, 1, 2, 3, 4, 5, 6, 7, 10, 200, 220])
        encoder['statuses_count'] = ValueFeatureOneHotEncoder(
            [0, 2500, 5000, 10000, 20000, 40000, 60000, 100000, 200000, 300000, 400000])
        encoder['favourites_count'] = ValueFeatureOneHotEncoder(
            [0, 5, 10, 25, 40, 80, 150, 250, 400, 1000, 10000, 30000])
        encoder['comments_count'] = ValueFeatureOneHotEncoder(
            [0, 10, 30, 50, 100, 150, 300, 1000, 5000, 10000, 20000, 30000])
        all_feature_one_hot = []
        for item in encoder.items():
            all_feature_one_hot.extend(item[1].transform(np.array([[int(data[item[0]])]])).tolist())
        if data['gender'] == 'f':
            all_feature_one_hot.extend([1, 0, 0])
        elif data['gender'] == 'm':
            all_feature_one_hot.extend([0, 1, 0])
        else:
            all_feature_one_hot.extend([0, 0, 1])
        return np.array(self.get_padding_features(all_feature_one_hot, self.data_max_sequence_length))

    def get_comment_features(self, comment):
        encoder = {}
        encoder['bi_followers_count'] = ValueFeatureOneHotEncoder(
            [0, 5, 10, 20, 40, 75, 100, 500, 1000])
        city = [i for i in range(1, 47)]
        city.extend([51, 52, 53, 81, 82, 83, 84, 90, 1000, 2000])
        encoder['city'] = ValueFeatureOneHotEncoder(city)
        encoder['friends_count'] = ValueFeatureOneHotEncoder([0, 50, 100, 150, 200, 300, 400, 1000, 3000])
        encoder['attitudes_count'] = ValueFeatureOneHotEncoder(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 300, 500, 1000, 10000])
        encoder['followers_count'] = ValueFeatureOneHotEncoder(
            [0, 10, 1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 100000000])
        verified_type = [i for i in range(-1, 8)]
        verified_type.extend([10, 200, 202, 300])
        encoder['verified_type'] = ValueFeatureOneHotEncoder(verified_type)
        encoder['statuses_count'] = ValueFeatureOneHotEncoder(
            [0, 100, 200, 500, 1000, 2000, 3000, 5000, 10000])
        encoder['comments_count'] = ValueFeatureOneHotEncoder(
            [0, 1, 3, 4, 5, 10, 20, 100])
        all_feature_one_hot = []
        for item in encoder.items():
            all_feature_one_hot.extend(item[1].transform(np.array([[int(comment[item[0]])]])).tolist())
        return np.array(self.get_padding_features(all_feature_one_hot, self.comment_max_sequence_length))

    def get_image_features(self, data):
        img_url = data['picture']
        img_id = img_url[img_url.rfind('/') + 1:]
        img_path = os.path.join(self.root, f"image/{img_id}")
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data

    def preprocessing_text(self, text):
        if text.startswith('转发微博') and len(text) <= 7:
            return None
        if text.startswith('轉發微博') and len(text) <= 7:
            return None
        if text.startswith('回复'):
            text = text[text.find(':') + 1:]
        re_tag = re.compile('@.+?\s@.+?\s')
        text = re.sub('@.+?\s@.+?\s', '', text)
        text = re.sub('</?\w+[^>]*>', '', text)
        text = re.sub(",+", ",", text)  # 合并逗号
        text = re.sub(" +", " ", text)  # 合并空格
        text = re.sub("[...|…|。。。]+", "...", text)  # 合并句号
        text = re.sub("-+", "--", text)  # 合并-
        text = re.sub("———+", "———", text)  # 合并-
        text = re.sub('[^\u4e00-\u9fa5]', '', text)
        text = text.strip()
        if text == '':
            return None
        return text

    def process(self):
        # Read data_utils into huge `Data` list.
        data_list = []
        pbar = tqdm(total=4665)
        with open(os.path.join(self.root, 'Weibo.txt')) as all_samples:
            sample = all_samples.readline()
            while sample != '':
                sample = all_samples.readline()
                pbar.update(1)
                print()
                try:
                    sample_file = sample[4:sample.find('\t')]
                    sample_label = 0 if sample[sample.find('label') + 6] == '0' else 1
                    with open(os.path.join(self.root, f'Weibo/{sample_file}.json')) as sample_reader:
                        sample_json = json.loads(sample_reader.read())
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
                        if len(sample_json[0]['text']) < 10:
                            continue
                        root_node_features = self.convert_sentence_to_features(sample_json[0]['text'],
                                                                               self.data_max_sequence_length)
                        node_type.append(0)
                        root_feature_node_features = self.get_data_features(sample_json[0])
                        node_type.append(1)
                        # img_features = self.get_image_features(sample_json[0])
                        comment_tree = self.prepare_comment_features(sample_json)

                        current_node_group = comment_tree
                        node_features = [root_node_features, root_feature_node_features]
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
                                current_node_group = comment_tree
                                is_top = True
                            else:
                                current_node_group = current_node_group[idx].children
                                if len(current_node_group) == 0:
                                    current_node_group = comment_tree
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
                        # img_features=img_features)
                        data_list.append(data)
                except BaseException:
                    continue

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

    BERT_PATH = '/sdd/yujunshuai/model/chinese_L-12_H-768_A-12'
    weight = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vectors.pt')
    w2id = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vocab.pt')
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    dataset = WeiboGraphDataset('/sdd/yujunshuai/data/weibo/', w2id, max_comment_num=10, restart_prob=0.6, delay=1000,
                                tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])

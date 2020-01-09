import os
import random
from itertools import repeat

import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset
from torchvision import transforms as T
from tqdm import tqdm
from transformers import BertTokenizer


class FakedditGraphDataset(InMemoryDataset):
    def __init__(self, root, w2id, img_root,
                 num_class=2,
                 transform=None, pre_transform=None,
                 isVal=False, isTest=False,
                 data_max_sequence_length=256, comment_max_sequence_length=256, max_comment_num=30,
                 min_comment_word_length=5,
                 bert_path='/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'):
        self.w2id = w2id
        self.unkown_idx = len(w2id)
        self.isTest = isTest
        self.isVal = isVal
        self.data_max_sequence_length = data_max_sequence_length
        self.comment_max_sequence_length = comment_max_sequence_length
        self.max_comment_num = max_comment_num
        self.root = root
        self.num_class = num_class
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.min_comment_word_length = min_comment_word_length
        self.img_root = img_root

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.transforms = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

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

    def convert_sentence_to_features(self, sentence, max_sequence_length):
        words = self.tokenizer.tokenize(sentence)
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
            try:
                clean_title = str(row['clean_title'])

                comments = list(str(row['comments']).split('\t'))
                comments = [comment for comment in comments if len(comment.split(' ')) >= self.min_comment_word_length]
                if len(comments) < 5:
                    continue

                title_feature = self.convert_sentence_to_features(clean_title, self.data_max_sequence_length)
                comments_features = [self.convert_sentence_to_features(str(comment), self.comment_max_sequence_length)
                                     for comment in comments]

                node_features = []
                comment_to_data_source_nodes = []
                comment_to_data_target_nodes = []
                data_to_comment_source_nodes = []
                data_to_comment_target_nodes = []

                node_features.append(title_feature)
                node_features.extend(comments_features)

                node_count = len(comments_features) + 1
                for i in range(1, node_count):
                    comment_to_data_source_nodes.append(i)
                    comment_to_data_target_nodes.append(0)
                    data_to_comment_source_nodes.append(0)
                    data_to_comment_target_nodes.append(i)

                node_features = torch.LongTensor(node_features)

                if self.num_class == 2:
                    label = row['2_way_label']
                elif self.num_class == 3:
                    label = row['3_way_label']
                elif self.num_class == 5:
                    label = row['5_way_label']

                img_path = os.path.join(self.img_root, row['id'] + '.jpg')
                data = Image.open(img_path).convert('RGB')
                data = self.transforms(data)

                data = Data(x=node_features,
                            comment_to_data_edge_index=torch.LongTensor(
                                [comment_to_data_source_nodes, comment_to_data_target_nodes]),
                            data_to_comment_edge_index=torch.LongTensor(
                                [data_to_comment_source_nodes, data_to_comment_target_nodes]),
                            y=torch.LongTensor([label]), image_tensor=data)

                data_list.append(data)
            except BaseException:
                continue
        print(f"data length : {len(data_list)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = \
                slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        if data.num_nodes - 1 > self.max_comment_num:
            comment_num = data.num_nodes - 1
            selected_comment = random.sample(range(1, comment_num + 1), self.max_comment_num)
            selected_node = [0]
            selected_node.extend(selected_comment)
            data.x = torch.index_select(input=data.x, index=torch.tensor(selected_node), dim=0)

            def reconstruect_edge(max_comment_num):
                comment_to_data_source_nodes = []
                comment_to_data_target_nodes = []
                data_to_comment_source_nodes = []
                data_to_comment_target_nodes = []

                for i in range(1, max_comment_num + 1):
                    comment_to_data_source_nodes.append(i)
                    comment_to_data_target_nodes.append(0)
                    data_to_comment_source_nodes.append(0)
                    data_to_comment_target_nodes.append(i)
                return (torch.LongTensor([comment_to_data_source_nodes,
                                          comment_to_data_target_nodes]),
                        torch.LongTensor([data_to_comment_source_nodes,
                                          data_to_comment_target_nodes]))

            data.comment_to_data_edge_index, data.data_to_comment_edge_index = reconstruect_edge(self.max_comment_num)
        return data


if __name__ == '__main__':
    wi = torch.load('/home/tanghengzhu/yjs/model/en_glove_vector/glove.42B.300d.w2id.pt')
    dataset2 = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0', w2id=wi,
                                    img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/train_image')
    dataset = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0', isTest=True, w2id=wi,
                                   img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/test_image')
    dataset1 = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0', isVal=True, w2id=wi,
                                    img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/validate_image')
    print(dataset[0])

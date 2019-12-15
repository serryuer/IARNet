from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from transformers import BertTokenizer

from torchvision import transforms as T
from PIL import Image

Image.MAX_IMAGE_PIXELS = 10000000000


class MultiModalDataset(Dataset):

    def __init__(self, num_class, text_root, img_root, bert_path, max_sequence_length, isVal=False, isTest=False):
        self.num_class = num_class
        self.text_root = text_root
        self.img_root = img_root
        self.bert_path = bert_path
        self.isTest = isTest
        self.isVal = isVal
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        if self.isTest:
            self.data = pd.read_csv(os.path.join(self.text_root, 'test_with_image.csv'),
                                    sep='\t')
        elif self.isVal:
            self.data = pd.read_csv(os.path.join(self.text_root, 'validate_with_image.csv'),
                                    sep='\t')
        else:
            self.data = pd.read_csv(os.path.join(self.text_root, 'train_with_image.csv'),
                                    sep='\t')

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        if self.isTest or isVal:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

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

        return MultiModalDataset.BertInputFeatures(
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )

    def __getitem__(self, item):
        line = self.data.loc[item]
        while (True):
            try:
                title = str(line['clean_title'])
                features = self.convert_sentence_to_features(title)
                input_ids = torch.LongTensor(features.input_ids)
                input_mask = torch.LongTensor(features.input_mask)

                # if self.isTest:
                #     return input_ids, input_mask, segment_ids

                if self.num_class == 2:
                    label = int(line['2_way_label'])
                elif self.num_class == 3:
                    label = int(line['3_way_label'])
                else:
                    label = int(line['5_way_label'])

                label = torch.tensor([label])

                img_path = os.path.join(self.img_root, line['id'] + '.jpg')
                data = Image.open(img_path).convert('RGB')
                data = self.transforms(data)
            except BaseException:
                import random
                line = self.data.loc[random.randint(1, self.data.shape[0])]
                continue

            return input_ids, input_mask, data, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = MultiModalDataset(num_class=2,
                                text_root='/sdd/yujunshuai/data/Fakeddit/fakeddit_v1.0',
                                img_root='/sdd/yujunshuai/data/Fakeddit/images/train_image',
                                bert_path='/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16',
                                max_sequence_length=256)
    print(f'dataset length : {len(dataset)}')
    print(dataset[0])

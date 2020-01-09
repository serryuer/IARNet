import logging
import os
import random

import torch
from sklearn.metrics import *
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW

from data_utils.FakedditGEARDataset import FakedditGEARDataset
from data_utils.WeiboGraphDataset import WeiboGraphDataset
from model.GEAR import GEAR
from model.MultiModalHAN import MultiModalHANClassification
from trainer import Train
from model.util import load_parallel_save_model, read_vectors
import numpy as np

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.DEBUG, format=C_LogFormat)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

BERT_PATH = '/sdd/yujunshuai/model/chinese_L-12_H-768_A-12'

BATCH_SIZE_PER_GPU = 10
GPU_COUNT = torch.cuda.device_count()

seed = 1024

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    weight = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vectors.pt')
    w2id = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vocab.pt')

    dataset = WeiboGraphDataset('/sdd/yujunshuai/data/weibo/',
                                w2id=w2id,
                                data_max_sequence_length=256,
                                comment_max_sequence_length=256,
                                max_comment_num=50)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT, shuffle=True)
    val_loader = DataListLoader(val_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT), shuffle=True)

    logging.debug(f"train data all steps: {len(train_loader)}, validate data all steps : {len(val_loader)}")

    model = MultiModalHANClassification(num_class=2, pretrained_weight=weight)
    # model = load_parallel_save_model(
    #     '/sdd/yujunshuai/save_model/weibo_multimodal_han/weibo_multimodal_han-0-0.4127604166666667.pt', model)
    model = DataParallel(model)

    model = model.cuda(0)

    # Prepare  optimizer and schedule(linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    crit = torch.nn.CrossEntropyLoss()

    trainer = Train(model_name='weibo_multimodal_han',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=None,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=crit,
                    epochs=100,
                    print_step=1,
                    early_stop_patience=10,
                    save_model_path='/sdd/yujunshuai/save_model/weibo_multimodal_han',
                    save_model_every_epoch=True,
                    metric=accuracy_score,
                    num_class=2,
                    tensorboard_path='/sdd/yujunshuai/tensorboard_log')
    trainer.train()
    # trainer.eval()

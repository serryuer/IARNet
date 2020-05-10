import logging
import random

import torch
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW, BertTokenizer

from data_utils.FakedditGraphDataset import FakedditGraphDataset
from data_utils.WeiboGraphDataset import WeiboGraphDataset
from model.util import load_parallel_save_model
from model.IARNet import IARNetForWeiboClassification
from trainer import Train

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

BERT_PATH = '/sdd/yujunshuai/model/chinese_L-12_H-768_A-12'

# BATCH_SIZE_PER_GPU = 1
GPU_COUNT = torch.cuda.device_count()

seed = 1024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

param_group = [
    # main experiments with bert
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_3_withbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 3,
     'batch_size': 3},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withbert_finetune',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 1,
     'batch_size': 1,
     'fine_tune': True},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withbert_finetune',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 2,
     'batch_size': 1,
     'fine_tune': True},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_3_withbert_finetune',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 3,
     'batch_size': 1,
     'fine_tune': True},

    # main experiments without bert
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_3_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 3,
     'batch_size': 2},

    # ablation study
    # without cs
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert_without_cs',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [0, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withoutbert_without_cs',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 0, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    # without sc
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert_without_sc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 0, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withoutbert_without_sc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 0, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    # without cc
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert_without_cc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 0, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withoutbert_without_cc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 0, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    # without uc
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert_without_uc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 0, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1000_layer_2_withoutbert_without_uc',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 0, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},

    # early detection
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_1_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_4_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 4,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_4_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 4,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_8_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 8,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_8_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 8,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_12_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 12,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_12_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 12,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_24_layer_1_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 24,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 1,
     'batch_size': 10},
    {'model_name': 'weibo_han_comments_10_prob_0.6_delay_24_layer_2_withoutbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 24,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': False,
     'layer': 2,
     'batch_size': 5},
]

weight = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vectors.pt')
w2id = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vocab.pt')

for params in param_group:
    for step in range(5):
        if params.get('use_bert', False):
            tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        else:
            tokenizer = None
        model_name = params['model_name'] + '_step_' + str(step)
        BATCH_SIZE_PER_GPU = params['batch_size']
        dataset = WeiboGraphDataset('/sdd/yujunshuai/data/weibo/', w2id=w2id,
                                    max_comment_num=params['max_comment_num'],
                                    restart_prob=params['restart_prob'], delay=params['delay'], tokenizer=tokenizer,
                                    step=step)

import logging
import random

import torch
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW, BertTokenizer

from data_utils.FakedditGraphDataset import FakedditGraphDataset
from data_utils.TWGraphDataset import TWGraphDataset
from data_utils.WeiboGraphDataset import WeiboGraphDataset
from model.util import load_parallel_save_model
from model.HAN import HANForWeiboClassification, HANForTWClassification
from trainer import Train

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

BERT_PATH = '/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'

# BATCH_SIZE_PER_GPU = 1
GPU_COUNT = torch.cuda.device_count()

seed = 1024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == '__main__':

    param_group = [
        # main experiments with bert
        # {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withbert',
        #  'max_comment_num': 10,
        #  'restart_prob': 0.6,
        #  'delay': 1000,
        #  'edge_mask': [1, 1, 1, 1, 1],
        #  'use_bert': True,
        #  'layer': 1,
        #  'batch_size': 1},
        # {'model_name': 'tw15_han_comments_20_prob_0.6_delay_1000_layer_1_withbert',
        #  'max_comment_num': 20,
        #  'restart_prob': 0.6,
        #  'delay': 1000,
        #  'edge_mask': [1, 1, 1, 1, 1],
        #  'use_bert': True,
        #  'layer': 1,
        #  'batch_size': 1},
        # {'model_name': 'tw15_han_comments_30_prob_0.6_delay_1000_layer_1_withbert',
        #  'max_comment_num': 30,
        #  'restart_prob': 0.6,
        #  'delay': 1000,
        #  'edge_mask': [1, 1, 1, 1, 1],
        #  'use_bert': True,
        #  'layer': 1,
        #  'batch_size': 1},
        # {'model_name': 'tw15_han_comments_40_prob_0.6_delay_1000_layer_1_withbert',
        #  'max_comment_num': 40,
        #  'restart_prob': 0.6,
        #  'delay': 1000,
        #  'edge_mask': [1, 1, 1, 1, 1],
        #  'use_bert': True,
        #  'layer': 1,
        #  'batch_size': 1},
        # {'model_name': 'tw15_han_comments_50_prob_0.6_delay_1000_layer_1_withbert',
        #  'max_comment_num': 50,
        #  'restart_prob': 0.6,
        #  'delay': 1000,
        #  'edge_mask': [1, 1, 1, 1, 1],
        #  'use_bert': True,
        #  'layer': 1,
        #  'batch_size': 1},

        # main experiments without bert
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert',
         'max_comment_num': 20,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert',
         'max_comment_num': 30,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 5},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withoutbert',
         'max_comment_num': 40,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 2},

        # ablation study
        # without cs
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withbert_without_cs',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [0, 1, 1, 1, 1],
         'use_bert': True,
         'layer': 1,
         'batch_size': 3},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_2_withbert_without_cs',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 0, 1, 1, 1],
         'use_bert': True,
         'layer': 2,
         'batch_size': 3},
        # without sc
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withbert_without_sc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 0, 1, 1, 1],
         'use_bert': True,
         'layer': 1,
         'batch_size': 3},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_2_withbert_without_sc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 0, 1, 1, 1],
         'use_bert': True,
         'layer': 2,
         'batch_size': 1},
        # without cc
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withbert_without_cc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 0, 1, 1],
         'use_bert': True,
         'layer': 1,
         'batch_size': 1},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_2_withbert_without_cc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 0, 1, 1],
         'use_bert': True,
         'layer': 2,
         'batch_size': 1},
        # without uc
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_1_withbert_without_uc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 1, 0, 1],
         'use_bert': True,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1000_layer_2_withbert_without_uc',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1000,
         'edge_mask': [1, 1, 1, 0, 1],
         'use_bert': True,
         'layer': 2,
         'batch_size': 5},

        # early detection
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1_layer_1_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_1_layer_2_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 1,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 2,
         'batch_size': 5},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_4_layer_1_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 4,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_4_layer_2_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 4,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 2,
         'batch_size': 5},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_8_layer_1_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 8,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_8_layer_2_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 8,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 2,
         'batch_size': 5},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_12_layer_1_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 12,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_12_layer_2_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 12,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 2,
         'batch_size': 5},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_24_layer_1_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 24,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 1,
         'batch_size': 10},
        {'model_name': 'tw15_han_comments_10_prob_0.6_delay_24_layer_2_withoutbert',
         'max_comment_num': 10,
         'restart_prob': 0.6,
         'delay': 24,
         'edge_mask': [1, 1, 1, 1, 1],
         'use_bert': False,
         'layer': 2,
         'batch_size': 5},
    ]

    for params in param_group:
        for step in range(1):
            if params.get('use_bert', False):
                tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
            else:
                tokenizer = None
            weight = torch.load('/sdd/yujunshuai/model/en_glove_vector/glove.twitter.27B.200d.weight.pt')
            w2id = torch.load('/sdd/yujunshuai/model/en_glove_vector/glove.twitter.27B.200d.vocab.pt')
            model_name = params['model_name'] + '_step_' + str(step)
            BATCH_SIZE_PER_GPU = params['batch_size']
            dataset = TWGraphDataset('/sdd/yujunshuai/data/twitter_15_16/twitter15/', tw_name='tw15', w2id=w2id,
                                     max_comment_num=params['max_comment_num'],
                                     restart_prob=params['restart_prob'], delay=params['delay'], tokenizer=tokenizer,
                                     step=step)
            # dataset = WeiboGraphDataset('/home/tanghengzhu/yjs/data/tw15/', w2id=w2id, max_comment_num=params['max_comment_num'],
            #                             restart_prob=params['restart_prob'], delay=params['delay'])
            train_size = int(0.675 * len(dataset))
            dev_size = int(0.225 * len(dataset))
            test_size = len(dataset) - train_size - dev_size

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                     [train_size, dev_size, test_size])

            train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT, shuffle=True)
            val_loader = DataListLoader(val_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT), shuffle=True)
            test_loader = DataListLoader(test_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT), shuffle=True)

            logging.info(f"train data all steps: {len(train_loader)}, "
                         f"validate data all steps : {len(val_loader)}, "
                         f"test data all steps : {len(test_loader)}")

            model = HANForTWClassification(num_class=4, dropout=0.3, pretrained_weight=weight, use_image=False,
                                           edge_mask=params['edge_mask'], use_bert=params.get('use_bert', False),
                                           bert_path=BERT_PATH, finetune_bert=params.get('fine_tune', False),
                                           layer=params.get('layer', 1))
            # model = load_parallel_save_model('/sdd/yujunshuai/save_model/gear/best-validate-model.pt', model)
            model = DataParallel(model)

            model = model.cuda(0)

            # Prepare  optimizer and schedule(linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
            crit = torch.nn.CrossEntropyLoss()

            trainer = Train(model_name=model_name,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            model=model,
                            optimizer=optimizer,
                            loss_fn=crit,
                            epochs=120,
                            print_step=1,
                            early_stop_patience=3,
                            # save_model_path=f"./save_model/{params['model_name']}",
                            save_model_path=f"/sdd/yujunshuai/save_model/{model_name}",
                            save_model_every_epoch=False,
                            metric=accuracy_score,
                            num_class=4,
                            # tensorboard_path='./tensorboard_log')
                            tensorboard_path='/sdd/yujunshuai/tensorboard_log')
            print(trainer.train())
            print(trainer.test())

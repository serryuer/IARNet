import logging

import torch
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW, BertTokenizer

from data_utils.FakedditGraphDataset import FakedditGraphDataset
from model.IARNet import IARNetForFakedditClassification
from trainer import Train

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

BERT_PATH = '/sdd/yujunshuai/model/bert-base-uncased_L-24_H-1024_A-16'

BATCH_SIZE_PER_GPU = 10
GPU_COUNT = torch.cuda.device_count()

param_group = [
    # main experiments with bert
    {'model_name': 'fakeddit_iarnet_comments_10_prob_0.6_delay_1000_layer_1_withbert',
     'max_comment_num': 10,
     'restart_prob': 0.6,
     'delay': 1000,
     'edge_mask': [1, 1, 1, 1, 1],
     'use_bert': True,
     'layer': 1,
     'batch_size': 15,
     'fixed_count': 100000},
]

if __name__ == '__main__':
    for params in param_group:
        for step in range(2):
            if params.get('use_bert', False):
                tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
            else:
                tokenizer = None
            model_name = params['model_name'] + '_step_' + str(step) + '_data_count_' + \
                         str(params['fixed_count'] if params['fixed_count'] else 0)
            w2id = torch.load('/sdd/yujunshuai/model/en_glove_vector/glove.twitter.27B.200d.vocab.pt')

            weight = torch.load('/sdd/yujunshuai/model/en_glove_vector/glove.twitter.27B.200d.weight.pt')

            train_dataset = FakedditGraphDataset('/sdd/yujunshuai/data/Fakeddit/fakeddit_v1.0/',
                                                 w2id=w2id,
                                                 img_root='/sdd/yujunshuai/data/Fakeddit/images/train_image',
                                                 max_comment_num=params['max_comment_num'],
                                                 fixed_count=params['fixed_count'],
                                                 data_max_sequence_length=256,
                                                 comment_max_sequence_length=256,
                                                 tokenizer=tokenizer)
            test_dataset = FakedditGraphDataset('/sdd/yujunshuai/data/Fakeddit/fakeddit_v1.0/', isTest=True,
                                                w2id=w2id,
                                                img_root='/sdd/yujunshuai/data/Fakeddit/images/test_image',
                                                max_comment_num=params['max_comment_num'],
                                                fixed_count=params['fixed_count'],
                                                data_max_sequence_length=256,
                                                comment_max_sequence_length=256,
                                                tokenizer=tokenizer)
            val_dataset = FakedditGraphDataset('/sdd/yujunshuai/data/Fakeddit/fakeddit_v1.0/', isVal=True,
                                               w2id=w2id,
                                               img_root='/sdd/yujunshuai/data/Fakeddit/images/validate_image',
                                               max_comment_num=params['max_comment_num'],
                                               fixed_count=params['fixed_count'],
                                               data_max_sequence_length=256,
                                               comment_max_sequence_length=256,
                                               tokenizer=tokenizer)

            train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT, shuffle=True)
            test_loader = DataListLoader(test_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15),
                                         shuffle=True)
            val_loader = DataListLoader(val_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15), shuffle=True)

            logging.info(f"train data all steps: {len(train_loader)}, "
                         f"validate data all steps : {len(val_loader)},"
                         f"test data all steps : {len(test_loader)}")

            model = IARNetForFakedditClassification(num_class=2, dropout=0.3, pretrained_weight=weight, use_bert=False)
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
                            epochs=10,
                            print_step=10,
                            early_stop_patience=3,
                            save_model_path=f'/sdd/yujunshuai/save_model/{model_name}',
                            save_model_every_epoch=False,
                            metric=accuracy_score,
                            num_class=2,
                            tensorboard_path='/sdd/yujunshuai/tensorboard_log/fakeddit/')

            print(trainer.train())
            print(f"Testing result :{trainer.test()}")

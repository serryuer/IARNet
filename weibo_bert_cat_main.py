import logging
import random

import torch
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW, BertTokenizer

from data_utils.WeiboGraphDataset import WeiboGraphDataset
from trainer import Train
from model.HAN import BertForWeiboClassification

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

BERT_PATH = '/sdd/yujunshuai/model/chinese_L-12_H-768_A-12'

BATCH_SIZE_PER_GPU = 4
GPU_COUNT = torch.cuda.device_count()

seed = 1024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    w2id = torch.load('/sdd/yujunshuai/model/chinese_pretrain_vector/sgns.weibo.word.vocab.pt')
    dataset = WeiboGraphDataset('/sdd/yujunshuai/data/weibo/', w2id=w2id, max_comment_num=10,
                                restart_prob=0.6, delay=1000, tokenizer=tokenizer)

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

    model = BertForWeiboClassification(num_class=2, dropout=0.3, bert_path=BERT_PATH, finetune_bert=True, concat=True)
    # model = load_parallel_save_model('/sdd/yujunshuai/save_model/gear/best-validate-model.pt', model)
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

    trainer = Train(model_name='weibo_bert_cat',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=crit,
                    epochs=10,
                    print_step=1,
                    early_stop_patience=3,
                    # save_model_path=f"./save_model/{params['model_name']}",
                    save_model_path=f"/sdd/yujunshuai/save_model/weibo_bert_cat",
                    save_model_every_epoch=False,
                    metric=accuracy_score,
                    num_class=2,
                    # tensorboard_path='./tensorboard_log')
                    tensorboard_path='/sdd/yujunshuai/tensorboard_log')
    trainer.train()
    trainer.test()



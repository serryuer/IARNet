import logging

import torch
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW

from data_utils.FakedditGraphDataset import FakedditGraphDataset
from model.HAN import HANForFakedditClassification
from trainer import Train

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

BERT_PATH = '/home/tanghengzhu/yjs/model/bert-base-uncased_L-24_H-1024_A-16'

BATCH_SIZE_PER_GPU = 10
GPU_COUNT = torch.cuda.device_count()

if __name__ == '__main__':
    w2id = torch.load('/home/tanghengzhu/yjs/model/en_glove_vector/glove.42B.300d.w2id.pt')

    weight = torch.load('/home/tanghengzhu/yjs/model/en_glove_vector/glove.42B.300d.vectors.pt')

    train_dataset = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/',
                                         w2id=w2id,
                                         img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/train_image',
                                         bert_path=BERT_PATH)
    test_dataset = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/', isTest=True,
                                        w2id=w2id,
                                        img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/test_image',
                                        bert_path=BERT_PATH)
    val_dataset = FakedditGraphDataset('/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/', isVal=True,
                                       w2id=w2id,
                                       img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/validate_image',
                                       bert_path=BERT_PATH)

    train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT, shuffle=True)
    test_loader = DataListLoader(test_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15), shuffle=True)
    val_loader = DataListLoader(val_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15), shuffle=True)

    logging.info(f"train data all steps: {len(train_loader)}, "
                 f"validate data all steps : {len(val_loader)},"
                 f"test data all steps : {len(test_loader)}")

    model = HANForFakedditClassification(num_class=2, dropout=0.3, pretrained_weight=weight)
    # model = load_parallel_save_model('/home/tanghengzhu/yjs/save_model/gear/best-validate-model.pt', model)
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

    trainer = Train(model_name='fakeddit_han',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=crit,
                    epochs=10,
                    print_step=10,
                    early_stop_patience=3,
                    save_model_path='/home/tanghengzhu/yjs/save_model/fakeddit_han',
                    save_model_every_epoch=True,
                    metric=accuracy_score,
                    num_class=2,
                    tensorboard_path='./tensorboard_log')

    trainer.train()
    print(f"Testing result :{trainer.test()}")

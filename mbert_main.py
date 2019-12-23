import logging
import ssl

import torch
from sklearn.metrics import *
from torch.nn import DataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW

from data_utils.MultiModalDataset import MultiModalDataset
from model.MulimodalBert import MultimodalBertForClassification
from trainer import Train
from model.util import load_parallel_save_model

ssl._create_default_https_context = ssl._create_unverified_context
# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

BERT_PATH = '/home/tanghengzhu/yjs/model/bert-base-uncased_L-24_H-1024_A-16'

BATCH_SIZE_PER_GPU = 24
GPU_COUNT = torch.cuda.device_count()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    train_dataset = MultiModalDataset(text_root='/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/',
                                      img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/train_image',
                                      bert_path=BERT_PATH,
                                      max_sequence_length=256,
                                      num_class=2)
    test_dataset = MultiModalDataset(text_root='/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/',
                                     img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/test_image',
                                     bert_path=BERT_PATH,
                                     max_sequence_length=256,
                                     num_class=2,
                                     isTest=True)
    val_dataset = MultiModalDataset(text_root='/home/tanghengzhu/yjs/data/Fakeddit/fakeddit_v1.0/',
                                    img_root='/home/tanghengzhu/yjs/data/Fakeddit/images/validate_image',
                                    bert_path=BERT_PATH,
                                    max_sequence_length=256,
                                    num_class=2,
                                    isVal=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(BATCH_SIZE_PER_GPU * GPU_COUNT * 15), shuffle=True)

    model = MultimodalBertForClassification(num_class=2, bert_path=BERT_PATH)
    model = load_parallel_save_model('./save_model/mbert/best-validate-model.pt', model)
    model = DataParallel(model)

    model = model.cuda()

    # Prepare  optimizer and schedule(linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader))
    crit = torch.nn.CrossEntropyLoss()

    logging.info(f"train data all steps: {len(train_loader)}, "
                 f"validate data all steps : {len(val_loader)},"
                 f"test data all steps : {len(test_loader)}")

    trainer = Train(model_name='mbert',
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model=model,
                    optimizer=optimizer,
                    loss_fn=crit,
                    epochs=10,
                    print_step=10,
                    early_stop_patience=3,
                    save_model_path='./save_model/mbert',
                    save_model_every_epoch=True,
                    metric=accuracy_score,
                    num_class=2,
                    tensorboard_path='./tensorboard_log')

    trainer.train()
    print(f"Testing result :{trainer.test()}")

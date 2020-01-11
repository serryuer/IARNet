import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import *
# log format
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.util import load_parallel_save_model

C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)


class Train(object):
    def __init__(self, model_name, train_loader, val_loader, test_loader, model, optimizer, loss_fn, epochs, print_step,
                 early_stop_patience, save_model_path, num_class, save_model_every_epoch=False,
                 metric=f1_score, tensorboard_path=None):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.print_step = print_step
        self.early_stop_patience = early_stop_patience
        self.save_model_every_epoch = save_model_every_epoch
        self.save_model_path = save_model_path
        self.metric = metric
        self.num_class = num_class

        self.tensorboard_path = tensorboard_path

        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.best_val_epoch = 0
        self.best_val_score = 0

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)

    def _save_model(self, model_name):
        torch.save(self.model, os.path.join(self.save_model_path, model_name + '.pt'))

    def _early_stop(self, epoch, score):
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_val_epoch = epoch
            self._save_model('best-validate-model')
        else:
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + self.model_name + f"Validate has not promote {epoch - self.best_val_epoch}/{self.early_stop_patience}")
            if epoch - self.best_val_epoch > self.early_stop_patience:
                logging.info(self.model_name + f"-epoch {epoch}" + ":"
                             + f"Early Stop Train, best score locate on {self.best_val_epoch}, "
                             f"the best score is {self.best_val_score}")
                return True
        return False

    def eval(self):
        logging.info(self.model_name + ":" + "## Start to evaluate. ##")
        self.model.eval()
        eval_loss = 0.0
        preds = None
        true_labels = None
        for batch_data in tqdm(self.val_loader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(batch_data)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0) \
                                                .squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        result = {}
        result['accuracy'] = accuracy_score(true_labels, preds)
        result['recall'] = recall_score(true_labels, preds, average='macro')
        result['f1'] = f1_score(true_labels, preds, average='macro')
        return result

    def train(self):
        preds = None
        true_labels = None
        for epoch in range(self.epochs):
            tr_loss = 0.0
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"## The {epoch} Epoch, all {self.epochs} Epochs ! ##")
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"The current learning rate is {self.optimizer.param_groups[0].get('lr')}")
            self.model.train()
            since = time.time()
            for batch_count, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss, logits = outputs[:2]
                loss = loss.sum()
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)

                if (batch_count + 1) % self.print_step == 0:
                    pred_label = np.argmax(preds, axis=1)
                    logging.info(self.model_name + f"-epoch {epoch}" + ":"
                                 + f"batch {batch_count + 1} : loss is {tr_loss / (batch_count + 1)}, "
                                 f"accuracy is {accuracy_score(true_labels, pred_label)}, "
                                 f"recall is {recall_score(true_labels, pred_label, average='macro')}, "
                                 f"f1 is {f1_score(true_labels, pred_label, average='macro')}")
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_loss', tr_loss / (batch_count + 1),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_accuracy',
                                              accuracy_score(true_labels, pred_label),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_recall',
                                              recall_score(true_labels, pred_label, average='macro'),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_f1', f1_score(true_labels, pred_label, average='macro'),
                                              batch_count + epoch * len(self.train_loader))

            val_score = self.eval()
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_accuracy', val_score['accuracy'], epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_recall', val_score['recall'], epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_f1', val_score['f1'], epoch)

            logging.info(self.model_name + ": epoch" +
                         f"Epoch {epoch} Finished with time {format(time.time() - since)}, " +
                         f"validate accuracy score {val_score}")
            if self.save_model_every_epoch:
                self._save_model(f"{self.model_name}-{epoch}-{val_score['accuracy']}")
            if self._early_stop(epoch, val_score['accuracy']):
                break
        self.tb_writer.close()

    def test(self):
        logging.info(self.model_name + ":" + "## Start to Test. ##")
        self.model = load_parallel_save_model(os.path.join(self.save_model_path, 'best-validate-model.pt'), self.model)
        self.model.eval()
        test_loss = 0.0
        preds = None
        true_labels = None
        for batch_data in tqdm(self.test_loader, desc="Testing"):
            with torch.no_grad():
                outputs = self.model(batch_data)
                tmp_eval_loss, logits = outputs[:2]
                test_loss += tmp_eval_loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name.startswith('gear') or self.model_name.startswith(
                            'weibo') or self.model_name.startswith('tw'):
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0) \
                                                .squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('pure-bert'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name.startswith('spot-fake') or self.model_name.startswith('pure-vgg'):
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'weibo_han_comments':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        result = {}
        result['accuracy'] = accuracy_score(true_labels, preds)
        result['precision'] = precision_score(true_labels, preds, average='macro')
        result['recall'] = recall_score(true_labels, preds, average='macro')
        result['f1'] = f1_score(true_labels, preds, average='macro')
        with open(f"{self.model_name[:self.model_name.find('_') + 1]}test_result.txt", mode='a', encoding='utf-8') as f:
            f.write(self.model_name + '\n')
            f.write(str(result) + '\n')
            f.write('\n')
        return result

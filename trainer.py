import os
import time
import torch

import torch.nn as nn
import torch.nn.utils as utils
import numpy as np

from sklearn.metrics import *

import logging

# log format
from tensorboardX import SummaryWriter
from tqdm import tqdm

C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)


class Train(object):
    def __init__(self, model_name, train_loader, val_loader, test_loader, model, optimizer, loss_fn, epochs, print_step,
                 early_stop_patience, save_model_path, num_class, scheduler, save_model_every_epoch=False,
                 metric=f1_score, model_checkpoint=None):
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
        self.scheduler = scheduler

        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.best_val_epoch = 0
        self.best_val_score = 0

        if model_checkpoint:
            self.model = torch.load(model_checkpoint)

    def _save_model(self, model_name):
        torch.save(self.model, os.path.join(self.save_model_path, model_name + '.pt'))

    def _early_stop(self, epoch, score):
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_val_epoch = epoch
            self._save_model('best-validate-model')
        else:
            logging.info(f"Validate has not promote {epoch - self.best_val_epoch}/{self.early_stop_patience}")
            if epoch - self.best_val_epoch > self.early_stop_patience:
                logging.info(f"Early Stop Train, best score locate on {self.best_val_epoch}, "
                             f"the best score is {self.best_val_score}")
                return True
        return False

    def eval(self):
        logging.info("## Start to evaluate. ##")
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
                    if self.model_name == 'gear':
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name == 'pure-bert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'spot-fake':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name == 'gear':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0) \
                                                .squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'pure-bert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'spot-fake':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        eval_score = self.metric(true_labels, preds)
        result = {}
        result['accuracy'] = accuracy_score(true_labels, preds)
        result['recall_ma'] = recall_score(true_labels, preds, average='macro')
        result['f1_ma'] = f1_score(true_labels, preds, average='macro')
        result['recall_mi'] = recall_score(true_labels, preds, average='micro')
        result['f1_mi'] = f1_score(true_labels, preds, average='micro')
        return result

    def train(self):
        tb_writer = SummaryWriter()
        tr_loss = 0.0
        preds = None
        true_labels = None
        for epoch in range(self.epochs):
            logging.info(f"## The {epoch} Epoch, all {self.epochs} Epochs ! ##")
            logging.info(f"The current learning rate is {self.optimizer.param_groups[0].get('lr')}")
            self.model.train()
            since = time.time()
            for batch_count, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss, logits = outputs[:2]
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tr_loss += loss.mean().item()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    if self.model_name == 'gear':
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name == 'pure-bert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'spot-fake':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name == 'gear':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0). \
                                                squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'pure-bert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'spot-fake':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                if batch_count % self.print_step == 0:
                    pred_label = np.argmax(preds, axis=1)
                    logging.info(f"batch {batch_count} : loss is {loss.item()}, "
                                 f"accuracy is {accuracy_score(true_labels, pred_label)}, "
                                 f"recall is {f1_score(true_labels, pred_label, average='macro')}, "
                                 f"f1 is {f1_score(true_labels, pred_label, average='macro')}")
            val_score = self.eval()
            logging.info(
                f"Epoch {epoch} Finished with time {format(time.time() - since)}, validate accuracy score {val_score}")
            if self.save_model_every_epoch:
                self._save_model(f"{self.model_name}-{epoch}-{val_score['accuracy']}")
            if self._early_stop(epoch, val_score['accuracy']):
                break

    def test(self):
        logging.info("## Start to Test. ##")
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
                    if self.model_name == 'gear':
                        true_labels = torch.stack([data.y for data in batch_data], dim=0). \
                            squeeze(-1).detach().cpu().numpy()
                    elif self.model_name == 'pure-bert':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'spot-fake':
                        true_labels = batch_data[3].detach().cpu().numpy()
                    elif self.model_name == 'mbert':
                        true_labels = batch_data[3].detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    if self.model_name == 'gear':
                        true_labels = np.append(true_labels, torch.stack([data.y for data in batch_data], dim=0) \
                                                .squeeze(-1).detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'pure-bert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'spot-fake':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
                    elif self.model_name == 'mbert':
                        true_labels = np.append(true_labels, batch_data[3].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        result = {}
        result['accuracy'] = accuracy_score(true_labels, preds)
        result['recall_ma'] = recall_score(true_labels, preds, average='macro')
        result['f1_ma'] = f1_score(true_labels, preds, average='macro')
        result['recall_mi'] = recall_score(true_labels, preds, average='micro')
        result['f1_mi'] = f1_score(true_labels, preds, average='micro')
        return result

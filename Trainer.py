import os

import torch
import logging
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import get_time
from data.data_loader import au_dataloader
from models.iresnetse import IrResNetSe
from models.iresnet import iresnet100, iresnet50
from sklearn.metrics import f1_score
import numpy as np
from config import get_config
device_ids = [0, 1]


logging.basicConfig(level=logging.INFO,
                    filename="train.log",
                    filemode="w",
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

class Trainer(object):
    def __init__(self, conf, inference=False):
        self.conf = conf

        if conf.model == 'iresnetse50':
            self.model = IrResNetSe(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        elif conf.model == 'iresnet100':
            self.model = iresnet100(dropout=conf.drop_ratio, num_class=conf.au_class_num).to(conf.device)
        elif conf.model == 'iresnet50':
            self.model = iresnet50(dropout=conf.drop_ratio, num_class=conf.au_class_num).to(conf.device)
        
        print('{} model generated'.format(conf.model))

        if not inference:
            self.eval_or_not = conf.eval_or_not
            self.milestones = conf.milestones

            self.train_loader = au_dataloader(conf.train_data_path, conf,
                                              batch_size=conf.batch_size, is_training=True,
                                              pred_txt_file=None, flag='train')
            self.eval_loader = au_dataloader(conf.train_data_path, conf,
                                             batch_size=conf.batch_size, is_training=False,
                                             pred_txt_file=None, flag='valid')
            self.writer = SummaryWriter(conf.log_path)
            self.au_step = 0
            self.au_board_loss_every = len(self.train_loader) // 100
            self.au_evaluate_every = len(self.train_loader) // 10
            self.au_save_every = len(self.train_loader) // 5

            if conf.optimizer == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=conf.lr, momentum=conf.momentum)

            print('optimizers generated')
            self.au_board_loss_every = len(self.train_loader) // 100
            self.evaluate_every = len(self.train_loader) // 5
            self.save_every = len(self.train_loader) // 2

        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor(
                            [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]).to(self.conf.device))
        self.multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(weight=torch.Tensor(
                            [1, 2, 1, 1, 1, 1, 1, 6, 6, 6, 1, 2]).to(self.conf.device))

    def save_state(self, loss, accuracy, f1_macro, f1_bin, to_save_folder=True, extra='1', model_only=False):
        if to_save_folder:
            save_path = self.conf.save_path
        else:
            save_path = self.conf.pretrain_path
        if f1_bin <= 0.48:
            return
        torch.save(
            self.model, os.path.join(save_path,
                ('model_{}_loss-{:.3f}_acc-{:.3f}_f1_macro-{:.3f}_f1_bin-{:.3f}_step-{}.pt'.format(
                                                      get_time(), loss, accuracy, f1_macro, f1_bin, self.au_step))))

    def load_state(self, fixed_str='', use_pretrain=True, model_only=True):
        if use_pretrain:

            save_path = self.conf.pretrain_path
            pretrained_dict = torch.load(save_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            model_path = self.conf.model_path
            self.model.load_state_dict(torch.load(model_path))

            if not model_only:
                self.optimizer.load_state_dict(torch.load(os.path.join(self.conf.save_path, fixed_str)))

        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.cuda(device=device_ids[0])

    def train(self):
        self.model.train()
        for e in range(self.conf.epochs):
            print('epoch {} started'.format(e))
            logging.info('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()

            with tqdm(self.train_loader) as t1:
                for imgs, labels, labels_float in iter(self.train_loader):
                    imgs, labels, labels_float = imgs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0]), labels_float.cuda(device=device_ids[0])

                    imgs = imgs.to(self.conf.device)
                    labels = labels.to(self.conf.device)
                    labels_float = labels_float.to(self.conf.device)
                    self.optimizer.zero_grad()

                    output = self.model(imgs)

                    loss_of_bce = self.bce_loss(output, labels_float.float())
                    loss_of_bce = loss_of_bce.mean()
                    loss_of_multi_label = self.multi_label_loss(output, labels.float())
                    au_loss = loss_of_bce + loss_of_multi_label

                    t1.set_postfix(au_loss=au_loss.item())
                    t1.update(1)
                    au_loss.backward()
                    self.optimizer.step()

                    if self.au_step % self.au_board_loss_every == 0 and self.au_step != 0:
                        self.writer.add_scalar('au_loss', au_loss, self.au_step)
                        running_loss_au = 0.

                    if self.au_step % self.au_save_every == 0 and self.au_step != 0:
                        if self.eval_or_not:
                            with torch.no_grad():
                                self.model.eval()
                                acc, totalf1_macro, totalf1_bin = self.eval_au(self.eval_loader)
                        else:
                            acc, totalf1_macro, totalf1_macro_bin = 1, 1, 1
                        self.save_state(loss=au_loss, accuracy=acc,
                                        f1_macro=totalf1_macro, f1_bin=totalf1_bin)
                        self.model.train()

                    self.au_step += 1

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

def eval_au(self, eval_loader):
    predicted = []
    true_labels = []
    correct = 0
    total = 0
    count = 0
    for imgs, labels in tqdm(iter(eval_loader)):
        imgs, labels = imgs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
        torch.no_grad()
        if count < 100:
            imgs = imgs.to(self.conf.device_val)
            labels = labels.to(self.conf.device_val)
            logits = self.model(imgs)

            predicts = torch.greater(logits, 0).type_as(labels)
            cmp = predicts.eq(labels).cpu().numpy()
            correct += cmp.sum()
            total += len(cmp) * 12

            predicted.append(predicts.cpu().numpy().astype(int))
            true_labels.append(labels.cpu().numpy().astype(int))
            count = count + 1
        else:
            break
    acc = correct / total
    print('acc:', acc)

    predicted=np.vstack(predicted)
    true_labels=np.vstack(true_labels)
    total_f1=0
    for i in range(0, 12):
        total_f1 +=f1_score(true_labels[:,i], predicted[:,i], average='macro')
    print('mean-f1:', total_f1/12)
    class_names = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12',
                    'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    return acc, total_f1/12
    def eval_au(self, eval_loader):
        predicted = []
        true_labels = []
        correct = 0
        total = 0
        count = 0
        for imgs, labels, labels_float in tqdm(iter(eval_loader)):
            if count >-1:
                imgs = imgs.to(self.conf.device_val)
                logits = self.model(imgs)
                pred_au = torch.sigmoid(logits).cpu().detach().numpy()
                predicts = pred_au > 0.5
                predicts = predicts.astype(int)
                labels = labels.numpy().astype(int)
                cmp = np.equal(predicts, labels)
                correct += cmp.sum()
                total += len(cmp) * 12
                if count == 0:
                    predicted = predicts
                    true_labels = labels
                else:
                    predicted = np.concatenate((predicted,predicts), axis = 0)
                    true_labels = np.concatenate((true_labels,labels), axis = 0)
                count = count + 1
            else:
                break
        acc = correct / total
        print('acc:', acc)
        logging.info('acc:{}'.format(acc))

        label_size = np.array(predicted).shape[1]
        
        logging.info(label_size)
        f1s_macro_each_au = []
        f1s_bin_each_au = []
        for i in range(label_size):
            f1_macro = f1_score(true_labels[:, i],predicted[:, i],average='macro')
            f1_bin = f1_score(true_labels[:, i],predicted[:, i])
            f1s_macro_each_au.append(f1_macro)
            f1s_bin_each_au.append(f1_bin)

        f1s_macro_mean = np.mean(f1s_macro_each_au)
        print('f1s macro mean:', f1s_macro_mean)
        logging.info('f1s macro mean:{}'.format(f1s_macro_mean))
        print('f1s macro each au:', f1s_macro_each_au)
        logging.info('f1s macro each au:{}'.format(f1s_macro_each_au))
        f1s_bin_mean = np.mean(f1s_bin_each_au)
        print('f1s bin mean:', f1s_bin_mean)
        logging.info('f1s bin mean:{}'.format(f1s_bin_mean))
        print('f1s bin each au:', f1s_bin_each_au)
        logging.info('f1s bin each au:{}'.format(f1s_bin_each_au))

        return acc, f1s_macro_mean, f1s_bin_mean

if __name__ == '__main__':

    conf = get_config()

    print(conf)

    if not os.path.exists(conf.log_path):
        os.makedirs(conf.log_path)
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)

    trainer = Trainer(conf)
    trainer.load_state()
    trainer.train()

from config import get_config
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision
from PIL import Image, ImageFile
import pandas as pd
import torch
import numpy as np
from collections import Counter
import csv
import os
import sys
sys.path.append('..')


ImageFile.LOAD_TRUNCATED_IMAGES = True


AU_list = [1, 2, 4, 6, 7, 10, 12,15, 23, 24, 25, 26]


class AUDataset(Dataset):
    def __init__(self, image_dir, config_au, label_width=12, transform=None, pred_txt_file=None, flag='train'):
        self.config_au=config_au
        self._image_dir = image_dir
        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12, 15, 23, 24, 25, 26]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]

        self._label_width = label_width
        self._transform = transform
        self._affwild_path = self.config_au.train_data_path
        if flag=='train':

            au_labels = pd.read_csv(self.config_au.au_train_annot_path).loc[:,['AU'+str(au) for au in AU_list]]
            au_labels = au_labels.reset_index(drop=True).values
            self._samples = au_labels

            self._sampled_id = list(range(self._samples.shape[0]))

            paths = pd.read_csv(self.config_au.au_train_annot_path).loc[:, 'path']
            paths = paths.reset_index(drop=True).values
            self._paths = [self._affwild_path+'/'+p for p in paths]
        else:

            au_labels = pd.read_csv(self.config_au.au_valid_annot_path).loc[:,['AU'+str(au) for au in AU_list]]
            au_labels = au_labels.reset_index(drop=True).values
            self._samples = au_labels

            self._sampled_id = list(range(self._samples.shape[0]))

            paths = pd.read_csv(self.config_au.au_valid_annot_path).loc[:, 'path']
            paths = paths.reset_index(drop=True).values
            self._paths = [self._affwild_path+'/'+p for p in paths]

    def __len__(self):
        return len(self. _sampled_id)

    def __getitem__(self, index):
        fname = self._paths[index]
        if 'AffWild2' in fname:
            fname = fname.replace(self.config_au.train_data_path, self._affwild_path)
        labels = self._samples[index]
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')#(112,112,3)

        if self._transform:
            img = self._transform(img)
        float_labels = torch.tensor(labels, dtype=torch.float32)
        int_labels = np.round(labels)
        labels = torch.tensor(int_labels, dtype=torch.int32)
        return img, labels, float_labels


def au_dataloader(image_dir, config_au, batch_size, is_training=True, pred_txt_file=None, flag='train'):
    need_balance=False
    if is_training:
        transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ColorJitter(brightness=0.3, contrast=0.3,
                                  saturation=0.3, hue=0.3),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

        ])
    else:
        transform = trans.Compose([trans.Resize(112),
                                   trans.ToTensor(),
                                   trans.Normalize(
                                       [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])
    
    if need_balance and is_training:
        print('=============use data balance================')
        # torchsampler package cite from https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels
        from torchsampler.imbalanced_sampler import SamplerFactory
        au_dataset = AUDataset(image_dir, config_au, transform=transform, pred_txt_file=pred_txt_file, flag=flag)
        sampler = SamplerFactory.get_by_name('AU', au_dataset)
        return DataLoader(au_dataset,
                      sampler = sampler,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=16,
                      drop_last=is_training)
    else:
        return DataLoader(AUDataset(image_dir, config_au, transform=transform, pred_txt_file=pred_txt_file, flag=flag),
                      batch_size=batch_size,
                      shuffle=is_training,
                      pin_memory=True,
                      num_workers=16,
                      drop_last=is_training)


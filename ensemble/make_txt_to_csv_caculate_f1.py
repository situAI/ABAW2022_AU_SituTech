import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


def make_112_au_data(image_224_path, au_224_label_path, save_path, train=True):
    if train:
        train_label_path = os.path.join(au_224_label_path, 'Train_Set')
        train_label_list = os.listdir(train_label_path)
        data_list = []
        for train_label in train_label_list:
            f = open(os.path.join(train_label_path, train_label), 'r')
            labels = f.readlines()
            images = os.listdir(os.path.join(image_224_path, train_label.split('.')[0]))
            images.sort()
            for image in images:
                if image.split('.')[1] == 'jpg':
                    if labels[int(image.split('.')[0].strip())].strip().split(',')[0] != -1:
                        d_ = (os.path.join(train_label.split('.')[0], image) + ',' + labels[
                            int(image.split('.')[0].strip())].strip()).split(',')
                        data_list.append(d_)
                    else:
                        continue
                else:
                    continue
        label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                       'AU26']

        xml_df = pd.DataFrame(data_list, columns=label_names)
        xml_df.to_csv('au_train.csv', index=None)
    else:
        val_label_path = os.path.join(au_224_label_path)
        val_label_list = os.listdir(val_label_path)
        data_list = []
        for train_label in val_label_list:
            f = open(os.path.join(val_label_path, train_label), 'r')
            labels = f.readlines()
            images = os.listdir(os.path.join(image_224_path, train_label.split('.')[0]))
            images.sort()
            for image in images:
                if image.split('.')[1] == 'jpg':
                    if labels[int(image.split('.')[0].strip())].strip().split(',')[0] != -1:
                        d_ = (os.path.join(train_label.split('.')[0], image) + ',' + labels[
                            int(image.split('.')[0].strip())].strip()).split(',')
                        data_list.append(d_)
                    else:
                        continue
                else:
                    continue
        label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                       'AU26']
        data_list.sort(key=lambda x: x[0], reverse=False)
        xml_df = pd.DataFrame(data_list, columns=label_names)
        xml_df.sort_values(by='path', ascending=True, inplace=True)
        xml_df.to_csv(os.path.join(save_path, 'pred_au_val.csv'), index=None)


def calculate_f1(val_csv, concatenate_csv):
    aulist = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
    val_data = pd.read_csv(val_csv)
    concatenate_data = pd.read_csv(concatenate_csv)
    val_labels = val_data.loc[:, ['AU' + str(i) for i in aulist]].values
    concatenate_labels = concatenate_data.loc[:, ['AU' + str(i) for i in aulist]].values
    f1s_macro_each_au = []
    f1s_binary_each_au = []
    for i in range(0, 12):
        f1_score_au_macro = f1_score(val_labels[:, i], concatenate_labels[:, i], average='macro')
        f1_score_au_binary = f1_score(val_labels[:, i], concatenate_labels[:, i])
        f1s_macro_each_au.append(f1_score_au_macro)
        f1s_binary_each_au.append(f1_score_au_binary)
    f1s_macro_mean = np.mean(f1s_macro_each_au)
    f1s_binary_mean = np.mean(f1s_binary_each_au)
    return f1s_binary_mean, f1s_binary_each_au, f1s_macro_mean, f1s_macro_each_au


if __name__ == '__main__':

    make_112_au_data("/root/autodl-tmp/data/cropped_aligned/",
                     "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/val_merge_txt/",
                     '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100',
                     False)


    f1s_binary_mean, f1s_binary_each_au, f1s_macro_mean, f1s_macro_each_au = calculate_f1(
        "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/au_val.csv",
        "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/pred_au_val.csv"
        )
    print(f'\nf1s_binary_mean={f1s_binary_mean}'
          f'\nf1s_binary_each_au={f1s_binary_each_au}'
          f'\nf1s_macro_mean={f1s_macro_mean}'
          f'\nf1s_macro_each_au={f1s_macro_each_au}')

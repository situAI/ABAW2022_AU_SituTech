import os
import random

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


def make_0_1(logits_csv_path, save_path):
    label_names = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                   'AU26']
    datas = pd.read_csv(logits_csv_path)
    labels = datas.loc[:, label_names].values
    p = datas.loc[:, ['path']].values
    pred = labels > 0.5
    pred = pred.astype(int)
    r = np.concatenate((p, pred), axis=1)
    data_list = r.tolist()
    data_list.sort(key=lambda x: x[0], reverse=False)
    xml_df = pd.DataFrame(data_list, columns=['path'] + label_names)
    xml_df.sort_values(by='path', ascending=True, inplace=True)
    xml_df.to_csv(os.path.join(save_path, 'pred_au_val_logits_01.csv'), index=None)


def make_112_au_data(image_224_path, au_224_label_path, save_path, train=False):
    if train:
        pass
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
        xml_df.to_csv(os.path.join(save_path, 'pred_au_val_logits.csv'), index=None)


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


def weight_avg_ensemble(weight_list, csv_list, save_path):
    temp = []
    au_list = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
    p = []
    count = 0
    for weight, csv_ in zip(weight_list, csv_list):
        datas = pd.read_csv(csv_)
        p = datas.loc[:, ['path']].values
        labels = datas.loc[:, ['AU' + str(i) for i in au_list]].values
        if count == 0:
            temp = labels * weight
        else:
            temp += labels * weight
        count += 1
    out = temp / 10
    result = np.concatenate((p, out), axis=1)
    data_list = result.tolist()
    label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                   'AU26']
    data_list.sort(key=lambda x: x[0], reverse=False)
    xml_df = pd.DataFrame(data_list, columns=label_names)
    xml_df.sort_values(by='path', ascending=True, inplace=True)
    xml_df.to_csv(os.path.join(save_path, 'pred_au_val_logits.csv'), index=None)


if __name__ == '__main__':
    # make_0_1("/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result/pred_au_val_logits.csv",
    #          '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result')
    # f1s_binary_mean, f1s_binary_each_au, f1s_macro_mean, f1s_macro_each_au = calculate_f1(
    #     "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/au_val.csv",
    #     "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result/pred_au_val_logits_01.csv")
    # print(f'\nf1s_binary_mean={f1s_binary_mean}'
    #       f'\nf1s_binary_each_au={f1s_binary_each_au}'
    #       f'\nf1s_macro_mean={f1s_macro_mean}'
    #       f'\nf1s_macro_each_au={f1s_macro_each_au}')

    # make_112_au_data(
    #     '/root/autodl-tmp/data/cropped_aligned/',
    #     '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/val_merge_logits_txt',
    #     '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/'
    # )
    # '''
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s',
                        # 时间 文件名 line:行号  levelname logn内容
                        datefmt='%d %b %Y,%a %H:%M:%S',  # 日 月 年 ，星期 时 分 秒
                        filename='/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/best_weight.log',
                        filemode='a')
    best = 0.731
    best_weight = []
    weight_list = []
    for i in range(0, 11):
        for j in range(0, 11):
            for k in range(0, 11):
                for l in range(0, 11):
                    if i + k + j + l == 10:
                        weight_list.append([i, j, k, l])

    for w in weight_list:
        path_list = [
            "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result/pred_au_val_logits.csv",
            "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_situ_data/pred_au_val_logits.csv",
            "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/wyn_24fps/pred_au_val_logits.csv",
            "/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/pred_au_val_logits.csv"
        ]

        weight_avg_ensemble(
            w,
            path_list,
            '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/')

        make_0_1('/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/pred_au_val_logits.csv',
                 '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/')
        f1s_binary_mean, f1s_binary_each_au, f1s_macro_mean, f1s_macro_each_au = calculate_f1(
            '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/au_val.csv',
            '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/pred_au_val_logits_01.csv')
        # print(f'\nf1s_binary_mean={f1s_binary_mean}'
        #       f'\nf1s_binary_each_au={f1s_binary_each_au}'
        #       f'\nf1s_macro_mean={f1s_macro_mean}'
        #       f'\nf1s_macro_each_au={f1s_macro_each_au}')
        print(f1s_macro_mean)
        if f1s_macro_mean > best:
            best = f1s_macro_mean
            best_weight = weight_list
            logging.info(f'f1s_macro_mean:{best}    best_weight:{w}\n')
    # '''

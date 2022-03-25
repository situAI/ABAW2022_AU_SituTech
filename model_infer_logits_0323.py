import os
import torch
import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from models.iresnet import iresnet100
import numpy as np
from sklearn.metrics import f1_score
import os
import logging

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s',
#                     # 时间 文件名 line:行号  levelname logn内容
#                     datefmt='%d %b %Y,%a %H:%M:%S',  # 日 月 年 ，星期 时 分 秒
#                     filename='/root/autodl-tmp/qfs/ABAW_Competition_0323/video_loss_frame_static.log',
#                     filemode='a')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class AUBatchInferDataset(Dataset):
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.aus = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
        self._transform = transforms.Compose([transforms.Resize(112),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                              ])
        self.paths = os.listdir(self.image_dir_path)
        self.paths.sort()
        self._paths = []
        for p in self.paths:
            if p.split('.')[1] == 'jpg':
                self._paths.append(self.image_dir_path + '/' + p)
            else:
                continue

    def __len__(self):
        return len(self._paths)

    def get_images(self):
        return self.paths

    def __getitem__(self, index):
        fname = self._paths[index]
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self._transform:
            img = self._transform(img)
        return img


def load_model(model_path):
    return torch.load(model_path)


def load_val_data_loader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=32)
    return data_loader


def batch_infer(model, logits_txt_names, txt_names, batch_dataset, batch_size, frame_nums=None, log=False,
                device='cuda'):
    f_logits = open(logits_txt_names, 'w')
    f = open(txt_names, 'w')
    f.write('AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26')
    f_logits.write('AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26')
    model = model.to(device)
    model.eval()
    data_loader = load_val_data_loader(batch_dataset, batch_size)
    count = 0
    predicted = []
    pred_logits = []
    with torch.no_grad():
        for i, (image) in enumerate(tqdm.tqdm(data_loader)):
            image = image.to(device)
            outputs = model(image)
            pred_au = torch.sigmoid(outputs).cpu().detach().numpy()
            predicts = pred_au > 0.5
            predicts = predicts.astype(int)
            if count == 0:
                pred_logits = pred_au
                predicted = predicts
            else:
                pred_logits = np.concatenate((pred_logits, pred_au), axis=0)
                predicted = np.concatenate((predicted, predicts), axis=0)
            count = count + 1
    dir_images = batch_dataset.get_images()
    temp = []
    predicted = predicted.tolist()
    log_loss_frame = []
    for k in range(1, frame_nums + 1):
        if str(k).zfill(5) + '.jpg' not in dir_images:
            temp.append(str(k + 1).zfill(5))
            predicted.insert(k - 1, [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    if log:
        logging.info(f'{txt_names.split("/")[-1]}={temp}')
    for pre in predicted:
        pre = [str(j) for j in pre]
        _str = '\n' + ','.join(pre)
        f.write(_str)
    print(f'{txt_names} saved successfully!')
    pred_logits = pred_logits.tolist()
    for k in range(1, frame_nums + 1):
        if str(k).zfill(5) + '.jpg' not in dir_images:
            pred_logits.insert(k - 1, [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    for pre_logit in pred_logits:
        pre_logit = [str(k) for k in pre_logit]
        __str = '\n' + ','.join(pre_logit)
        f_logits.write(__str)
    print(f'{logits_txt_names} saved successfully!')
    f.close()
    f_logits.close()


if __name__ == '__main__':
    frame_num_dicts = {}
    # 视频帧数txt文件
    t_f = open("./tools/new_all_video_frame_static.txt", 'r')
    frame_data = t_f.readlines()
    for f_d in frame_data:
        frame_num_dicts[f_d.strip().split(',')[0].strip()] = int(f_d.strip().split(',')[1])
    # crop align图像地址
    image_dir = "/user3/Aff-Wild2/cropped_aligned/"
    # 测试集视频txt文件
    val_txt_path = './tools/val_video.txt'
    test_txt_pth = './tools/Action_Unit_Detection_Challenge_test_set_release.txt'
    # val_txt_path = '/root/autodl-tmp/qfs/ABAW_Competition_0320/Action_Unit_Detection_Challenge_test_set_release.txt'
    # 模型pt文件夹地址
    # model_paths = "/root/autodl-tmp/qfs/ABAW_Competition_0321/0321_model/"
    # save_root_path='/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/'
    # save_txt_path = './tools/val_result/val_result_txt_0324'
    # save_logits_txt_path = './tools/val_result/val_logits_result_txt_0324'

    save_txt_path = './tools/test_result/'
    save_logits_txt_path = './tools/test_result/'

    test_txt_names = open(val_txt_path, 'r').readlines()
    test_txt_list = [k.strip() + '.txt' for k in test_txt_names]
    count_log = 0
    root_model_path = './tools/selected_models'
    model_paths_list = ['fps24']
    for m in model_paths_list:
        # 结果保存地址
        save_txt_path = os.path.join(save_txt_path, m, 'test_result')
        save_logits_txt_path = os.path.join(save_logits_txt_path, m, 'test_logits_result')
        for model_path in os.listdir(os.path.join(root_model_path, m)):
            model = load_model(os.path.join(root_model_path, m, model_path))
            for val_txt_name in test_txt_list:
                d = AUBatchInferDataset(os.path.join(image_dir, val_txt_name.split('.')[0]))
                if model_path.__contains__('.pt'):
                    if not os.path.exists(
                            f'{save_txt_path}/{model_path.split(".")[0]}'):
                        os.makedirs(
                            f'{save_txt_path}/{model_path.split(".")[0]}')
                    if not os.path.exists(
                            f'{save_logits_txt_path}/{model_path.split(".")[0]}'):
                        os.makedirs(
                            f'{save_logits_txt_path}/{model_path.split(".")[0]}')
                    name = model_path.split('.pt')[0]
                    # au_dataset = AUDataset(image_dir, val_annot_path)
                    #model = load_model(os.path.join(root_model_path,m, model_path))
                    # acc, f1s_mean,f1s_each_au = val_infer(model=model, data_loader=load_val_data_loader(au_dataset, 324),names=name)
                    # logging.info(f'\nmodel_path={model_path}   \nacc = {acc}    \nf1s_mean = {f1s_mean}  \nf1s_each_au = {f1s_each_au}')
                    # print(f'acc = {acc}    f1s_mean = {f1s_mean}  f1s_each_au = {f1s_each_au}')
                    if val_txt_name.split('.')[0].__contains__('left') or val_txt_name.split('.')[0].__contains__(
                            'right'):
                        frame_nums = frame_num_dicts[val_txt_name.split('.')[0].split('_')[0].strip()]
                    else:
                        frame_nums = frame_num_dicts[val_txt_name.split('.')[0]]
                    if count_log == 0:
                        batch_infer(model,
                                    f'{save_logits_txt_path}/{model_path.split(".")[0]}/{val_txt_name}',
                                    f'{save_txt_path}/{model_path.split(".")[0]}/{val_txt_name}',
                                    d, 64, frame_nums, True)
                    else:
                        batch_infer(model,
                                    f'{save_logits_txt_path}/{model_path.split(".")[0]}/{val_txt_name}',
                                    f'{save_txt_path}/{model_path.split(".")[0]}/{val_txt_name}',
                                    d, 64, frame_nums)
            count_log += 1

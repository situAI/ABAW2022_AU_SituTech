import os
from easydict import EasyDict as edict
import torch
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.train_data_path = '/root/autodl-tmp/data/cropped_aligned'
    conf.au_train_annot_path = '/root/autodl-tmp/jwq/ABAW_Competition_bagging2/tools/au_train_bagging_smooth_modify.csv'
    conf.au_valid_annot_path = '/root/autodl-tmp/jwq/ABAW_Competition_bagging2/tools/au_val_smooth.csv'

    conf.work_path = '/root/autodl-tmp/jwq/ABAW2_logs_au_1/bagging1/save'
    conf.work_name = 'bagging1'

    conf.log_path = os.path.join(conf.work_path, conf.work_name, 'log')
    conf.save_path = os.path.join(conf.work_path, conf.work_name, 'save')
#--------------------model config-----------------------------------
    conf.model = 'iresnet100' 

    if conf.model == 'iresnet100':
        conf.pretrain_path = os.path.join('/root/autodl-tmp/', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    elif conf.model == 'iresnet100_nohead':
        conf.pretrain_path = os.path.join('/root/autodl-tmp/', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    elif conf.model == 'iresnet50':
        conf.pretrain_path = os.path.join('/root/autodl-tmp/', 'model_ir_se50.pth')

    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.au_class_num = 12
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.device_val = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.need_balance = True
#--------------------Training Config ------------------------
    if training:
        conf.epochs = 20
        conf.momentum = 0.9
        conf.batch_size = 2
        conf.lr = 0.001
        conf.weight_decay = 5e-4
        conf.pin_memory = True
        conf.num_workers = 8
        conf.milestones = [4, 6, 8]
        conf.sample = False
        conf.optimizer = 'SGD'
        conf.eval_or_not = True
#--------------------Inference Config ------------------------
    else:
        conf.batch_size = 2
        conf.pin_memory = True
        conf.num_workers = 8
    return conf

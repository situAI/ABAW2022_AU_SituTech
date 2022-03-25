import os
from datetime import datetime
import torch

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def get_filename(n):
    filename, ext = os.path.splitext(os.path.basename(n))
    return filename

def get_extension(n):
    filename, ext = os.path.splitext(os.path.basename(n))
    return ext

def get_path(n):
    head, tail = os.path.split(n)
    return head

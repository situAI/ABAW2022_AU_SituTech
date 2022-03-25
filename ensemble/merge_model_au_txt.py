import pandas as pd
import numpy as np
import os

import tqdm


def merge_au_txt(au_names_list, root_au_path, save_path,logits=False):
    aulist = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
    for au_name in tqdm.tqdm(au_names_list):
        for au_txts in os.listdir(os.path.join(root_au_path, au_name)):
            count = []
            for i in au_names_list:
                data = pd.read_csv(os.path.join(root_au_path, i, au_txts))
                label = data.loc[:, [i.split('-')[0].upper()]].values
                if len(count) == 0:
                    count = label
                else:
                    count = np.concatenate((count, label), axis=1)
            df = pd.DataFrame(count.tolist(), columns=["AU" + str(j) for j in aulist])
            if logits:
                if not os.path.exists(os.path.join(save_path, 'val_merge_logits_txt')):
                    os.makedirs(os.path.join(save_path, 'val_merge_logits_txt'))
                df.to_csv(os.path.join(save_path, 'val_merge_logits_txt', f'{au_txts}'), index=None)
            else:
                if not os.path.exists(os.path.join(save_path, 'val_merge_txt')):
                    os.makedirs(os.path.join(save_path, 'val_merge_txt'))
                df.to_csv(os.path.join(save_path, 'val_merge_txt', f'{au_txts}'), index=None)


if __name__ == '__main__':
    au_names_list = ['AU1',
                     'AU2',
                     'AU4',
                     'AU6',
                     'AU7',
                     'AU10',
                     'AU12',
                     'AU15',
                     'AU23',
                     'AU24',
                     'AU25',
                     'AU26',
                     ]
    # merge_au_txt(au_names_list,
    #              '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result/val_result/',
    #              '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/abaw_situ_data_our_label_smooth_result/',
    #              False)
    merge_au_txt(au_names_list,
                 '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/val_logits_result/',
                 '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/',
                 True)
    merge_au_txt(au_names_list,
                 '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/val_result/',
                 '/root/autodl-tmp/qfs/ABAW_Competition_0324/0324_val_result/submit_res100/',
                 False)

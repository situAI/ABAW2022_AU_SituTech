import numpy as np
import pandas as pd
import os
from tqdm import tqdm

anno_root = '/user3/Aff-Wild2/annotations/AU_Set/'
train_set_root = '/user3/Aff-Wild2/annotations/AU_Set/Train_Set/'
valid_set_root = '/user3/Aff-Wild2/annotations/AU_Set/Validation_Set/'
transformed_root = '/user3/Aff-Wild2/annotations/output/annots'
img_root = '/user3/Aff-Wild2/cropped_aligned'


def get_absolute_file_path(path):
    """
    Get all absolute file path in current path

    Parameters
    ----------
    path(str): current path

    Returns
    -------
    file_name_list(list(str)): all file path in current path
    """
    file_list = os.listdir(path)
    file_name_list = list()
    for file_name in file_list:
        file_name_list.append(os.path.join(path, file_name))

    return file_name_list


def txt2list(txt_file):
    """
    Transform txt file to list
    such as valence, arousal \n -0.01, -0.005 \n ..... in 5-60-1920x1080-2.txt -> df.valence = ... and df.arousal = ...
    -> valence_list, arousal_list

    Parameters
    ----------
    txt_file(str): **Absolute Path** of txt file

    Returns
    -------
    valence_list(list(float)): contain valence in this txt file
    arousal_list(list(float)): contain arousal in this txt file
    """
    df = pd.read_csv(txt_file, sep=',')
    valence_list, arousal_list = len(df.valence), len(df.arousal)

    return valence_list, arousal_list


def get_all_img_list(img_root):
    """
    Get all **Absoulte Path** of image

    Parameters
    ----------
    path(str): **Absolute Path** of image folder root path

    Returns
    -------
    img_path_list(list(str)): get image relative path (like )
    """
    img_list = os.listdir(img_root)
    img_list = sorted(img_list)

    return img_list


def convert_txt_to_df(txt):
    """
    transform txt file to df

    Parameters
    ----------
    txt(str): **Absolute Path** of txt file

    Returns
    -------
    df(pd.DataFrame): column is [path, valence, arousal]
    """
    df = pd.read_csv(txt, sep=',')
    length = len(df)
    txt_name = txt.split('/')[-1].split('.')[0]
    name_list = list()
    for idx in range(length):
        name_list.append('%05d' % (idx + 1))
    path_list = list()
    for name in name_list:
        path_list.append('{}/{}.jpg'.format(txt_name, name))
    df.insert(0, 'path', path_list)

    return df


def convert_df_list_to_df(df_list):
    """
    Convert all df into ONE-ALL df

    Parameters
    ----------
    df_list(list(pd.DataFrame)): DataFrame list

    Returns
    -------
    df(list(pd.DataFrame)): DataFrame of all data
    """
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)

    return df


def filter(df, img_root):
    """
    Filter annotations

    Parameters
    ----------
    df(pd.DataFrame): the overall-or-any df
    img_root(str): the root path of img

    Returns
    -------
    after(pd.DataFrame): the filtered df
    """
    non_list = list()
    length = len(df.path)
    for idx in tqdm(range(length)):
        if not os.path.exists(os.path.join(img_root, df.path[idx])) or df.AU1[idx] == -1:
            non_list.append(idx)
    df = df.drop(index=non_list)
    df = df.reset_index(drop=True)

    return df


def main():
    train_txt_list = get_absolute_file_path(train_set_root)
    valid_txt_list = get_absolute_file_path(valid_set_root)

    train_df_list = list()
    valid_df_list = list()

    for train_txt in train_txt_list:
        train_df_list.append(convert_txt_to_df(train_txt))

    for valid_txt in valid_txt_list:
        valid_df_list.append(convert_txt_to_df(valid_txt))

    train_df = convert_df_list_to_df(train_df_list)
    valid_df = convert_df_list_to_df(valid_df_list)

    print('transforming valid')
    transformed_valid_df = filter(valid_df, img_root)
    transformed_valid_df.to_csv(os.path.join(transformed_root, 'valid.csv'))
    print('done! saving to {}'.format(os.path.join(transformed_root, 'valid.csv')))

    print('transforming train')
    transformed_train_df = filter(train_df, img_root)
    transformed_train_df.to_csv(os.path.join(transformed_root, 'train.csv'))
    print('done! saving to {}'.format(os.path.join(transformed_root, 'train.csv')))


if __name__ == '__main__':
    main()

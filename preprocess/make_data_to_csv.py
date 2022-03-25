import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def make_112_au_data_modify():

    f1 = open('au_bagging_train.csv', 'r')
    labels1 = f1.readlines()
    labels1 = labels1[1:]
    data_list = []
    for line in labels1:
        arr = line.strip().split(',')
        smooth_label = [arr[0]]
        label = arr[1:]
        for l in label:
            if int(l) == 0:
                l = round(random.uniform(0, 0.05), 2)
            else:
                l = round(random.uniform(0.9, 1), 2)
            smooth_label.append(str(l))

        data_list.append(smooth_label)
    
    f2 = open('situ_data.csv', 'r')
    labels2 = f2.readlines()
    labels2 = labels2[1:]
    for line in labels2:
        arr = line.strip().split(',')
        smooth_label = [arr[1]]
        label = arr[2:]
        for l in label:
            if int(l) == 1:
                l = round(random.uniform(0.9, 1), 2)
            else:
                l = round(random.uniform(0, 0.05), 2)
            
            smooth_label.append(str(l))
        data_list.append(smooth_label)

    label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    xml_df = pd.DataFrame(data_list, columns=label_names)
    xml_df.to_csv('au_train_bagging_smooth_situ_new.csv', index=None)

def make_112_au_data(image_224_path, au_224_label_path, train=True):
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
                    if int(labels[int(image.split('.')[0].strip())].strip().split(',')[0]) != -1:
                        d_ = (os.path.join(train_label.split('.')[0], image) + ',' + labels[
                            int(image.split('.')[0].strip())].strip()).split(',')
                        label = labels[int(image.split('.')[0].strip())].strip().split(',')
                        smooth_label = [os.path.join(train_label.split('.')[0], image)]
                        for l in label:
                            if int(l) == 0:
                                l = round(random.uniform(0, 0.05), 2)
                            else:
                                l = round(random.uniform(0.9, 1), 2)
                            smooth_label.append(str(l))
                        
                        smooth_label[8] = round(random.uniform(0, 0.05), 2)
                        smooth_label[9] = round(random.uniform(0, 0.05), 2)
                        smooth_label[11] = round(random.uniform(0, 0.05), 2)
                        smooth_label[12] = round(random.uniform(0, 0.05), 2)

                        if int(label[1]) == 0 or int(label[7]) == 0 or int(label[8]) == 0 or int(label[9]) == 0 or int(label[11]) == 0:
                            random_num = random.randint(1, 12)
                            if random_num < 8:
                                data_list.append(smooth_label)
                        else:
                            data_list.append(smooth_label)
                    else:
                        continue
                else:
                    continue
        label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                       'AU26']
        xml_df = pd.DataFrame(data_list, columns=label_names)
        xml_df.to_csv('au_train_bagging_smooth_24.csv', index=None)
    else:
        val_label_path = os.path.join(au_224_label_path, 'Validation_Set')
        val_label_list = os.listdir(val_label_path)
        data_list = []
        for train_label in val_label_list:
            f = open(os.path.join(val_label_path, train_label), 'r')
            labels = f.readlines()
            images = os.listdir(os.path.join(image_224_path, train_label.split('.')[0]))
            images.sort()
            for image in images:
                if image.split('.')[1] == 'jpg':
                    if int(labels[int(image.split('.')[0].strip())].strip().split(',')[0]) != -1:
                        d_ = (os.path.join(train_label.split('.')[0], image) + ',' + labels[
                            int(image.split('.')[0].strip())].strip()).split(',')
                        label = labels[int(image.split('.')[0].strip())].strip().split(',')
                        smooth_label = [os.path.join(train_label.split('.')[0], image)]
                        for l in label:
                            if int(l) == 0:
                                l = round(random.uniform(0, 0.05), 2)
                            else:
                                l = round(random.uniform(0.9, 1), 2)
                            smooth_label.append(str(l))

                        smooth_label[8] = round(random.uniform(0, 0.05), 2)
                        smooth_label[9] = round(random.uniform(0, 0.05), 2)
                        smooth_label[11] = round(random.uniform(0, 0.05), 2)
                        smooth_label[12] = round(random.uniform(0, 0.05), 2)

                        data_list.append(smooth_label)
                    else:
                        continue
                else:
                    continue
        label_names = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25',
                       'AU26']
        xml_df = pd.DataFrame(data_list, columns=label_names)
        xml_df.to_csv('au_val_smooth_24.csv', index=None)






def make_random_csv():
    #batch_size, ID_num_per_batch, pic_num_per_AU, negtive_num
    heads = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    # input_train_csv = '/user3/Aff-Wild2/annotations/output/annots/train.csv'
    input_train_csv = './train_au_del_0316.csv'
    all_path_list = pd.read_csv(input_train_csv).loc[:, heads]
    all_file_length = len(all_path_list)
    print(all_file_length)
    sample_rate = 24
    sample_num = int(all_file_length // sample_rate *2)
    sample_index = np.random.randint(24, size = sample_num)
    print(sample_index[0:72])
    for i in range(len(sample_index)):
        sample_index[i] = sample_index[i] + ((i//2 )*24)
    print(sample_index[0:72])
    sampled_data = pd.read_csv(input_train_csv).loc[sample_index, heads]
    sampled_data_df = pd.DataFrame(sampled_data, columns=heads)
    sampled_data_df.to_csv('./sample_24fps_train_clean_0316.csv', index = None)


    # for each video , sample one each 24frames
def make_batch_csv():
    #make all video list
    all_video_list = os.listdir('/user3/Aff-Wild2/cropped_aligned')
    # batch_num
    # make dict ={'video_name':{'AU1':AU1_list, 'AU2':AU2_list, 'AU3':AU3_list, 'all_neg':all_neg_list}
    org_dic = {}
    heads = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    heads_au = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    # for one_video in all_video_list:
    #     print(one_video)
    #     org_dic[one_video] = {}
    #     df = pd.read_csv('./sample_24fps_train.csv')
    #     for i in range(12):
    #         print(heads_au[i])
    #         au_list = df[df['path'].str.contains(one_video + '/') & (df[heads_au[i]] == 1)] #& (df[heads_au[i]] == 1)
    #
    #         #[df['path'].str.contains(one_video)].loc[:, ['path']]
    #         #print(au_list)
    #         if au_list.empty:
    #             org_dic[one_video][heads_au[i]] = []
    #         else:
    #             org_dic[one_video][heads_au[i]] = au_list

    df = pd.read_csv('./sample_24fps_train.csv')
    #AU_dict = {'AU1':{'v1':AU1_v1_list}, 'AU2':{'v1':AU1_v1_list}}
    AU_dict = {}
    #au_list = []
    for i in range(12):
        au_path = df[df[heads_au[i]] == 1].loc[:, 'path'].reset_index(drop=True).values
        #au_list.append(au_path)
        #print(au_path)
        AU_dict[heads_au[i]] = au_path

        au_all_negative_path = df[(df[heads_au[0]] == 0) & (df[heads_au[1]] == 0)  & (df[heads_au[2]] == 0 ) & (df[heads_au[3]] == 0 ) & (df[heads_au[4]] == 0 )& (df[heads_au[5]] == 0) & (df[heads_au[6]] == 0 )& (df[heads_au[7]] == 0) & (df[heads_au[8]] == 0)& (df[heads_au[9]] == 0) & (df[heads_au[10]] == 0) & (df[heads_au[11]] == 0)].loc[:, 'path'].reset_index(drop=True).values
        AU_dict['neg'] = au_all_negative_path
        # AU_dict[heads_au[i]] = {}

        # for one_video in all_video_list:
        #     au_path_ = []
        #     for j in range(len(au_path)):
        #         if one_video == au_path[j].split('/')[0]:
        #             au_path_.append(au_path[j])
        #     AU_dict[heads_au[i]][one_video] = au_path_
        # #au_list = df[df[heads_au[i]] == 1].loc[:, 'path']

    # # start sampling
    # all_sampler_len = len(df.loc[:, 'path'].reset_index(drop=True).values)
    # one_batch_sampler = []
    # batch_size = 96 #72pos(12*6)     24neg
    # for i in range(all_sampler_len // batch_size):
    #     #sample pos
    #     for j in range(12):
    #         au = heads_au[j]
    #         random_four_videos_index = np.random.randint(len(all_video_list), size = 4)
    #
    #         for k in range(6):
    #             vi_name = all_video_list[k]
    #             my_list = AU_dict[au][vi_name]

    all_sampler_len = len(df.loc[:, 'path'].reset_index(drop=True).values)
    all_batch_sampler = []
    batch_size = 96 #72pos(12*6)     24neg
    for b_num in range(all_sampler_len // batch_size):
        one_batch_sampler = []
        for au_num in range(12):
            au = heads_au[au_num]
            au_list = AU_dict[au]
            six_index = np.random.randint(len(au_list), size = 6)
            for i in range(6):
                one_batch_sampler.append(au_list[six_index[i]])
        neg_index = np.random.randint(len(AU_dict['neg']), size = 24)
        for j in range(len(neg_index)):
            one_batch_sampler.append(AU_dict['neg'][neg_index[j]])
        print(len(one_batch_sampler))
        all_batch_sampler.extend(one_batch_sampler)
    #print(all_batch_sampler)

    # add labels
    new_data = []

    for m in range(len(all_batch_sampler)):
        #print(all_batch_sampler[m])
        new_df = df[(df['path'].str.contains(all_batch_sampler[m]))].loc[:, heads].reset_index(drop=True).values.tolist()
        #print(new_df)
        new_data.extend(new_df)
        #print(new_data)
    sampled_data_df = pd.DataFrame(new_data, columns=heads)
    sampled_data_df.to_csv('./batch_sampler_train.csv', index=None)







    #print(AU_dict['AU1'])

    # df[df[]]
    # AU_dict = {}
    # for i in range(12):
    #     au = heads_au[i]
    #     AU_dict[au]={}



def make_batch_csv_index():
    #make all video list
    all_video_list = os.listdir('/user3/Aff-Wild2/cropped_aligned')
    # batch_num
    # make dict ={'video_name':{'AU1':AU1_list, 'AU2':AU2_list, 'AU3':AU3_list, 'all_neg':all_neg_list}
    org_dic = {}
    heads = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    heads_au = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    # for one_video in all_video_list:
    #     print(one_video)
    #     org_dic[one_video] = {}
    #     df = pd.read_csv('./sample_24fps_train.csv')
    #     for i in range(12):
    #         print(heads_au[i])
    #         au_list = df[df['path'].str.contains(one_video + '/') & (df[heads_au[i]] == 1)] #& (df[heads_au[i]] == 1)
    #
    #         #[df['path'].str.contains(one_video)].loc[:, ['path']]
    #         #print(au_list)
    #         if au_list.empty:
    #             org_dic[one_video][heads_au[i]] = []
    #         else:
    #             org_dic[one_video][heads_au[i]] = au_list

    df = pd.read_csv('./sample_24fps_train.csv')
    #AU_dict = {'AU1':{'v1':AU1_v1_list}, 'AU2':{'v1':AU1_v1_list}}
    AU_dict = {}
    #au_list = []
    for i in range(12):
        au_path = df[df[heads_au[i]] == 1].index.values
        #au_list.append(au_path)
        print(au_path)
        AU_dict[heads_au[i]] = au_path

        au_all_negative_path = df[(df[heads_au[0]] == 0) & (df[heads_au[1]] == 0)  & (df[heads_au[2]] == 0 ) & (df[heads_au[3]] == 0 ) & (df[heads_au[4]] == 0 )& (df[heads_au[5]] == 0) & (df[heads_au[6]] == 0 )& (df[heads_au[7]] == 0) & (df[heads_au[8]] == 0)& (df[heads_au[9]] == 0) & (df[heads_au[10]] == 0) & (df[heads_au[11]] == 0)].index.values
        AU_dict['neg'] = au_all_negative_path
        # AU_dict[heads_au[i]] = {}

        # for one_video in all_video_list:
        #     au_path_ = []
        #     for j in range(len(au_path)):
        #         if one_video == au_path[j].split('/')[0]:
        #             au_path_.append(au_path[j])
        #     AU_dict[heads_au[i]][one_video] = au_path_
        # #au_list = df[df[heads_au[i]] == 1].loc[:, 'path']

    # # start sampling
    # all_sampler_len = len(df.loc[:, 'path'].reset_index(drop=True).values)
    # one_batch_sampler = []
    # batch_size = 96 #72pos(12*6)     24neg
    # for i in range(all_sampler_len // batch_size):
    #     #sample pos
    #     for j in range(12):
    #         au = heads_au[j]
    #         random_four_videos_index = np.random.randint(len(all_video_list), size = 4)
    #
    #         for k in range(6):
    #             vi_name = all_video_list[k]
    #             my_list = AU_dict[au][vi_name]

    all_sampler_len = len(df.index.values)
    all_batch_sampler = []
    batch_size = 96 #72pos(12*6)     24neg
    for b_num in range(all_sampler_len // batch_size):
        one_batch_sampler = []
        for au_num in range(12):
            au = heads_au[au_num]
            au_list = AU_dict[au]
            six_index = np.random.randint(len(au_list), size = 6)
            for i in range(6):
                one_batch_sampler.append(au_list[six_index[i]])
        neg_index = np.random.randint(len(AU_dict['neg']), size = 24)
        for j in range(len(neg_index)):
            one_batch_sampler.append(AU_dict['neg'][neg_index[j]])
        #print(len(one_batch_sampler))
        all_batch_sampler.extend(one_batch_sampler)
    #print(all_batch_sampler)

    # add labels
    new_data = []
    new_data = df.loc[all_batch_sampler, heads].values
    sampled_data_df = pd.DataFrame(new_data, columns=heads)
    sampled_data_df.to_csv('./batch_sampler_train.csv', index=None)







    #print(AU_dict['AU1'])

    # df[df[]]
    # AU_dict = {}
    # for i in range(12):
    #     au = heads_au[i]
    #     AU_dict[au]={}

def make_batch_csv_index_sampler_more_sample_15():
    #make all video list
    all_video_list = os.listdir('/user3/Aff-Wild2/cropped_aligned')
    # batch_num
    # make dict ={'video_name':{'AU1':AU1_list, 'AU2':AU2_list, 'AU3':AU3_list, 'all_neg':all_neg_list}
    org_dic = {}
    heads = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    heads_au = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    # for one_video in all_video_list:
    #     print(one_video)
    #     org_dic[one_video] = {}
    #     df = pd.read_csv('./sample_24fps_train.csv')
    #     for i in range(12):
    #         print(heads_au[i])
    #         au_list = df[df['path'].str.contains(one_video + '/') & (df[heads_au[i]] == 1)] #& (df[heads_au[i]] == 1)
    #
    #         #[df['path'].str.contains(one_video)].loc[:, ['path']]
    #         #print(au_list)
    #         if au_list.empty:
    #             org_dic[one_video][heads_au[i]] = []
    #         else:
    #             org_dic[one_video][heads_au[i]] = au_list

    # df = pd.read_csv('./sample_24fps_train.csv')
    df = pd.read_csv('./sample_24fps_train_clean_0316.csv')
    #AU_dict = {'AU1':{'v1':AU1_v1_list}, 'AU2':{'v1':AU1_v1_list}}
    AU_dict = {}
    #au_list = []
    for i in range(12):
        # if i == 9:#AU24
        #     au_path = df[(df[heads_au[i]] == 1) & (df[heads_au[i+1]] == 0)].index.values
        # elif i == 8: #AU23
        #     au_path = df[(df[heads_au[i]] == 1) & (df[heads_au[10]] == 0)].index.values
        # elif i == 7: #AU15
        #     au_path = df[(df[heads_au[i]] == 1) & (df[heads_au[10]] == 0)].index.values
        # else:
        au_path = df[df[heads_au[i]] == 1].index.values
        #au_list.append(au_path)
        #print(au_path)
        AU_dict[heads_au[i]] = au_path

        au_all_negative_path = df[(df[heads_au[0]] == 0) & (df[heads_au[1]] == 0)  & (df[heads_au[2]] == 0 ) & (df[heads_au[3]] == 0 ) & (df[heads_au[4]] == 0 )& (df[heads_au[5]] == 0) & (df[heads_au[6]] == 0 )& (df[heads_au[7]] == 0) & (df[heads_au[8]] == 0)& (df[heads_au[9]] == 0) & (df[heads_au[10]] == 0) & (df[heads_au[11]] == 0)].index.values
        AU_dict['neg'] = au_all_negative_path
        # AU_dict[heads_au[i]] = {}

        # for one_video in all_video_list:
        #     au_path_ = []
        #     for j in range(len(au_path)):
        #         if one_video == au_path[j].split('/')[0]:
        #             au_path_.append(au_path[j])
        #     AU_dict[heads_au[i]][one_video] = au_path_
        # #au_list = df[df[heads_au[i]] == 1].loc[:, 'path']

    # # start sampling
    # all_sampler_len = len(df.loc[:, 'path'].reset_index(drop=True).values)
    # one_batch_sampler = []
    # batch_size = 96 #72pos(12*6)     24neg
    # for i in range(all_sampler_len // batch_size):
    #     #sample pos
    #     for j in range(12):
    #         au = heads_au[j]
    #         random_four_videos_index = np.random.randint(len(all_video_list), size = 4)
    #
    #         for k in range(6):
    #             vi_name = all_video_list[k]
    #             my_list = AU_dict[au][vi_name]

    all_sampler_len = len(df.index.values)
    all_batch_sampler = []
    batch_size = 96 #72pos(12*6)     24neg
    for b_num in range(all_sampler_len // batch_size):
        one_batch_sampler = []
        for au_num in range(12):
            #if au_num == 6 or au_num == 7 or au_num == 8 or au_num == 9 or au_num == 1:
            if au_num == 7:
                au = heads_au[au_num]
                au_list = AU_dict[au]
                six_index = np.random.randint(len(au_list), size = 72)
                for i in range(72):
                    one_batch_sampler.append(au_list[six_index[i]])
        neg_index = np.random.randint(len(AU_dict['neg']), size = 21)
        for j in range(len(neg_index)):
            one_batch_sampler.append(AU_dict['neg'][neg_index[j]])
        print(np.sum(df.loc[one_batch_sampler, heads_au].values, axis = 0))
        all_batch_sampler.extend(one_batch_sampler)
    #print(all_batch_sampler)

    # add labels
    new_data = []
    new_data = df.loc[all_batch_sampler, heads].values
    sampled_data_df = pd.DataFrame(new_data, columns=heads)
    sampled_data_df.to_csv('./batch_sampler_train.csv', index=None)







    #print(AU_dict['AU1'])

    # df[df[]]
    # AU_dict = {}
    # for i in range(12):
    #     au = heads_au[i]
    #     AU_dict[au]={}

def make_batch_csv_index_sampler_more_sample_15_save_img():
    #make all video list
    all_video_list = os.listdir('/user3/Aff-Wild2/cropped_aligned')
    # batch_num
    # make dict ={'video_name':{'AU1':AU1_list, 'AU2':AU2_list, 'AU3':AU3_list, 'all_neg':all_neg_list}
    org_dic = {}
    heads = ['path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
    heads_au = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    # for one_video in all_video_list:
    #     print(one_video)
    #     org_dic[one_video] = {}
    #     df = pd.read_csv('./sample_24fps_train.csv')
    #     for i in range(12):
    #         print(heads_au[i])
    #         au_list = df[df['path'].str.contains(one_video + '/') & (df[heads_au[i]] == 1)] #& (df[heads_au[i]] == 1)
    #
    #         #[df['path'].str.contains(one_video)].loc[:, ['path']]
    #         #print(au_list)
    #         if au_list.empty:
    #             org_dic[one_video][heads_au[i]] = []
    #         else:
    #             org_dic[one_video][heads_au[i]] = au_list

    df = pd.read_csv('./sample_24fps_train.csv')
    #df = pd.read_csv('./sample_24fps_train_clean_0314.csv')
    #AU_dict = {'AU1':{'v1':AU1_v1_list}, 'AU2':{'v1':AU1_v1_list}}
    AU_dict = {}
    #au_list = []
    for i in range(12):
        if i == 9:#AU24 + AU25
            au_path = df[(df[heads_au[i]] == 1) & (df[heads_au[i+1]] == 1)].index.values
            AU_dict[heads_au[i]] = au_path


            # add labels
            new_data = df.loc[au_path, ['path']].values
            for one_path in new_data:
                pic_path = '/user3/Aff-Wild2/cropped_aligned/' + one_path
                os.system('cp ' + pic_path + ' /user3/Aff-Wild2/sample_analyze/AU24+AU25/')







    #print(AU_dict['AU1'])

    # df[df[]]
    # AU_dict = {}
    # for i in range(12):
    #     au = heads_au[i]
    #     AU_dict[au]={}






if __name__ == '__main__':
    image_path = '/root/autodl-tmp/data/cropped_aligned/'
    au_label_path = '/root/autodl-tmp/data/annotations/AU_Detection_Challenge'

    make_112_au_data_modify()
    # make_112_au_data(image_path, au_label_path, train=True)
    # make_112_au_data(image_path, au_label_path, train=False)

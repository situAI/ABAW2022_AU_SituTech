import os
import pandas as pd
import tqdm


def post_results(txts_dir_path, save_dir_path):
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    au_list = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
    for txt in tqdm.tqdm(os.listdir(txts_dir_path)):
        datas = pd.read_csv(os.path.join(txts_dir_path, txt))
        labels = datas.loc[:, ['AU' + str(i) for i in au_list]].values
        for i in range(len(labels)):
            if i == 0 and int(labels[i][1]) == -1:
                for i in range(len(labels)):
                    if int(labels[i][1]) != -1:
                        for j in range(i):
                            labels[j] = labels[i]
                        break
            else:
                if int(labels[i][1]) == -1:
                    labels[i] = labels[i - 1]
                else:
                    continue
        df = pd.DataFrame(labels.tolist(), columns=['AU' + str(k) for k in au_list])
        df.to_csv(os.path.join(save_dir_path, txt), index=None)


if __name__ == '__main__':
    # post_results("/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_situ_data/test_merge_txt/",
    #              '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_situ_data/post_processing_test_merge_txt/')
    # post_results("/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_situ_data/test_merge_logits_txt/",
    #              '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_situ_data/post_processing_test_merge_logits_txt/')
    post_results("/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse3/test_merge_txt/",
                 '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse3/post_processing_test_merge_txt/')
    post_results(
        "/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse3/test_merge_logits_txt/",
        '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse3/post_processing_test_merge_logits_txt/')
    post_results("/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse4/test_merge_txt/",
                 '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse4/post_processing_test_merge_txt/')
    post_results(
        "/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse4/test_merge_logits_txt/",
        '/root/autodl-tmp/qfs/ABAW_Competition_0323/20220323_test_result/submit_confuse4/post_processing_test_merge_logits_txt/')

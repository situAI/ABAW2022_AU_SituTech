def smooth_logits_csv(input_logits = './20220319_concatenate_val_sigmoid_result.csv', input_val_gt = './au_val.csv', output_logits = './20220319_concatenate_val_sigmoid_result_smooth.csv'):
    # read csv
    heads = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

    best_f1 = -1
    best_w = -1
    #for w in [41,31,21,11,5,3]: # AU25: 24, AU2 24,
    # for w in [10,24,10,10,10,10,10,10,1,10,24,10]:
    #     print(w)
    macro_list = []
    for j in range(12):
        # if j == 10 or j == 1:
        #     w = 24
        # elif j == 8:
        #     w = 1
        # else:
        #     w = 10

        if j == 10 or j == 1:
            w = 24
        elif j == 8:
            w = 1
        else:
            w = 10
        print(w)
        all_logits = pd.read_csv(input_logits).loc[:, heads[j]].values
        # if j == 7 or j == 8 or j == 9:
        #     weight = np.ones(all_logits.shape)
        #     weight[all_logits > 0.5] = 1.9
        #     all_logits = all_logits*weight
        #all_logits = all_logits
        for k in range(len(all_logits)):
            if k < int(w / 2):
                all_logits[k] = all_logits[k]
            elif k > (len(all_logits) - int(w / 2)):
                all_logits[k] = all_logits[k]

        df_gt = pd.read_csv(input_val_gt).loc[:, heads[j]].values
        a = all_logits
        n = w

        #one_new_logits = gaussian_filter(a, sigma = n)
        one_new_logits = np.convolve(a, np.ones((n,))/n, mode = 'valid')
        all_logits[int(w / 2): (len(all_logits) - int(w / 2)) + 1] = one_new_logits
        #one_new_logits = all_logits
        one_new_predict_label = (all_logits > 0.5).astype(int)

        gt = df_gt
        f1macro = f1_score(gt, one_new_predict_label, average = 'macro')
        #print(f1macro)
        macro_list.append(f1macro)
    print(macro_list)
    print(np.array(macro_list).mean())

def smooth_logits_txt(dir_logits ='./tools/val_logits_result_txt/au1-75' , dir_smooth_logits = './tools/val_logits_result_txt/au1-75-smooth', dir_smooth_labels = './tools/val_logits_result_txt/au1-75-smooth-labels'):
    # read each logits txt
    all_video_logits_txt = os.listdir(dir_logits)
    for one_txt in all_video_logits_txt:
        all_logits = np.loadtxt(dir_logits + '/'+ one_txt, delimiter = ',', skiprows = 1, dtype = float)
        #print(all_lines[1:10])
        #print(all_logits)
        # smooth each logits txt
        for j in range(12):
            if j == 10 or j == 1:
                w = 24
            elif j == 8:
                w = 1
            else:
                w = 10
            n = w
            one_au_logits = all_logits[:,j]
            for k in range(len(one_au_logits)):
                if k  < int(w/2):
                    one_au_logits[k] = one_au_logits[k]
                elif k > (len(one_au_logits) - int(w/2)):
                    one_au_logits[k] = one_au_logits[k]

            convolve_logits = np.convolve(one_au_logits, np.ones((n,)) / n, mode='valid')
            print(len(convolve_logits))
            print(int(w/2))
            print(len(one_au_logits) - int(w / 2))
            one_au_logits[int(w / 2): (len(one_au_logits) - int(w / 2)) + 1] = convolve_logits

            #all_logits[:, j] = one_new_logits
        np.savetxt(dir_smooth_logits + '/' + one_txt, all_logits, fmt = '%.8f', delimiter = ',', newline = '\n', header = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26', comments = '')
        #print(all_logits)
        all_labels = (all_logits > 0.5).astype(int)
        #print(all_labels)
        np.savetxt(dir_smooth_labels + '/' + one_txt, all_labels, fmt = '%d', delimiter = ',', newline = '\n', header = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26', comments = '')

        # save each logits txt

# Name:         generate_h5
# Description:  generate hdf5 file, for multi_class
# Update:       bymxzh

import h5py
import numpy as np
import random
import glob
from tqdm import tqdm
import read_data as rd
import setting_dir as sd
import os
import re
import pandas as pd

def create_hdf5(wavs, h5_file, samples_labels_dict):
    print("wavs logmels len > ",len(wavs))

    h5f = h5py.File(h5_file, "w")
    # dt = h5py.special_dtype(vlen=str)
    x = h5f.create_dataset("samples_logmels", (128, rd.time_size, 40),
                           maxshape=(None, rd.time_size, 40),
                           dtype=np.float32)
    z = h5f.create_dataset("samples_labels", (128, ), maxshape=(None,),
                           dtype=np.int32)
    samples_logmels = []
    samples_labels = []

    wavs_logmels = rd.read_data(wavs)

    for i in tqdm(range(len(wavs))):
        wav = wavs[i]
        sample_filename=str(wav)
        sample_label = samples_labels_dict[sample_filename]
        sample_logmel = wavs_logmels[i]

        if i != 0 and i % 128 == 0:
            
            x.resize([i, rd.time_size, 40])
            z.resize([i, ])
            # y.resize([i, ])

            samples_logmels = np.array(samples_logmels).reshape(
                (128, rd.time_size, 40))  
            samples_labels = np.array(samples_labels)

            x[i - 128:i] = samples_logmels
            z[i - 128:i] = samples_labels

            samples_logmels = []
            samples_labels = []
            samples_logmels.append(sample_logmel)
            samples_labels.append(sample_label)

        else:
            samples_logmels.append(sample_logmel)
            samples_labels.append(sample_label)


def neg_pos_h5file(sample_list_):
    samples_labels_dict_ = {}
    sample_list_one_ = []
    label_ = 0
    for sublist_ in sample_list_:
        for sample_ in sublist_:
            samples_labels_dict_[str(sample_)] = label_
            # sample_list_one_.extend(sample_)
        label_ += 1
        sample_list_one_.extend(sublist_)
        random.shuffle(sample_list_one_)

    print(len(sample_list_))
    random.shuffle(sample_list_)

    return sample_list_one_, samples_labels_dict_

# # 用于合并多个目录下的样本
# def get_wav(wav_list_):
#     l_len = len(wav_list_)
#     all_list = []
#     for i in range(l_len):
#         sub_list = glob.glob(wav_list_[i])
#         all_list.extend(sub_list)
#     return all_list


# 用于指定目录下样本个数的合并
def wav_ratio(wav_list_,wav_num_):
    all_list_ = []
    for i in range(len(wav_list_)):
        sub_list_0 = glob.glob(wav_list_[i])
        random.shuffle(sub_list_0)
        sub_list_ = sub_list_0[:wav_num_[i]]
        all_list_.append(sub_list_)
    return all_list_


def judgesubdir(path):
    if not os.path.exists(path):
        os.system('mkdir %s'%path)
        print('had mkdir %s'%path)


def judgedir(pathlist_):
    for subpath in pathlist_:
        judgesubdir(subpath)


def judgefileratio(wavlist):
    filenamelist = {}
    for subfile in wavlist:
        fileprefix0 = subfile.split('/')[-1].split('.')[0]
        fileprefix =re.search('\D*',fileprefix0).group()
        if fileprefix not in filenamelist:
            filenamelist[fileprefix] = 1
        elif fileprefix in filenamelist:
            filenamelist[fileprefix] += 1
    return filenamelist


if __name__ == "__main__":

    pathlist = [sd.wav_to_h5_dir,
                os.path.join(sd.wav_to_h5_dir, 'h5_sample/'),
                os.path.join(sd.wav_to_h5_dir, 'h5_result/'),
                os.path.join(sd.wav_to_h5_dir, 'h5_result/weights/')]
    judgedir(pathlist)

    samples_dir = ['/workspace/users/mxzh/output/wakeup_sample_data_v2/negtive_sample/neg1s/*/*/*wav',
                   '/workspace/users/mxzh/output/senswords_sample_data_v1/wav_v2/to_1s_vad/dajie*wav',
                   '/workspace/users/mxzh/output/senswords_sample_data_v1/wav_v2/to_1s_vad/jiuming*wav',
                   '/workspace/users/mxzh/output/sens_micro_sample_data_v1/wav_v1/to_1s_vad/pos/lairen*wav']

    samples_num = [30000,
                   10000,
                   10000,
                   10000]

    samples_list = wav_ratio(samples_dir,samples_num)
    wavs, samples_labels_dict = neg_pos_h5file(samples_list)
    print("wavs length> ", len(wavs))
    print(judgefileratio(wavs))

    create_list = True
    _list_dir = os.path.join(sd.wav_to_h5_dir,"h5_sample/samples_list.csv")
    if create_list == True:
        _list = pd.DataFrame(wavs,index=None,columns=['sample_list'])
        _list.to_csv(_list_dir,index=False)
    # elif create_list == False:
    #     wavs = list(pd.read_csv(_list_dir)['neg_sample_list'].to_numpy())

    create_hdf5(
        wavs, os.path.join(sd.wav_to_h5_dir, "h5_sample/senswords_v01.h5"), samples_labels_dict)

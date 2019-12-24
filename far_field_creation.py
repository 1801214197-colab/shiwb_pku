# coding: utf-8

# Name:        far_field_creation
# Description: 唤醒词数据远场营造
# Author:      Feng Zhijin
# Date:        2019/12/15


import os
import glob
from tqdm import tqdm
import random
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from pydub import AudioSegment


fs = 16000
room_dim = np.array([6.0, 4.0, 4.0])
mic_dim = np.array([[1.5, 2.0, 4.0], [3.0, 2.0, 4.0], [4.5, 2.0, 4.0]]).T
source_people_dim = [3.0, 2.0, 1.75]
source_noise_dim = [3.0, 0.0, 2.0]


def clip(signal, high, low):
    s = signal.copy()

    s[np.where(s > high)] = high
    s[np.where(s < low)] = low

    return s


def normalize(signal, bits=None):
    s = signal.copy()
    s /= np.abs(s).max()

    if bits is not None:
        s *= 2 ** (bits - 1) - 1
        s = clip(s, 2 ** (bits - 1) - 1, -2 ** (bits - 1))

    return s


def far_field_simulate(room_size, audio_path, noise_path, simulate_path):
    audio_anechoic = AudioSegment.from_wav(audio_path)
    voice_db = random.uniform(-20, -15)
    audio_anechoic = audio_anechoic + float(voice_db - audio_anechoic.dBFS)

    if len(audio_anechoic) < 1300:
        len_kong = 1500 - len(audio_anechoic)
        index_kong = int(random.random() * len_kong)
        audio_anechoic = AudioSegment.silent(duration=index_kong) + audio_anechoic + AudioSegment.silent(
            duration=int(len_kong - index_kong))
    elif len(audio_anechoic) < 1500:
        random_num_1 = random.randint(100, 300)
        random_num_2 = random.randint(100, 300)
        audio_anechoic = audio_anechoic.fade_in(random_num_1).fade_out(random_num_2)
    elif len(audio_anechoic) < 1800:
        random_num_1 = random.randint(100, 300)
        random_num_2 = random.randint(100, 300)
        len_kong = len(audio_anechoic) - 1500
        index_kong = int(random.random() * len_kong)
        audio_anechoic = audio_anechoic[index_kong: 1500 + index_kong].fade_in(random_num_1).fade_out(random_num_2)
    else:
        return 0

    # create the shoebox
    room = pra.ShoeBox(
        room_size,
        absorption=0.25,
        fs=fs,
        max_order=17)

    # snr = random.randint(10, 20)
    # if audio_noise != False:
    #     audio_noise = audio_noise + (audio_noise.dBFS - audio_anechoic.dBFS / 10**(snr/10))

    # source location
    room.add_source(source_people_dim, signal=audio_anechoic.get_array_of_samples())
    if noise_path != False:
        # time_0 = time.time()
        noise_anechoic = AudioSegment.from_wav(noise_path)
        # time_1 = time.time()
        noise_anechoic = noise_anechoic + (noise_anechoic.dBFS - audio_anechoic.dBFS / 10 ** (4 / 10))
        # time_2 = time.time()
        noise_index = random.randint(0, len(noise_anechoic) - 2000)

        room.add_source(source_noise_dim,
                        signal=noise_anechoic[noise_index: noise_index + 1500].get_array_of_samples())
        # print(time_2 - time_1, time_1-time_0)

    # mic location
    room.add_microphone_array(
        pra.MicrophoneArray(
            mic_dim,
            fs
            )
        )

    # run ism
    room.simulate()
    datas = room.mic_array.signals
    datas = normalize(datas, bits=16)
    datas = np.array(datas, dtype=np.int16)

    for i in range(len(datas)):
        wavfile.write(simulate_path[: -4] + '_' + str(i) + '.wav', fs, datas[i])

    # room.mic_array.to_wav(simulate_path, norm=True, bitdepth=np.int16)


def far_field_simulate_sound(room_size, audio_anechoic, noise_path, simulate_path):
    voice_db = random.uniform(-20, -15)
    audio_anechoic = audio_anechoic + float(voice_db - audio_anechoic.dBFS)

    # create the shoebox
    room = pra.ShoeBox(
        room_size,
        absorption=0.25,
        fs=fs,
        max_order=17)

    # snr = random.randint(10, 20)
    # if audio_noise != False:
    #     audio_noise = audio_noise + (audio_noise.dBFS - audio_anechoic.dBFS / 10**(snr/10))

    # source location
    room.add_source(source_people_dim, signal=audio_anechoic.get_array_of_samples())
    if noise_path != False:
        # time_0 = time.time()
        noise_anechoic = AudioSegment.from_wav(noise_path)
        # time_1 = time.time()
        noise_anechoic = noise_anechoic + (noise_anechoic.dBFS - audio_anechoic.dBFS / 10 ** (4 / 10))
        # time_2 = time.time()
        noise_index = random.randint(0, len(noise_anechoic) - 1500)

        room.add_source(source_noise_dim,
                        signal=noise_anechoic[noise_index: noise_index + 1300].get_array_of_samples())
        # print(time_2 - time_1, time_1-time_0)

    # mic location
    room.add_microphone_array(
        pra.MicrophoneArray(
            mic_dim,
            fs
            )
        )

    # run ism
    room.simulate()
    datas = room.mic_array.signals
    datas = normalize(datas, bits=16)
    datas = np.array(datas, dtype=np.int16)

    for i in range(len(datas)):
        wavfile.write(simulate_path[: -4] + '_' + str(i) + '.wav', fs, datas[i])


if __name__ == "__main__":
    print("Do far field creation.")

    # word_size_dict = {'word_0': 0, 'word_1': 0, 'word_2': 0, 'word_3': 0, 'word_4': 0, 'word_5': 0, 'word_6': 0}
    word_list = ['word_0', 'word_1', 'word_2', 'word_3', 'word_4']

    # room dimension tall with 3

    noises = glob.glob("../vad_data/background_noises/*.wav")
    for i in word_list[2:]:
        nums = 0
        if not os.path.exists("../data_far/" + i):
            os.mkdir("../data_far/" + i)
        wavs = glob.glob("../vad_data/" + i + "/*.wav")
        for j in tqdm(wavs):
            for k in range(20):
                if k < 15:
                    far_field_simulate(room_dim, j, random.choice(noises),
                                       "../data_far/" + j.split('/')[-2] + '/' + str(nums) + '.wav')
                    nums += 1
                else:
                    far_field_simulate(room_dim, j, False,
                                       "../data_far/" + j.split('/')[-2] + '/' + str(nums) + '.wav')
                    nums += 1

    wav_path = "../../data/AISHELL/aidatatang_200zh/corpus/*/*.wav"
    nums = 0
    wavs = glob.glob(wav_path)
    sound = AudioSegment.silent(duration=0)
    if not os.path.exists("../data_far/word_5"):
        os.mkdir("../data_far/word_5")
    for i in tqdm(range(len(wavs))):
        if nums >= 100000:
            break
        sound += AudioSegment.from_wav(wavs[i])
        if len(sound) >= 1300000:
            for j in range(1000):
                if j % 5 == 0:
                    far_field_simulate_sound(room_dim, sound[j * 1300: (j + 1) * 1300], False,
                                             "../data_far/word_5/" + str(nums) + '.wav')
                else:
                    far_field_simulate_sound(room_dim, sound[j * 1300: (j + 1) * 1300], random.choice(noises),
                                             "../data_far/word_5/" + str(nums) + '.wav')
                nums += 1
            sound = AudioSegment.silent(duration=0)


# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

sample_rate = 16000
time_stride = 1.0    # need_modify
time_size = int((time_stride*16000 - 30*16) / (10*16) + 1)  # 98 liping


def get_log_mel(wav_data):
    # wav_loader = io_ops.read_file(wav)
    # wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=24000).audio
    # wav_decoder_clip = tf.clip_by_value(wav_decoder, -1.0, 1.0)

    wav_data = tf.reshape(wav_data, [-1, 1])
    spectrogram = contrib_audio.audio_spectrogram(wav_data, window_size=480,
                                                  stride=160, magnitude_squared=True)
    # mfcc = contrib_audio.mfcc(spectrogram, 16000, dct_coefficient_count=40)
    num_spectrogram_bins = spectrogram.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 40
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    # mel_ = mel_spectrograms
    log_mel_ = tf.log(mel_spectrograms + 1e-6)
    return log_mel_


def read_data(wavs):
    sess = tf.InteractiveSession()

    wav_placeholder = tf.placeholder(
        tf.float32, [int(sample_rate*time_stride)], name="wav_decoder")

    # wav_placeholder = tf.placeholder(tf.string, [])
    sample_logmel_ = get_log_mel(wav_placeholder)
    for i in tqdm(range(len(wavs))):
        wav = wavs[i]
        wave_data = AudioSegment.from_file(wav)
        wav_normalize = [
            x*1.0/2**15 for x in wave_data.get_array_of_samples()]
        wav_ = np.array(wav_normalize)
        # print(wav,'the shape of wav_normalize----->>>>>>',wav_.shape)

        wav_normalize = np.reshape(wav_normalize, int(sample_rate*time_stride))

        sample_logmel = sess.run(sample_logmel_, feed_dict={
            wav_placeholder: wav_normalize})  # 98 * 40 #bymxzh


        # print("logmel shape> ", sample_logmel.shape)
        # print(sample_logmel, label)
        # print(type(sample_logmel))
        # print(len(sample_logmel))

        # spker_label = wav.split('/')[-1].split('_')[0][4:]
        label = wav.split('/')[-2][4:]

        if i == 0:
            sample_logmels = []
            # lable_list = []
            sample_logmels.append(sample_logmel[0])
            # lable_list.append(label)
        else:
            sample_logmels.append(sample_logmel[0])
            # lable_list.append(label)

    return np.array(sample_logmels)#, lable_list

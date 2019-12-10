# coding: utf-8

# Name:         model_cgru
# Description:  CGRU model within keras
# Update:       lpp

from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D, GRU, Dropout, Reshape
from keras.layers import Flatten
from keras import regularizers
from keras import optimizers
# from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model    # bymxzh
import h5py
from keras.constraints import Constraint
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
import numpy as np
import keras
import read_data as rd
import setting_dir as sd
import os

# max_time = 1.0
# time_size = int((max_time*16000 - 30*16)/(10*16) + 1)  # time_size=98
batch_size = 128


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)
        # 逐元素clip（将超出指定范围的数强制变为边界值），将p中所有元素的值都限制在（-self.c, self.c）范围内

    def get_config(self):       # 此函数后面没有显式调用，因此此函数并没有用到
        return {'name': self.__class__.__name__, 'c': self.c}


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def sre_loss(y_true, y_pred):    # 此损失函数未用到
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    return cross_entropy_mean


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def model_cgru_2fc(samples_logmels_, samples_labels_onehot_, labels_number):
    reg = 0.000001    # bymxzh reg = 0.000001
    constraint = WeightClip(1)    # bymxzh 0.499

    main_input = Input(shape=(rd.time_size, 40, 1), batch_shape=(
        None, rd.time_size, 40, 1), name="main_input")
    print(main_input)

    # public cgru
    # model_cnn_shape:(batch_size,rd.time_size/strides[0],dct_factor/strides[1],filters)=(128,148/4,40/4,16)
    model_cnn = Conv2D(filters=16, kernel_size=(21, 9), strides=(4, 4), padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg),
                       activity_regularizer=regularizers.l2(reg), name="model_cnn"
                       )(main_input)
    print(model_cnn)

    model_cnn_drop = Dropout(rate=0.3)(model_cnn)    # bymxzh rate = 0.3
    print(model_cnn_drop)

    # model_cnn_shape: (batch_size,rd.time_size/strides[0],dct_factor/strides[1]*filters) = (128,37,160)
    # print("reshape size > ",int(rd.time_size/4) + 1)
    model_cnn_reshape = Reshape(
        ((int(rd.time_size/4)+1), 160))(model_cnn_drop)     # need_modify
    print(model_cnn_reshape)
    # model_gru_shape:
    # if return_sequences=True then (batch_size,rd.time_size/strides[0],gru_num) = (128,37,32)
    # if return_sequences=False then (batch_size,rd.time_size/strides[0],1) = (128,37,1)
    model_gru = GRU(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, name='model_gru',
                    kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), dropout=0.3,
                    kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(
                        model_cnn_reshape)  # liping # fengzhijin

    # model_gru_flatten = Flatten()(model_gru)#liping

    print(model_gru)
    # sre_fc_1_shape: (batch_size,64) = (128,64)
    sre_fc_1 = Dense(256, activation='relu', name='sre_fc_1', kernel_constraint=constraint, bias_constraint=constraint)(
        model_gru)  # liping  # fengzhijin
    # sre_fc_2_shape: (batch_size,2) = (128,2)
    # if sre_fc_2's activation = simoid ,dense_output can use '1',if activation=None ,dense_output use '2'
    sre_fc_2 = Dense(2, activation='softmax', name='sre_fc_2',
                     kernel_constraint=constraint, bias_constraint=constraint)(sre_fc_1)
    print(sre_fc_2)

    # model inputs outputs
    with tf.device("cpu:0"):
        # , asr_fc_1, asr_ctc])
        model = Model(inputs=main_input, outputs=[sre_fc_2])

    print('网络形状 > ')
    print(model.summary())

    # gpus
    GPU_counts = 3  # bymxzh
    parallel_model = multi_gpu_model(model, GPU_counts)

    # adam_t = optimizers.Adam(lr=1e-3)
    Nadam = optimizers.Nadam(lr=0.002, beta_1=0.9,
                             beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # 保存模型回调函数
    checkpoint = ParallelModelCheckpoint(model, os.path.join(sd.wav_to_h5_dir, "h5_result/weights/weights.{epoch:02d}.hdf5"), monitor='loss', verbose=1,  # bymxzh
                                         mode='min')
    # 学习率衰减回调函数
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=3, min_lr=1e-8, mode="min", cooldown=2)
    tbCallBack = TensorBoard(
        log_dir=os.path.join(sd.wav_to_h5_dir, "h5_result/log"), histogram_freq=0, write_graph=True, write_images=True)

    # model Iterator,loss,Weighted cost
    parallel_model.compile(loss={'sre_fc_2': "categorical_crossentropy"},  # , 'asr_fc_1': ctc_loss},
                           metrics=['accuracy'], optimizer=Nadam, loss_weights={'sre_fc_2': 1})    # , 'asr_fc_1': 0.5})

    # Model training setup
    parallel_model.fit({"main_input": samples_logmels_}, {'sre_fc_2': samples_labels_onehot_},
                       batch_size=batch_size * GPU_counts, verbose=1,
                       epochs=60, validation_split=0.1, callbacks=[checkpoint, reduce_lr, tbCallBack])  # epochs=100


if __name__ == "__main__":
    # reade data
    print('Loading data...')
    with h5py.File(os.path.join(sd.wav_to_h5_dir, "h5_sample/senswords_v01.h5"), 'r') as hf:
        for key in hf.keys():
            print(hf[key].name)
            print(hf[key].shape)
        samples_logmels = hf["samples_logmels"][:]  # 导入logmel值
        samples_labels = hf["samples_labels"][:]  # 导入正负属性
    print('done.')
    classes = 4
    samples_labels_onehot = keras.utils.to_categorical(
        samples_labels, num_classes=classes)
    # print(samples_labels_onehot)
    print('Train...')
    print(np.shape(samples_logmels), np.shape(samples_labels_onehot))
    # print("first logmel > ",samples_logmels[0])
    # print("last logmel > ",samples_logmels[-1])

    samples_logmels = np.reshape(samples_logmels, (-1, rd.time_size, 40, 1))
    print(samples_labels_onehot[0:10])
    print(samples_labels_onehot[len(samples_labels_onehot)-10:])
    model_cgru_2fc(samples_logmels, samples_labels_onehot, classes)

import numpy as np
from keras.models import Model
from keras.layers import Input,Conv2D, GRU, Reshape, Dense, Dropout, Embedding
from keras.constraints import Constraint
from keras import regularizers
from keras.layers import Flatten,Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
import glob
from keras import backend as K
import tensorflow as tf
import os
import random

import read_data as rd
import setting_dir as sd

np.set_printoptions(threshold=np.inf)

batch_size = 128

class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}

def get_output_function(model, output_layer_index):
    '''
    model: 要保存的模型
    output_layer_index：要获取的那一个层的索引
    '''
    backend_function = K.function([model.layers[0].input], [model.layers[output_layer_index].output])
    # print(backend_function)
    def inner(sample_logmel_):
        # print(sample_logmel_.shape)
        # print(type(sample_logmel_))
        sample_logmel_ = np.array(sample_logmel_).reshape((-1,rd.time_size,40,1))
        vector = backend_function([sample_logmel_])[0]		#bymxzh
        return vector
    return inner

def get_weights(samples_logmels_): #bymxzh

    reg = 0.000001
    constraint = WeightClip(1)

    main_input = Input(shape=(rd.time_size, 40, 1),batch_shape=(None, rd.time_size, 40, 1), name="main_input")       # bymxzh
    # print(main_input)
    # public cgru
    model_cnn = Conv2D(filters=16, kernel_size=(21, 9), strides=(4, 4), padding='same', activation='relu',
                       # kernel_regularizer=regularizers.l2(reg),  # bias_regularizer=regularizers.l2(reg),
                       name="model_cnn")(main_input)

    # print(model_cnn)
    # model_cnn_drop = Dropout(rate=0.3)(model_cnn)

    model_cnn_reshape = Reshape((int(rd.time_size/4) + 1, 160))(model_cnn)   # bymxzh    # need_modify
    # print(model_cnn_reshape)
    model_gru = GRU(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, name='model_gru',  # bymxzh
                    )(model_cnn_reshape)

    # model_gru_flatten = Flatten()(model_gru)

    # print(model_gru)
    sre_fc_1 = Dense(256, activation = 'relu', name = 'sre_fc_1', kernel_constraint=constraint, bias_constraint=constraint)(
        model_gru)
    sre_fc_2 = Dense(4, activation = 'softmax', name = 'sre_fc_2' # , kernel_constraint=constraint,
                     )(sre_fc_1)
    print("sre_fc_2的形状:")
    print(sre_fc_2)

    # load_model
    model = Model(inputs=main_input, outputs=[sre_fc_2, sre_fc_1, model_gru, model_cnn_reshape, model_cnn])   # bymxzh

    model.load_weights(os.path.join(sd.wav_to_h5_dir,"h5_result/weights/weights.60.hdf5"))    # bymxzh
    names = [weight.name for layer in model.layers for weight in layer.weights]

    print(model.summary())  # bymxzh 查看各层名称及维度

    weights = model.get_weights()
    with open(os.path.join(sd.wav_to_h5_dir,"h5_result/model_weight_cgru.h"), 'w') as h:
        for name, weight in zip(names, weights):
            if name == 'model_cnn/kernel:0':
                weight = np.transpose(weight, (3, 0, 1, 2))
            print(name, weight.shape)
            weight_name = name[:-2]
            weight_name = weight_name.replace('/', '_')
            weight_shape = weight.shape
            weight_data = np.reshape(weight, [-1, ])

            h.write("/*The Shape of " + weight_name + " is " + str(weight_shape) + "*/\n")
            if len(weight_shape) == 2:
                h.write("float " + weight_name + "[" + str(weight_shape[0] * weight_shape[1]) + "] = {\n")
            else:
                h.write("float " + weight_name + "[" + str(weight_shape[0]) + "] = {\n")
            for i in range(len(weight_data)):
                h.write(str(float(weight_data[i])) + ',')
                if (i+1) % 5 == 0:
                    h.write("\n")
            h.write("};\n")

    result, out_fc1, out_gru, out_reshape, out_cnn = model.predict(samples_logmels_, batch_size=1, verbose=2)

    model_frame = [out_cnn, out_reshape, out_gru, out_fc1, result]
    n = 1
    with open(os.path.join(sd.wav_to_h5_dir,"h5_result/senswords_cgru_model.txt"), 'w') as hh: #bymxzh

        for i in model_frame:
            outs = model.layers[n].output_shape
            hh.write("第" + str(n) + "层网络输出结果：" + str(i) + "\n" + "网络结构" + str(outs) + "\n")
            n += 1

    return result


def wav_ratio(wav_dir_dict):
    all_list_ = []
    for key, value in wav_dir_dict.items():
        sub_list_ = glob.glob(key)[:value]
        all_list_.extend(sub_list_)
    return all_list_


def judge(vec_, wavs_):

    class_name = ['neg', 'dajie', 'jiuming', 'lairen']
    class_index = {}
    for i in range(len(class_name)):
        class_index[class_name[i]] = 0

    for sub_vec_ in vec_:
        sub_class_index = list(sub_vec_).index(max(list(sub_vec_)))
        class_index[class_name[sub_class_index]] += 1

    # for sub_vec_ in vec_:
    #     if max(list(sub_vec_)) <= 0.3:
    #         class_index[class_name[0]] += 1
    #     else:
    #         class_index[class_name[sub_class_index]] += 1

    return class_index


config=tf.ConfigProto()
config.gpu_options.visible_device_list = "3"

if __name__ == "__main__":
    print("提取模型权重")

    # test_sample_dir = {"/workspace/users/mxzh/output/senswords_sample_data_v1/test_sample/multi_class/neg/*.wav": 100,
    #                    "/workspace/users/mxzh/output/senswords_sample_data_v1/test_sample/multi_class/dajie/*.wav": 100,
    #                    "/workspace/users/mxzh/output/senswords_sample_data_v1/test_sample/multi_class/jiuming/*.wav": 100,
    #                    "/workspace/users/mxzh/output/sens_micro_sample_data_v1/wav_v1/to_1s_vad/pos/lairen*.wav": 100}
    #
    # wavs = random.sample(wav_ratio(test_sample_dir), 400)

    wavs = ["/workspace/users/mxzh/output/wakeup_sample_data_v2/h5_result/first_test_0_1s.wav"]

    samples_logmels  = rd.read_data(wavs)
    print("samples_logmels_shape:", samples_logmels.shape)
    # # print("label_list:",label_list)
    samples_logmels = np.array(samples_logmels).reshape((-1,rd.time_size, 40,1))    # bymxzh
    result_vector = get_weights(samples_logmels)
    print(result_vector)
    print(judge(result_vector,wavs))


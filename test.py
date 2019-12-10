# ��������ģ��
import tensorflow as tf
import os
import tensorflow

import csv
from Nclasses import labels
import numpy as np
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time


# ����Ԥ������
# preprocess_input = tensorflow.keras.applications.resnet50.preprocess_input
# preprocess_input = tf.keras.applications.densenet.preprocess_input
# preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
# preprocess_input = tf.keras.applications.inception_v3.preprocess_input
# preprocess_input = tf.keras.applications.mobilenet.preprocess_input
preprocess_input = tf.keras.applications.xception.preprocess_input

# ����ģ����
# ResNet50 = tensorflow.keras.applications.ResNet50
# DenseNet121 = tf.keras.applications.DenseNet121
# DenseNet169 = tf.keras.applications.DenseNet169
# DenseNet201 = tf.keras.applications.DenseNet201
# InceptionResNetV2 = tf.keras.applications.InceptionResNetV2
# InceptionV3 = tf.keras.applications.InceptionV3
# MobileNet = tf.keras.applications.MobileNet
# NASNetLarge = tf.keras.applications.NASNetLarge
# VGG16 = tf.keras.applications.VGG16
# VGG19 = tf.keras.applications.VGG19
Xception = tf.keras.applications.Xception

# ����image��ģ��
image = tf.keras.preprocessing.image

# �����֤��ͼ���ַ�ͱ�ǩ
image_paths, image_labels = utils.get_paths_labels()

# ʵ����ģ����
# resnet50 = ResNet50()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = InceptionResNetV2()
# model = InceptionV3()
# model = MobileNet()
# model = NASNetLarge()
# model = VGG16()
# model = VGG19()
model = Xception()

# ��¼top1��top5��ȷ��Ŀ
top1_cnt = 0
top5_cnt = 0

# ���ڼ�¼������ʱ
begin_time = time.clock()

# ��¼��ͼ����Ŀ
cnt = 0

for image_path, image_label in zip(image_paths, image_labels):
    # Ԥ����ʼ
    raw_image = image.img_to_array(image.load_img(image_path))
    image_copy = np.copy(raw_image)
    shape = image_copy.shape
    h, w = shape[0], shape[1]
    if h > w:
        h_start = (h - w) // 2
        image_copy = image_copy[h_start:h_start+w, :]
    else:
        w_start = (w - h) // 2
        image_copy = image_copy[:, w_start:w_start+h]
    image_resized = cv2.resize(
        image_copy, (299, 299), interpolation=cv2.INTER_CUBIC)
    processed_image = preprocess_input(
        image_resized).reshape((1, 299, 299, -1))

    # Ԥ�����������ģ��ʵ������Ԥ��
    res = model.predict(processed_image)

    # ����õ��Ľ�������ǩ���жԱ�
    # argsort()��numpy.ndarray�ĳ�Ա��������С�������򣬷�������õĸ�Ԫ�ض�Ӧ������ǰ���±�
    top5 = res.argsort().squeeze()[-1:-6:-1]

    if image_label in top5:
        top5_cnt += 1

    if image_label == top5[0]:
        top1_cnt += 1

    cnt += 1

    # ÿ10000��ͼƬ�����һ�β��Ժ�ʱ
    if cnt % 10000 == 0:
        end_time = time.clock()
        print('%d steps: %f' % (cnt, end_time - begin_time))
        begin_time = end_time

print('top1 accuracy:', top1_cnt / 50000)
print('top5 accuracy:', top5_cnt / 50000)

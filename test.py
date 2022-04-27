import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import tensorflow as tf
import os

# --- invcrf_list
CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))

"""dorf and the inverse rf test"""
# with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
#     lines = f.readlines()
#     lines = [line.strip() for line in lines]

# crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
# crf_list = np.float32([ele.split() for ele in crf_list])
# np.random.RandomState(730).shuffle(crf_list)

# train_crf_list = crf_list[0]

# print(np.shape(train_crf_list))

# s, = train_crf_list.shape
# train_crf_list[0] = 0.0
# train_crf_list[-1] = 1.0
# f= interp1d(train_crf_list, np.linspace(0.0, 1.0, num=s))
# train_invcrf_list = f(np.linspace(0.0, 1.0, num=s))

# print(np.shape(train_invcrf_list))

# plt.plot(train_crf_list, np.linspace(0.0, 1.0, num=s), 'o', train_invcrf_list, np.linspace(0.0, 1.0, num=s), '-')
# plt.show()

"""t_test"""
# get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
# t = get_t_list(7)

# hdr = tf.ones((1,5,5,3))
# _hdr_t = hdr * tf.reshape(t, [1, 1, 1, 1])

# tf.print(_hdr_t, summarize=-1)

"""Soft histogram"""
def histogram_layer(img, max_bin):
    # histogram branch
    tmp_list = []
    
    _threshold = 1. / max_bin
    condition = lambda x: tf.less(x, _threshold)

    for i in range(1, max_bin + 1):
        # TODO correct the formula
        distance = tf.abs(img - tf.divide((2.*i -1.), 2.*max_bin))
        histo = tf.where(condition(distance) , tf.subtract(1., tf.multiply(distance, max_bin)), 0)
        tmp_list.append(histo)

    histogram_tensor = tf.concat(tmp_list, -1)
    return histogram_tensor
    # histogram_tensor = tf.layers.average_pooling2d(histogram_tensor, 16, 1, 'same')

"""datset"""
import pickle
def _load_pkl(name):
    with open(os.path.join(CURR_PATH_PREFIX, name + '.pkl'), 'rb') as f:
        out = pickle.load(f)
    return out
i_dataset_train_posfix_list = _load_pkl('i_dataset_train')

# print(i_dataset_train_posfix_list)

import cv2

DS = "/home/cvnar2/Desktop/nvme/singleHDR/SingleHDR_training_data/HDR-Synth"

batch = []
for idx, path in enumerate(i_dataset_train_posfix_list):
    
    if idx == 8 : break
    
    print(path)
    paths = os.path.join(DS, path)
    img = cv2.imread(paths, -1)
    img = cv2.resize(img, (256,256))
    batch.append(img)
    

batch = tf.convert_to_tensor(batch)

tf.print(tf.shape(batch))

res = histogram_layer(batch, 4)

tf.print(tf.shape(res))

plt.imshow(res[0,:,:,1])
plt.show()


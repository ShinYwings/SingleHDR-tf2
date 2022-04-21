import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import tensorflow as tf
import os

# --- invcrf_list
CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))

"""dorf and the inverse rf test"""
with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
crf_list = np.float32([ele.split() for ele in crf_list])
np.random.RandomState(730).shuffle(crf_list)

train_crf_list = crf_list[0]

print(np.shape(train_crf_list))

s, = train_crf_list.shape
train_crf_list[0] = 0.0
train_crf_list[-1] = 1.0
f= interp1d(train_crf_list, np.linspace(0.0, 1.0, num=s))
train_invcrf_list = f(np.linspace(0.0, 1.0, num=s))

print(np.shape(train_invcrf_list))

plt.plot(train_crf_list, np.linspace(0.0, 1.0, num=s), 'o', train_invcrf_list, np.linspace(0.0, 1.0, num=s), '-')
plt.show()

"""t_test"""
# get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
# t = get_t_list(7)

# hdr = tf.ones((1,5,5,3))
# _hdr_t = hdr * tf.reshape(t, [1, 1, 1, 1])

# tf.print(_hdr_t, summarize=-1)
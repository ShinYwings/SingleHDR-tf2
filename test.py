import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import tensorflow as tf
import os
import tf_utils

from numpy.random import randint

AUTO = tf.data.AUTOTUNE
# --- invcrf_list
CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 5
IMSHAPE = (32,128,3)
RENDERSHAPE = (64,64,3)
np.random.seed(3)

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'render': tf.io.FixedLenFeature([], tf.string),
        'azimuth' : tf.io.FixedLenFeature([], tf.float32),
        'elevation' : tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    hdr = tf.io.decode_raw(example['image'], tf.float32)
    hdr = tf.reshape(hdr, IMSHAPE)
    hdr = hdr[:,:,::-1]
    return hdr

def configureDataset(dirpath):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecord"), shuffle=False)
    tfrecords_list.extend(a)
    buf_size = len(tfrecords_list)
    print("buf_size", buf_size)
    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)
    ds = ds.shuffle(buffer_size = 10000).take(100).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    
    return ds

"""lavalskydb test"""
def preprocessing(hdr, crf_src, t_src):

    b, h, w, c, = tf_utils.get_tensor_shape(hdr)

    crf_len = len(crf_src)
    t_len = len(t_src)
    
    crf = []
    t = []
    
    for _ in range(b):
        crf.append(crf_src[randint(0,crf_len)])
        t.append(t_src[randint(0,t_len)])
    crf = tf.convert_to_tensor(crf)
    t = tf.convert_to_tensor(t)
    print("crf  : ", tf.shape(crf))
    print("t  : ", tf.shape(t))
        
    _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                                    dtype=tf.float32, seed=1)
    sigma_c = 0.005 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
    noise_s_map = sigma_s * _hdr_t
    noise_s = tf.random.normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * tf.random.normal(shape=tf.shape(_hdr_t), seed=1)
    temp_x = temp_x + noise_c
    _hdr_t = tf.nn.relu(temp_x)

    # Dynamic range clipping
    clipped_hdr_t = _clip(_hdr_t)

    # Camera response function
    ldr = tf_utils.apply_rf(clipped_hdr_t, crf)

    # Quantization and JPEG compression
    quantized_hdr = tf.round(ldr * 255.0)
    quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
    jpeg_img_list = []
    for i in range(BATCH_SIZE):
        II = quantized_hdr_8bit[i]
        II = tf.image.adjust_jpeg_quality(II, int(round(float(i)/float(BATCH_SIZE-1)*10.0+90.0)))
        jpeg_img_list.append(II)
    jpeg_img = tf.stack(jpeg_img_list, 0)
    jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0

    # loss mask to exclude over-/under-exposed regions
    # gray = tf.image.rgb_to_grayscale(jpeg_img)
    # over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
    # over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
    # over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
    # under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
    # under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
    # under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
    # extreme_cases = tf.logical_or(over_exposed, under_exposed)
    # loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

    return ldr, jpeg_img_float

"""dorf and the inverse rf test"""
def getDoRF():
    with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
    crf_list = np.float32([ele.split() for ele in crf_list])

    train_crf_list = crf_list

    # print(np.shape(train_crf_list))

    # s, = train_crf_list.shape
    # train_crf_list[0] = 0.0
    # train_crf_list[-1] = 1.0
    # f= interp1d(train_crf_list, np.linspace(0.0, 1.0, num=s))
    # train_invcrf_list = f(np.linspace(0.0, 1.0, num=s))

    # print(np.shape(train_invcrf_list))

    # plt.plot(train_crf_list, np.linspace(0.0, 1.0, num=s), 'o', train_invcrf_list, np.linspace(0.0, 1.0, num=s), '-')
    # plt.show()

    return train_crf_list

"""t_test"""
def get_T():
    get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
    t = get_t_list(600)

    # tf.print(_hdr_t, summarize=-1)

    return t


if __name__=="__main__":
    import os
    import matplotlib.pyplot as plt
    import cv2

    cur = os.getcwd()
    dir = os.path.join(cur, "dataset/tfrecord/train")
    ds = configureDataset(dir)

    crf = getDoRF()
    t = get_T()

    for hdr in ds:
        i = 0
        plt.figure(figsize=(64,256))

        # ldr, jpeg_img_float, loss_mask = tf.py_function(preprocessing, [hdr, crf, t], [tf.float32, tf.float32, tf.float32])

        ldr, jpeg_img_float = preprocessing(hdr, crf, t)
        for h, l, j in zip(hdr, ldr, jpeg_img_float):
            ax = plt.subplot(BATCH_SIZE,3,i+1)
            plt.imshow(h)
            ax.title.set_text("hdr")
            plt.axis('off')
            ax = plt.subplot(BATCH_SIZE,3,i+2)
            plt.imshow(l)
            ax.title.set_text("ldr")
            plt.axis('off')
            ax = plt.subplot(BATCH_SIZE,3,i+3)
            plt.imshow(j)
            ax.title.set_text("jpeg")
            plt.axis('off')

            i+=3
            h = h[:,:,::-1]
            l = tf.cast(l[:,:,::-1] * 255., dtype=tf.int8)
            j = tf.cast(j[:,:,::-1] * 255., dtype=tf.int8)
            cv2.imwrite(f"tmp/hdr_{i}.exr", h.numpy())
            cv2.imwrite(f"tmp/ldr_{i}.jpg", l.numpy())
            cv2.imwrite(f"tmp/jpeg_{i}.jpg", j.numpy())
        plt.show()

        

"""datset"""
# import pickle
# def _load_pkl(name):
#     with open(os.path.join(CURR_PATH_PREFIX, name + '.pkl'), 'rb') as f:
#         out = pickle.load(f)
#     return out
# i_dataset_train_posfix_list = _load_pkl('i_dataset_train')


# import cv2

# DS = "/home/cvnar2/Desktop/nvme/singleHDR/SingleHDR_training_data/HDR-Synth"

# batch = []
# for idx, path in enumerate(i_dataset_train_posfix_list):
    
#     if idx == 8 : break
    
#     print(path)
#     paths = os.path.join(DS, path)
#     img = cv2.imread(paths, -1)
#     img = cv2.resize(img, (256,256))
#     batch.append(img)

# batch = tf.convert_to_tensor(batch)

# tf.print(tf.shape(batch))

# res = histogram_layer(batch, 4)

# tf.print(tf.shape(res))

# plt.imshow(res[0,:,:,1])
# plt.show()


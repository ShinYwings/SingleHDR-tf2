import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import os

class resBlock_type1(Model):

    def __init__(self, branch1_filter = "branch1", branch2_filters = [], kernel_size=(1,1), strides=(1,1), padding="padding"):
        super(resBlock_type1, self).__init__()
        
        # branch #1
        self.conv1 = tf.keras.layers.Conv2D(branch1_filter, kernel_size=(1,1), strides=strides, use_bias=False, padding=padding)
        self.norm1   = tf.keras.layers.BatchNormalization()

        # branch #2
        self.conv2 = tf.keras.layers.Conv2D(branch2_filters[0], kernel_size=(1,1), strides=strides, use_bias=False, padding=padding)
        self.norm2   = tf.keras.layers.BatchNormalization()
        self.act2  = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(branch2_filters[1], kernel_size=(3,3), strides=(1,1), use_bias=False, padding=padding)
        self.norm3   = tf.keras.layers.BatchNormalization()
        self.act3  = tf.keras.layers.ReLU()

        self.conv4 = tf.keras.layers.Conv2D(branch2_filters[2], kernel_size=(1,1), strides=(1,1), use_bias=False, padding=padding)
        self.norm4   = tf.keras.layers.BatchNormalization()

        self.act4  = tf.keras.layers.ReLU() # bn1 + bn4
        
    def call(self, x, training = "training"):
        
        # branch #1
        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1, training=training)
        
        # branch #2
        conv2 = self.conv2(x)
        norm2 = self.norm2(conv2, training=training)
        act2  = self.act2(norm2)

        conv3 = self.conv3(act2)
        norm3 = self.norm3(conv3, training=training)
        act3  = self.act3(norm3)

        conv4 = self.conv4(act3)
        norm4 = self.norm4(conv4, training=training)

        output = tf.add(norm1, norm4)
        return self.act4(output)

class resBlock_type2(Model):

    def __init__(self, filters = [], kernel_size=(1,1), strides=(1,1), padding="padding"):
        super(resBlock_type2, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=(1,1), strides=(1,1), use_bias=False, padding=padding)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.act1  = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(3,3), strides=(1,1), use_bias=False, padding=padding)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.act2  = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=(1,1), strides=(1,1), use_bias=False, padding=padding)
        self.norm3 = tf.keras.layers.BatchNormalization()

        self.act3  = tf.keras.layers.ReLU() # bn1 + bn4
        
    def call(self, x, training = "training"):
        
        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1, training=training)
        act1  = self.act1(norm1)

        conv2 = self.conv2(act1)
        norm2 = self.norm2(conv2, training=training)
        act2  = self.act2(norm2)

        conv3 = self.conv3(act2)
        norm3 = self.norm3(conv3, training=training)

        output = tf.add(x, norm3)

        return self.act3(output)

class crfFeatureNet(Model):
    
    def __init__(self, padding = "SAME"):

        super(crfFeatureNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(2,2), padding=padding)
        self.norm1   = tf.keras.layers.BatchNormalization()
        self.act1  = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding=padding)
        
        self.res1 = resBlock_type1(branch1_filter = 256, branch2_filters=[64, 64, 256], kernel_size=(1,1), strides=(1,1), padding=padding)
        self.res2 = resBlock_type2(filters=[64, 64, 256], kernel_size=(1,1), strides=(1,1), padding=padding)
        self.res3 = resBlock_type2(filters=[64, 64, 256], kernel_size=(1,1), strides=(1,1), padding=padding)

        self.res4 = resBlock_type1(branch1_filter = 512, branch2_filters=[128, 128, 512], kernel_size=(1,1), strides=(2,2), padding=padding)
        self.res5 = resBlock_type2(filters=[128, 128, 512], kernel_size=(1,1), strides=(1,1), padding=padding)

        # self.pool2 = tf.keras.layers.AveragePooling2D((7,7),strides=1, padding='vaild')

    def call(self, ldr, training="training"):

        conv1 = self.conv1(ldr)
        norm1 = self.norm1(conv1, training)
        act1  = self.act1(norm1)
        pool1 = self.pool1(act1)

        res1  = self.res1(pool1, training)
        res2  = self.res2(res1, training)
        res3  = self.res3(res2, training)
        res4  = self.res4(res3, training)
        res5  = self.res5(res4, training)

        return tf.reduce_mean(res5, [1, 2], keepdims=False)

    def overwrite_init(self, sess):

        # np_var
        def refine_np_var(input, output_dict, curr_tag=''):
            if type(input) is dict:
                for key, val in input.items():
                    if 'fc11' not in key:
                        refine_np_var(val, output_dict, curr_tag + '/%s' % key)
            else:
                assert curr_tag not in output_dict
                output_dict[curr_tag] = input

        np_var = {}
        refine_np_var(
            np.load('crf_net_v2.npy', encoding='latin1').item(),
            np_var,
        )

        # tf_var
        def tf_name_2_np_name(tf_name):
            np_name = tf_name
            np_name = np_name.replace(':0', '')
            np_name = np_name.replace('/BatchNorm', '')
            np_name = np_name.replace(f'{self.scope}', '')
            '''
            offset = beta
            scale = gamma
            '''
            np_name = np_name.replace('beta', 'offset')
            np_name = np_name.replace('gamma', 'scale')
            np_name = np_name.replace('moving_variance', 'variance')
            np_name = np_name.replace('moving_mean', 'mean')
            return np_name

        tf_var = {tf_name_2_np_name(var.name): var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope,
        )}

        # chk all
        print(tf_var)

        for key, var in np_var.items():
            print(key)
            assert key in tf_var

        # load all
        for key, var in np_var.items():
            if '/conv1/' not in key:
                tf_var[key].load(var, sess)

        return

class AEInvcrfDecodeNet(Model):

    def __init__(self, n_digit=2):

        super(AEInvcrfDecodeNet, self).__init__()

        self.n_digit = n_digit
        # self.decode_spec = []
        self.s = 1024
        self.n_p = 12
        self.act = tf.nn.tanh
        self.reg = tf.keras.regularizers.L2(l2 = 1e-3)

        self.fc = tf.keras.layers.Dense(self.n_p - 1)

    # [b, s]
    def call(self, feature):  # [b, n_digit]

        # for c in self.decode_spec:
        #     x = tf.keras.layers.Dense(feature, c, activation=self.act, kernel_regularizer=self.reg)
        x = self.fc(feature)  # [b, n_p - 1]
        invcrf = self.invcrf_pca_w_2_invcrf(x)
        # x = tf.concat([x, 1.0 - tf.reduce_sum(x, axis=-1, keepdims=True)], -1) # [b, n_p]
        # x = self._f(x) # [b, s]
        return invcrf

    # [b, n_p]
    def _f(self, p):  
    
        '''
        m =
        x_0^1, x_1^1
        x_0^2, x_1^2
        '''
        m = []
        for i in range(self.n_p):
            m.append([x ** (i + 1) for x in np.linspace(0, 1, num=self.s, dtype='float64')])
        m = tf.constant(m, dtype=tf.float64)  # [n_c, s]
        return tf.matmul(
            p,  # [b, n_p]
            m,  # [n_p, s]
        )  # [b, s]
    
    # b, g0, hinv
    # [1024], [1024], [1024, 11]
    def parse_invemor(self):

        with open(os.path.join('invemor.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        b = self._parse(lines, 'B =')
        g0 = self._parse(lines, 'g0 =')
        hinv = np.stack([self._parse(lines, f'hinv({i + 1})=') for i in range(11)], axis=-1)

        return b, g0, hinv

    # _, B = parse_dorf()
    # _, F0, H = parse_emor()
    def invcrf_pca_w_2_invcrf(self, invcrf_pca_w):  # [b, 11]
    
        _, G0, HINV = self.parse_invemor()
        b, _, = invcrf_pca_w.get_shape()

        invcrf_pca_w = tf.expand_dims(invcrf_pca_w, -1)  # [b, 11, 1]

        G0 = tf.constant(G0)  # [   s   ]
        G0 = tf.reshape(G0, [1, -1, 1])  # [1, s, 1]

        HINV = tf.constant(HINV)  # [   s, 11]
        HINV = tf.expand_dims(HINV, 0)  # [1, s, 11]
        HINV = tf.tile(HINV, [b, 1, 1])  # [b, s, 11]

        # Add G0 to the output of matmul for each batch
        invcrf = G0 + tf.matmul(
            HINV,  # [b, s, 11]
            invcrf_pca_w,  # [b, 11, 1]
        )  # [b, s, 1]

        invcrf = tf.squeeze(invcrf, -1)  # [b, s]

        return invcrf

    @staticmethod
    def _parse(lines, tag):

        for line_idx, line in enumerate(lines):
            if line == tag:
                break

        s_idx = line_idx + 1

        r = []
        for idx in range(s_idx, s_idx + 256): # int(1024 / 4)
            r += lines[idx].split()

        return np.float32(r)

    @staticmethod
    def parse_dorf():

        with open(os.path.join('dorfCurves.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        i = [lines[idx + 3] for idx in range(0, len(lines), 6)]
        b = [lines[idx + 5] for idx in range(0, len(lines), 6)]

        i = [ele.split() for ele in i]
        b = [ele.split() for ele in b]

        i = np.float32(i)
        b = np.float32(b)

        return i, b

    # e, f0, h
    # [1024], [1024], [1024, 11]
    def parse_emor(self):

        with open(os.path.join('emor.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        e = self._parse(lines, 'E =')
        f0 = self._parse(lines, 'f0 =')
        h = np.stack([self._parse(lines, f'h({i + 1})=') for i in range(11)], axis=-1)

        return e, f0, h

    
class model(Model):

    def __init__(self):
        super(model, self).__init__()
        self.crf_feature_net = crfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet()
    
    def call(self, img, training):
        # edge branch
        edge_1 = tf.image.sobel_edges(img)

        edge_1 = tf.reshape(edge_1, [tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], 6])

        tf.summary.image('edge0', edge_1[:, :, :, 0:3])
        tf.summary.image('edge1', edge_1[:, :, :, 3:6])

        # edge_1 = tf.reshape(edge_1, [-1, img.get_shape().as_list()[1], img.get_shape().as_list()[2], 1])

        feature = self.crf_feature_net(
            tf.concat([img, edge_1, self.histogram_layer(img, 4), self.histogram_layer(img, 8), self.histogram_layer(img, 16)], -1), training)
        feature = tf.cast(feature, tf.float32)

        invcrf = self.ae_invcrf_decode_net(feature)
        # [b, 1024]

        invcrf = self._increase(invcrf)
        # [b, 1024]

        invcrf = tf.cast(invcrf, tf.float32)
        # float32

        return invcrf

    def histogram_layer(self, img, max_bin):
        # histogram branch
        tmp_list = []
        
        for i in range(max_bin + 1):
            # TODO correct the formula
            histo = tf.nn.relu(1 - tf.abs(img - i / float(max_bin)) * float(max_bin))
            tmp_list.append(histo)

        histogram_tensor = tf.concat(tmp_list, -1)
        return histogram_tensor
        # histogram_tensor = tf.layers.average_pooling2d(histogram_tensor, 16, 1, 'same')

    @staticmethod
    def _resize_img(img, t):
        _, h, w, _, = img.get_shape()
        ratio = h / w
        pred = tf.greater(ratio, 1.0)
        _round = lambda x: tf.cast(tf.round(x), tf.int32)
        
        t_h = tf.cond(pred, lambda: _round(t * ratio), lambda: t)
        t_w = tf.cond(pred, lambda: t, lambda: _round(t / ratio))

        img = tf.image.resize(img, [t_h, t_w], method=tf.image.ResizeMethod.BILINEAR)
        
        img = tf.image.resize_with_crop_or_pad(img, t, t)
        return img

    @staticmethod
    def _increase(rf):
        g = rf[:, 1:] - rf[:, :-1]
        # [b, 1023]

        min_g = tf.reduce_min(g, axis=-1, keepdims=True)
        # [b, 1]

        # r = tf.nn.relu(1e-6 - min_g)
        r = tf.nn.relu(-min_g)
        # [b, 1023]

        new_g = g + r
        # [b, 1023]

        new_g = new_g / tf.reduce_sum(new_g, axis=-1, keepdims=True)
        # [b, 1023]

        new_rf = tf.cumsum(new_g, axis=-1)
        # [b, 1023]

        new_rf = tf.pad(new_rf, [[0, 0], [1, 0]], 'CONSTANT')
        # [b, 1024]

        return new_rf

    
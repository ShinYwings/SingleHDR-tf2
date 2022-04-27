import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import Model

class down(Model):
    
    def __init__(self, outChannels, kernel_size=(3,3), strides=1, padding="SAME"):
        super(down, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.pool  = tf.keras.layers.AveragePooling2D((2,2), strides=2)
    def call(self, x):
        x = self.pool(x)
        x = tf.nn.leaky_relu(self.conv1(x), 0.1)
        x = tf.nn.leaky_relu(self.conv2(x), 0.1)
        return x

class up(Model):
    
    def __init__(self, outChannels, kernel_size=3, strides=1, padding="SAME"):
        super(up, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, x, skpCn):
        x = tf.image.resize(x, 2*tf.shape(x)[1:3], method=tf.image.ResizeMethod.BILINEAR)
        x = tf.nn.leaky_relu(self.conv1(x), 0.1)
        x = tf.nn.leaky_relu(self.conv2(tf.concat([x, skpCn], -1)), 0.1)

        return x

class model(Model):
    def __init__(self, strides=(1,1), padding="SAME"):
        super(model, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=(7,7), strides=strides, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(7,7), strides=strides, padding=padding)
        self.d2 = down(outChannels=32, kernel_size=(5,5))
        self.d3 = down(outChannels=64, kernel_size=(3,3))
        self.d4 = down(outChannels=128, kernel_size=(3,3))
        self.enc = down(outChannels=128, kernel_size=(3,3))
        
        self.u4 = up(outChannels=128, kernel_size=(3,3))
        self.u3 = up(outChannels=64, kernel_size=(3,3))
        self.u2 = up(outChannels=32, kernel_size=(3,3))
        self.u1 = up(outChannels=16, kernel_size=(3,3))

        self.out = tf.keras.layers.Conv2D(3, kernel_size=(3,3), strides=strides, padding=padding)

    def call(self, input_images, training="training"):

        x = tf.nn.leaky_relu(self.conv1(input_images), 0.1)
        s1 = tf.nn.leaky_relu(self.conv2(x), 0.1)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        x = self.enc(s4)

        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        x = self.out(x)
        res = tf.add(input_images[..., 0:3], x)

        return tf.nn.relu(res)
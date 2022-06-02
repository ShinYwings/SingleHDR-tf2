"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2017, Gabriel Eilertsen.
 " All rights reserved.
 "
 " Redistribution and use in source and binary forms, with or without
 " modification, are permitted provided that the following conditions are met:
 "
 " 1. Redistributions of source code must retain the above copyright notice,
 "    this list of conditions and the following disclaimer.
 "
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 "
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software
 "    without specific prior written permission.
 "
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: TensorFlow autoencoder CNN for HDR image reconstruction.
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: Aug 2017
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class down1(Model):
    
    def __init__(self, outChannels, kernel_size=(3,3), strides=1, padding="SAME"):
        super(down1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.pool  = tf.keras.layers.MaxPool2D((2,2), strides=2, padding=padding)

    def call(self, x):
        
        x = tf.nn.relu(self.conv1(x))
        skip_layer = tf.nn.relu(self.conv2(x))
        x = self.pool(skip_layer)

        return x, skip_layer

class down2(Model):
    
    def __init__(self, outChannels, kernel_size=(3,3), strides=1, padding="SAME"):
        super(down2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv3 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.pool  = tf.keras.layers.MaxPool2D((2,2), strides=2, padding=padding)

    def call(self, x):
        
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        skip_layer = tf.nn.relu(self.conv3(x))
        x = self.pool(skip_layer)

        return x, skip_layer

class up(Model):
    
    def __init__(self, outChannels, kernel_size=(3,3), strides=1, padding="SAME"):
        super(up, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, x, training="training"):
        x = tf.image.resize(x, 2*tf.shape(x)[1:3], method=tf.image.ResizeMethod.BILINEAR)
        x = tf.nn.relu(self.conv1(x))
        x = self.norm1(x, training)
        x = tf.nn.relu(x)

        return x

class skipLayer(Model):
    
    def __init__(self, outChannels, kernel_size=(1,1), strides=1, padding="SAME"):
        super(skipLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, x, sk):
        
        sk = tf.scalar_mul(1.0 / 255, sk)

        x = tf.concat([x, sk], -1)

        x = self.conv1(x)
        
        return x

class model(Model):
    def __init__(self, VGG_MEAN = [103.939, 116.779, 123.68], padding="SAME"):
        super(model, self).__init__()

        self.VGG_MEAN = VGG_MEAN

        self.d1 = down1(outChannels=64)
        self.d2 = down1(outChannels=128)
        self.d3 = down2(outChannels=256)
        self.d4 = down2(outChannels=512)
        self.d5 = down2(outChannels=512)
        
        self.conv1 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding=padding)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.actv1 = tf.keras.layers.ReLU()

        self.u5 = up(outChannels=512)
        self.s5 = skipLayer(outChannels=512)

        self.u4 = up(outChannels=512)
        self.s4 = skipLayer(outChannels=512)
        
        self.u3 = up(outChannels=256)
        self.s3 = skipLayer(outChannels=256)
        
        self.u2 = up(outChannels=128)
        self.s2 = skipLayer(outChannels=128)
        
        self.u1 = up(outChannels=64)
        self.s1 = skipLayer(outChannels=64)
        
        self.conv2 = tf.keras.layers.Conv2D(3, kernel_size=(1,1), strides=(1,1), padding=padding)
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.actv2 = tf.keras.layers.ReLU()

        self.s0 = skipLayer(outChannels=3)


    def call(self, input_layer, training="training"):

        x_in = tf.scalar_mul(255.0, input_layer)

        # Convert RGB to BGR
        blue, green, red = tf.split(x_in, 3, 3)
        bgr = tf.concat([blue - self.VGG_MEAN[0], green - self.VGG_MEAN[1], red - self.VGG_MEAN[2]], axis=3)

        # Encoder
        x, d1 = self.d1(bgr)
        x, d2 = self.d2(x)
        x, d3 = self.d3(x)
        x, d4 = self.d4(x)
        enc, d5 = self.d5(x)

        # Fully convolutional layers on top of VGG conv layers
        x = self.conv1(enc)
        x = self.norm1(x, training)
        x = self.actv1(x)

        # Decoder
        x = self.u5(x, training)
        x = self.s5(x, d5)

        x = self.u4(x, training)
        x = self.s4(x, d4)
        
        x = self.u3(x, training)
        x = self.s3(x, d3)

        x = self.u2(x, training)
        x = self.s2(x, d2)
        
        x = self.u1(x, training)
        x = self.s1(x, d1)

        x = self.conv2(x)
        x = self.norm2(x, training)
        x = self.actv2(x)

        # Final skip-connection
        out = self.s0(x, bgr)
        
        return tf.nn.relu(out)
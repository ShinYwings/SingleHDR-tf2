import logging
from tabnanny import check

from finetune_real_dataset import LEARNING_RATE
logging.basicConfig(level=logging.INFO)

import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

import utils
import tf_utils
# from random_tone_map import random_tone_map

import dequantization_net as deq
import linearization_net as lin
import hallucination_net as hal
import refinement_net as ref

import glob

AUTO = tf.data.AUTOTUNE

"""
BGR input but RGB conversion in dataset.py (due to tf.image.rgb_to_grayscale and other layers)
"""
# Hyper parameters
THRESHOLD = 0.12
IMSHAPE = (256,256,3)

HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

CURRENT_WORKINGDIR = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "testImg/HDR-Real")

DEQ_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/deq_pretrained_40k")
LIN_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/lin")
HAL_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/hal")
REF_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/ref")

if __name__=="__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    _deq  = deq.model()
    _lin = lin.model()
    _hal = hal.model()
    _ref = ref.model()
    
    """Model initialization"""
    outputDir = utils.createNewDir(CURRENT_WORKINGDIR, "HDR-real_output")
    
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")

    optimizer_deq = tf.keras.optimizers.Adam(LEARNING_RATE)
    optimizer_lin = tf.keras.optimizers.Adam(LEARNING_RATE)
    optimizer_hal = tf.keras.optimizers.Adam(LEARNING_RATE)
    optimizer_ref = tf.keras.optimizers.Adam(LEARNING_RATE)

    def checkpoint_restore(model, opt, pretrained, name):
        ckpt = tf.train.Checkpoint(
                                epoch = tf.Variable(0),
                                lin=model,
                            optimizer=opt,)
        ckpt_manager = tf.train.CheckpointManager(ckpt, pretrained, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f'Latest {name} checkpoint has restored!!')

        return ckpt, ckpt_manager

    ckpt_deq, ckpt_manager_deq = checkpoint_restore(_deq, optimizer_deq, DEQ_PRETRAINED_DIR, "DEQ")
    ckpt_lin, ckpt_manager_lin = checkpoint_restore(_lin, optimizer_lin, LIN_PRETRAINED_DIR, "LIN")
    ckpt_hal, ckpt_manager_hal = checkpoint_restore(_hal, optimizer_hal, HAL_PRETRAINED_DIR, "HAL")
    ckpt_ref, ckpt_manager_ref = checkpoint_restore(_ref, optimizer_ref, REF_PRETRAINED_DIR, "REF")

    """
    Check out the dataset that properly work
    """
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,20))
    # i = 0
    # for (ldr, hdr) in ds.take(15):
    #     print(tf.shape(hdr))
    #     ax = plt.subplot(2,15,i+1)
    #     plt.imshow(ldr[0])
    #     ax = plt.subplot(2,15,i+2)
    #     plt.imshow(hdr[0])
    #     plt.axis('off')
    #     i+=2
    # plt.show()
    
    with tf.device('/GPU:0'):
        
        @tf.function
        def inference(ldr):

            # Dequantization
            pred_deq = _deq(ldr, training= False)
            C_pred = tf.clip_by_value(pred_deq, 0, 1)

            # Linearization
            pred_invcrf = _lin(C_pred, training= False)
            B_pred = tf_utils.apply_rf(C_pred, pred_invcrf)

            # Hallucination
            alpha = tf.reduce_max(B_pred, axis=[3])
            alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + THRESHOLD) / THRESHOLD)
            alpha = tf.reshape(alpha, [-1, tf.shape(B_pred)[1], tf.shape(B_pred)[2], 1])
            alpha = tf.tile(alpha, [1, 1, 1, 3])

            hal_res = _hal(B_pred, training= False)
            A_pred = (B_pred) + alpha * hal_res

            # Refinement
            refinement_output = _ref(tf.concat([A_pred, B_pred, C_pred], -1), training=False)
            refinement_output = refinement_output / (1e-6 + tf.reduce_mean(refinement_output, axis=[1, 2, 3], keepdims=True)) * 0.5
            
            return refinement_output
    
    ckpts = [ckpt_deq, ckpt_lin, ckpt_hal, ckpt_ref]
    ckpt_managers = [ckpt_manager_deq, ckpt_manager_lin, ckpt_manager_hal, ckpt_manager_ref]

    print("시작")
    
    dir_list = os.listdir(DATASET_DIR)
    ldr_imgs = [glob.glob(os.path.join(os.path.join(DATASET_DIR, dir), 'input.jpg')) for dir in dir_list]
    ldr_imgs = sorted(ldr_imgs)

    import cv2

    for ldr_img_path in ldr_imgs:

        start = time.perf_counter()
        
        # input rgb
        ldr_img_path = ldr_img_path[0]

        ldr_img = cv2.imread(ldr_img_path)

        # import matplotlib.pyplot as plt
        # plt.imshow(ldr_img)
        # plt.show()

        ldr_val = np.flip(ldr_img, -1).astype(np.float32) / 255.0

        ORIGINAL_H = ldr_val.shape[0]
        ORIGINAL_W = ldr_val.shape[1]

        """resize to 64x"""
        if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
            RESIZED_H = int(np.ceil(float(ORIGINAL_H) / 64.0)) * 64
            RESIZED_W = int(np.ceil(float(ORIGINAL_W) / 64.0)) * 64
            ldr_val = cv2.resize(ldr_val, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC)
        
        padding = 32
        ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

        ldr_val = tf.convert_to_tensor(ldr_val, dtype=tf.float32)
        ldr_val = tf.expand_dims(ldr_val, axis=0)
        HDR_out_val = inference(ldr_val)

        HDR_out_val = np.flip(HDR_out_val[0], -1)
        HDR_out_val = HDR_out_val[padding:-padding, padding:-padding]
        if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
            HDR_out_val = cv2.resize(HDR_out_val, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)
            
        # cv2.imwrite(os.path.join(outputDir, os.path.split(ldr_img_path)[-1][:-3]+'hdr'), HDR_out_val.numpy()) # rgb output
        outputfile_path= os.path.split(os.path.split(ldr_img_path)[0])[-1]+'.hdr'
        cv2.imwrite(os.path.join(outputDir, outputfile_path), HDR_out_val.numpy()) # rgb output
        print(f"Spends time : {time.perf_counter() - start} seconds")

    print("끝")
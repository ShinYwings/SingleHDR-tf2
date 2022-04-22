import logging
from sys import stderr
logging.basicConfig(level=logging.INFO)

import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

import utils
from dataset import get_train_dataset, RandDatasetReader
import tf_utils
# from random_tone_map import random_tone_map

import dequantization_net as deq
import linearization_net as lin
import hallucination_net as hal

AUTO = tf.data.AUTOTUNE

# HDR_PREFIX = "/media/shin/2nd_m.2/singleHDR/SingleHDR_training_data/HDR-Synth"
HDR_PREFIX = "/home/cvnar2/Desktop/nvme/SingleHDR_training_data/HDR-Synth"

# Hyper parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 8

EPOCHS = 5000000
IMSHAPE = (32,128,3)
RENDERSHAPE = (64,64,3)

HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

CURRENT_WORKINGDIR = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "dataset/tfrecord")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TRAIN_DEQ = False
TRAIN_LIN = True
TRAIN_HAL = False

DEQ_PRETRAINED_DIR = None
LIN_PRETRAINED_DIR = None
HAL_PRETRAINED_DIR = None

def hdr_logCompression(x, validDR = 5000.):

    # disentangled way
    x = tf.math.multiply(validDR, x)
    numerator = tf.math.log(1.+ x)
    denominator = tf.math.log(1.+validDR)
    output = tf.math.divide(numerator, denominator) - 1.

    return output

def hdr_logDecompression(x, validDR = 5000.):

    x = x + 1.
    denominator = tf.math.log(1.+validDR)
    x = tf.math.multiply(x, denominator)
    x = tf.math.exp(x)
    output = tf.math.divide(x, validDR)
    
    return output

def _tone_mapping(module, hdr, crf, t):
    b, h, w, c, = tf_utils.get_tensor_shape(hdr)
    b, k, = tf_utils.get_tensor_shape(crf)
    b, = tf_utils.get_tensor_shape(t)
    
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
    gray = tf.image.rgb_to_grayscale(jpeg_img)
    over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
    over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
    over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
    under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
    under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
    under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
    extreme_cases = tf.logical_or(over_exposed, under_exposed)
    loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

    if module == "deq":
        return [ldr, jpeg_img_float, loss_mask]

    elif module == "lin":
        return [ldr, clipped_hdr_t, loss_mask]

    elif module == "hal":
        return

    else:
        exit(0)
    
def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'render': tf.io.FixedLenFeature([], tf.string),
        'azimuth' : tf.io.FixedLenFeature([], tf.float32),
        'elevation' : tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    hdr = tf.io.decode_raw(example['image'], np.float32)
    hdr = tf.reshape(hdr, IMSHAPE)

    # TODO correct to HDR-Real dataset

    return hdr

def configureDataset(dirpath, train= "train"):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecord"), shuffle=False)
    tfrecords_list.extend(a)

    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)

    # if train:
    #     ds = ds.shuffle(buffer_size = 10000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    # else:
    #     ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    # deq_ds, lin_ds = ds, ds
    
    # TODO DEBUG
    if train:
        ds  = ds.take(700).shuffle(buffer_size = 700).batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    else:
        ds  = ds.take(300).batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    return ds
        
if __name__=="__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    
    """Path for tf.summary.FileWriter and to store model checkpoints"""
    root_dir=os.getcwd()
    
    """Init Dataset"""
    # train_ds = configureDataset(TRAIN_DIR, train=True)
    # test_ds  = configureDataset(TEST_DIR, train=False)

    """CheckPoint Create"""
    checkpoint_path = utils.createNewDir(root_dir, "checkpoints")

    # TODO
    _deq  = deq.model()
    _lin = lin.model()
    # _hal = hal.model()

    """"Create Output Image Directory"""
    if(TRAIN_DEQ):
        train_summary_writer_deq, test_summary_writer_deq, logdir_deq = tf_utils.createDirectories(root_dir, name="deq", dir="tensorboard")
        print('tensorboard --logdir={}'.format(logdir_deq))
        # train_outImgDir_deq, test_outImgDir_deq = tf_utils.createDirectories(root_dir, name="deq", dir="outputImg")

        """Model initialization"""
        optimizer_deq, train_loss_deq, test_loss_deq = tf_utils.model_initialization("deq", LEARNING_RATE) 

        ckpt_deq, ckpt_manager_deq = tf_utils.checkpoint_initialization(
                                        model_name="deq",
                                        pretrained_dir=DEQ_PRETRAINED_DIR,
                                        checkpoint_path=checkpoint_path,
                                        model=_deq,
                                        optimizer=optimizer_deq)
    
    # TODO
    if(TRAIN_LIN):
        train_summary_writer_lin, test_summary_writer_lin, logdir_lin = tf_utils.createDirectories(root_dir, name="lin", dir="tensorboard")
        print('tensorboard --logdir={}'.format(logdir_lin))
        # train_outImgDir_lin, test_outImgDir_lin = tf_utils.createDirectories(root_dir, name="lin", dir="outputImg")
        
        """Model initialization"""
        optimizer_lin, train_loss_lin, test_loss_lin = tf_utils.model_initialization("lin", LEARNING_RATE)

        ckpt_lin, ckpt_manager_lin = tf_utils.checkpoint_initialization(
                                        model_name="lin",
                                        pretrained_dir=LIN_PRETRAINED_DIR,
                                        checkpoint_path=checkpoint_path,
                                        model=_lin,
                                        optimizer=optimizer_lin)
    # if(TRAIN_HAL):
    #     train_summary_writer_hal, test_summary_writer_hal, logdir_hal = tf_utils.createDirectories(root_dir, name="hal", dir="tensorboard")
    #     print('tensorboard --logdir={}'.format(logdir_hal))
    #    # train_outImgDir_hal, test_outImgDir_hal = tf_utils.createDirectories(root_dir, name="hal", dir="outputImg")

    #     """Model initialization"""
    #     optimizer_hal, train_loss_hal, test_loss_hal = tf_utils.model_initialization("hal", LEARNING_RATE)
    
    #     ckpt_hal, ckpt_manager_hal = tf_utils.checkpoint_initialization(
    #                                     model_name="hal",
    #                                     pretrained_dir=HAL_PRETRAINED_DIR,
    #                                     checkpoint_path=checkpoint_path,
    #                                     model=_hal,
    #                                     optimizer=optimizer_hal)

    """
    Check out the dataset that properly work
    """
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,20))
    # for i, (image, means) in enumerate(train_ds.take(25)):
    #     ax = plt.subplot(5,5,i+1)
    #     plt.imshow(image[i])
    #     plt.axis('off')
    # plt.show()
    
    with tf.device('/GPU:0'):

        _clip = lambda x: tf.clip_by_value(x, 0, 1)

        ##################
        # Dequantization #
        ##################
        @tf.function
        def deq_train_step(ds):
            ldr, jpeg_img_float, loss_mask = ds
            with tf.GradientTape() as deq_tape:
                pred = _deq(jpeg_img_float, training= True)
                pred = _clip(pred)
                loss = tf_utils.get_l2_loss_with_mask(pred, ldr)
                # loss = tf_utils.get_l2_loss(pred, ldr)
                mask_loss = tf.reduce_mean(tf.multiply(loss,loss_mask))
            
            # TODO reduce_mean or not?
            # gradients_deq = deq_tape.gradient(tf.reduce_mean(loss*loss_mask), _deq.trainable_variables)
            gradients_deq = deq_tape.gradient(mask_loss, _deq.trainable_variables)
            optimizer_deq.apply_gradients(zip(gradients_deq, _deq.trainable_variables))
            train_loss_deq(mask_loss)

            return [pred]

        @tf.function
        def deq_test_step(gt):
            
            pred = _deq(gt, training= False)
            l1_loss = tf.reduce_mean(tf.square(pred - gt))
            test_loss_deq(l1_loss)

        ##################
        # Linearization  #
        ##################
        @tf.function
        def lin_train_step(ds):
            ldr, clipped_hdr_t, loss_mask, invcrf = ds

            with tf.GradientTape() as lin_tape:
                pred_invcrf = _lin(ldr, training= True)
                pred_lin_ldr = tf_utils.apply_rf(ldr, pred_invcrf)
                crf_loss = tf.reduce_mean(tf.square(pred_invcrf - invcrf), axis=1, keepdims=True)
                loss = tf_utils.get_l2_loss_with_mask(pred_lin_ldr, clipped_hdr_t)
                mask_loss = tf.reduce_mean(tf.multiply(tf.add(loss, 0.1*crf_loss),loss_mask))
            
            gradients_lin = lin_tape.gradient(mask_loss, _lin.trainable_variables)
            optimizer_lin.apply_gradients(zip(gradients_lin, _lin.trainable_variables))
            train_loss_lin(mask_loss)

            return [pred_lin_ldr, tf.reduce_mean(crf_loss)]
        @tf.function
        def lin_test_step(gt):
            
            pred = _lin(gt, training= False)
            l1_loss = tf.reduce_mean(tf.square(pred - gt))
            test_loss_lin(l1_loss)

        ##################
        # Hallucination  #
        ##################

        # TODO
        # @tf.function
        # def hal_train_step(gt):

        #     with tf.GradientTape() as hal_tape:
        #         pred = _hal(gt, training= True)
        #         l1_loss = tf.reduce_mean(tf.abs(pred - gt))
            
        #     gradients_hal = hal_tape.gradient(l1_loss, _hal.trainable_variables)
        #     optimizer_hal.apply_gradients(zip(gradients_hal, _hal.trainable_variables))
        #     train_loss_hal(l1_loss)

        # @tf.function
        # def hal_test_step(gt):
            
        #     pred = _hal(gt, training= False)
        #     l1_loss = tf.reduce_mean(tf.abs(pred - gt))
        #     test_loss_hal(l1_loss)
    
    def train(module="module",
                train_step="train_step", test_step="test_step",
                train_loss="train_loss", test_loss="test_loss",
                # train_ds = "train_ds", test_ds = "test_ds", # TODO model verification
                train_summary_writer = "train_summary_writer",
                test_summary_writer = "test_summary_writer",
                ckpt = "ckpt",
                ckpt_manager = "ckpt_manager"):

        dataset_reader = RandDatasetReader(get_train_dataset(HDR_PREFIX), BATCH_SIZE)
        
        # print("hdr len : ", hdr.__len__() , "   hdr shape : ", np.shape(hdr))
        # print("crf len : ", crf.__len__() , "   crf shape : ", np.shape(crf))
        # print("t   len : ", t.__len__() , "   t shape : ", np.shape(t))
        
        if module == "deq":
            EPOCHS = 47000     # overfitted on around 52.4k iter
        
        # TODO model verification
        #########################################
        for epoch in range(EPOCHS): # ACTUALLY iteraion, NOT Epoch in this paper, 

            start = time.perf_counter()

            train_loss.reset_states()

            hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()

            preprocessed_dataset = tf.py_function(_tone_mapping, [module, hdr_val, crf_val, t_val], [tf.float32, tf.float32, tf.float32])
            
            if module == "lin":
                preprocessed_dataset.append(invcrf_val)
            
            pred = train_step(preprocessed_dataset)

            with train_summary_writer.as_default():

                ldr       = preprocessed_dataset[0]
                loss_mask = preprocessed_dataset[2]

                tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
                
                tf.summary.image('ldr', ldr, step=epoch+1)

                if module == "deq":
                    tf.summary.image('jpeg_img_float', preprocessed_dataset[1], step=epoch+1)
                    tf.summary.image('pred', pred[0], step=epoch+1)

                if module == "lin":
                    tf.summary.image('pred_lin_ldr', pred[0], step=epoch+1)
                    tf.summary.scalar('crf_loss', pred[1], step=epoch+1)
                    tf.summary.image('clipped_hdr_t', preprocessed_dataset[1], step=epoch+1)
                
                tf.summary.scalar('loss_mask 0', tf.squeeze(loss_mask[0]), step=epoch+1)
                tf.summary.scalar('loss_mask 1', tf.squeeze(loss_mask[1]), step=epoch+1)
                tf.summary.scalar('loss_mask 2', tf.squeeze(loss_mask[2]), step=epoch+1)

            print('IN {}, iteration: {}, Train Loss: {}'.format(module, epoch+1, train_loss.result()))
        
            print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))
            
            ckpt.epoch.assign_add(1)

            if ckpt.epoch == 1 or ckpt.epoch % 10000 == 0:
                save_path =  ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.epoch), save_path))
        #########################################

        # for epoch in range(EPOCHS):

        #     start = time.perf_counter()

        #     train_loss.reset_states()
        #     test_loss.reset_states()

        #     for step, hdrs in enumerate(tqdm(train_ds)):
                

        #         train_step(hdrs)

        #     with train_summary_writer.as_default():
        #         tf.summary.scalar('loss', train_loss.result(), step=epoch+1)

        #     for step, hdrs in enumerate(tqdm(test_ds)):
                
        #         test_step(hdrs)

        #     with test_summary_writer.as_default():
        #         tf.summary.scalar('loss', test_loss.result(), step=epoch+1)

        #     print('IN {}, Epoch: {}, Train Loss: {}, Test Loss: {}'.format(module, epoch+1, train_loss.result(), test_loss.result()))
        
        #     print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))
            
        #     ckpt.epoch.assign_add(1)

        #     if int(ckpt.epoch) % 10 == 0:
        #         save_path =  ckpt_manager.save()
        #         print("Saved checkpoint for step {}: {}".format(int(ckpt.epoch), save_path))

    print("시작")
    # isFirst = True

    if TRAIN_DEQ:
        train(module="deq",
                train_step=deq_train_step, test_step=deq_test_step, 
                train_loss=train_loss_deq, test_loss=test_loss_deq,
                # train_ds = train_ds, test_ds = test_ds, # TODO model verification
                train_summary_writer = train_summary_writer_deq,
                test_summary_writer = test_summary_writer_deq,
                ckpt = ckpt_deq,
                ckpt_manager = ckpt_manager_deq)
    
    # TODO
    if TRAIN_LIN:
        train(module="lin",
                train_step=lin_train_step, test_step=lin_test_step, 
                train_loss=train_loss_lin, test_loss=test_loss_lin,
                # train_ds = train_ds, test_ds = test_ds, # TODO model verification
                train_summary_writer = train_summary_writer_lin,
                test_summary_writer = test_summary_writer_lin,
                ckpt = ckpt_lin,
                ckpt_manager = ckpt_manager_lin)
    
    # if TRAIN_HAL:
    #     train(module="hal",
    #             train_step=hal_train_step, test_step=hal_test_step,  
    #             train_loss=train_loss_hal, test_loss=test_loss_hal,
    #             train_ds = train_ds, test_ds = test_ds,
    #             train_summary_writer = train_summary_writer_hal,
    #             test_summary_writer = test_summary_writer_hal,
    #             ckpt = ckpt_hal,
    #             ckpt_manager = ckpt_manager_hal)
    
    print("끝")


    # if(TRAIN_DEQ):

    #     for epoch in range(EPOCHS):

    #         start = time.perf_counter()

    #         train_loss_deq.reset_states()
    #         test_loss_deq.reset_states()
            
    #         for step, (hdrs) in enumerate(tqdm(train_ds)):
    #             deq_train_step(hdrs)

    #         with train_summary_writer_deq.as_default():
    #             tf.summary.scalar('loss', train_loss_deq.result(), step=epoch+1)
            
    #         for step, (hdrs) in enumerate(tqdm(test_ds)):
                
    #             outImg = deq_test_step(hdrs)

    #         with test_summary_writer_deq.as_default():
    #             tf.summary.scalar('loss', test_loss_deq.result(), step=epoch+1)

    #         print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss_deq.result(), test_loss_deq.result()))
        
    #         print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))

    #         ckpt_deq.epoch.assign_add(1)

    #         if int(ckpt_deq.epoch) % 5 == 0:
    #             save_path =  ckpt_manager_deq.save()
    #             print("Saved checkpoint for step {}: {}".format(int(ckpt_deq.epoch), save_path))
        
    #     if(TRAIN_LIN):
        
    #         for step, (hdrs) in enumerate(tqdm(train_ds)):
    #             lin_train_step(hdrs)

    #         with train_summary_writer_lin.as_default():
    #             tf.summary.scalar('loss', train_loss_lin.result(), step=epoch+1)
            
    #         for step, (hdrs) in enumerate(tqdm(test_ds)):
                
    #             outImg = lin_test_step(hdrs)

    #         with test_summary_writer_lin.as_default():
    #             tf.summary.scalar('loss', test_loss_lin.result(), step=epoch+1)

    #         print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss_lin.result(), test_loss_lin.result()))
        
    #         print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))

    #         ckpt_lin.epoch.assign_add(1)

    #         if int(ckpt_lin.epoch) % 5 == 0:
    #             save_path =  ckpt_manager_lin.save()
    #             print("Saved checkpoint for step {}: {}".format(int(ckpt_lin.epoch), save_path))
        
    #     if(TRAIN_HAL):
        
    #         for step, (hdrs) in enumerate(tqdm(train_ds)):
    #             hal_train_step(hdrs)

    #         with train_summary_writer_hal.as_default():
    #             tf.summary.scalar('loss', train_loss_hal.result(), step=epoch+1)
            
    #         for step, (hdrs) in enumerate(tqdm(test_ds)):
                
    #             outImg = hal_test_step(hdrs)

    #         with test_summary_writer_hal.as_default():
    #             tf.summary.scalar('loss', test_loss_hal.result(), step=epoch+1)

    #         print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch+1, train_loss_hal.result(), test_loss_hal.result()))
        
    #         print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch+1))

    #         ckpt_hal.epoch.assign_add(1)

    #         if int(ckpt_hal.epoch) % 5 == 0:
    #             save_path =  ckpt_manager_hal.save()
    #             print("Saved checkpoint for step {}: {}".format(int(ckpt_hal.epoch), save_path))



    ##############################################




    # if isFirst:
    #     isFirst = False
    #     groundtruth_dir = utils.createNewDir(test_outImgDir_deq, "groundTruth")
        
    #     for i in range(hdrs.get_shape()[0]):
    #         utils.writeHDR(hdrs[i].numpy(), "{}/{}_gt.{}".format(groundtruth_dir,i,HDR_EXTENSION), hdrs.get_shape()[1:3])

    # if (epoch+1) % 10 == 0:
    #     outHDR = tf_utils.hdr_logDecompression(outImg)
    #     epoch_dir = utils.createNewDir(test_outImgDir_deq, "{}Epoch".format(epoch+1))
    #     for i in range(outHDR.get_shape()[0]):
    #         utils.writeHDR(outHDR[i].numpy(), "{}/{}.{}".format(epoch_dir,i,HDR_EXTENSION), outHDR.get_shape()[1:3])

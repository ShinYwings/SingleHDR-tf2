import logging
logging.basicConfig(level=logging.INFO)

import os
import tensorflow as tf
import time

from dataset import get_train_dataset, RandDatasetReader
import tf_utils
from utils import str2bool

import dequantization_net as deq
import linearization_net as lin
import hallucination_net as hal

from vgg16 import Vgg16

# Hyper parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5000000 # "EPOCHS" means "iteraion", NOT literally "epoch" in this code. 
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr
CURRENT_WORKINGDIR = os.getcwd()
AUTO = tf.data.AUTOTUNE

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def _preprocessing(module, hdr, crf, t):
    b, h, w, c, = tf_utils.get_tensor_shape(hdr)
    
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
        return [_hdr_t, clipped_hdr_t, loss_mask]

    else:
        exit(0)

def run(args):

    # HDR_PREFIX = "/home/cvnar2/Desktop/nvme/singleHDR/SingleHDR_training_data/HDR-Synth"
    HDR_PREFIX = args.dir
    """
    BGR input but RGB conversion in dataset.py (due to tf.image.rgb_to_grayscale and other layers)
    """
    
    # TODO
    TRAIN_DEQ = str2bool(args.deq)
    TRAIN_LIN = str2bool(args.lin)
    TRAIN_HAL = str2bool(args.hal)

    # Absolute path
    DEQ_PRETRAINED_DIR = args.deq_w
    LIN_PRETRAINED_DIR = args.lin_w
    HAL_PRETRAINED_DIR = args.hal_w

    """Path for tf.summary.FileWriter and to store model checkpoints"""
    
    _deq  = deq.model()
    _lin = lin.model()
    _hal = hal.model()
    vgg = Vgg16(args.vgg_w)
    vgg2 = Vgg16(args.vgg_w)

    """"Create Output Image Directory"""
    if(TRAIN_DEQ):
        train_summary_writer_deq, _, logdir_deq = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="deq", dir="tensorboard")
        print('tensorboard --logdir={}'.format(logdir_deq))
        
        """Model initialization"""
        optimizer_deq, train_loss_deq, _ = tf_utils.model_initialization("deq", LEARNING_RATE) 

        ckpt_deq, ckpt_manager_deq = tf_utils.checkpoint_initialization(
                                        model_name="deq",
                                        pretrained_dirpath=DEQ_PRETRAINED_DIR,
                                        model=_deq,
                                        optimizer=optimizer_deq)
    
    if(TRAIN_LIN):
        train_summary_writer_lin, _, logdir_lin = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="lin", dir="tensorboard")
        print('tensorboard --logdir={}'.format(logdir_lin))
        
        """Model initialization"""
        optimizer_lin, train_loss_lin, _ = tf_utils.model_initialization("lin", LEARNING_RATE)

        ckpt_lin, ckpt_manager_lin = tf_utils.checkpoint_initialization(
                                        model_name="lin",
                                        pretrained_dirpath=LIN_PRETRAINED_DIR,
                                        model=_lin,
                                        optimizer=optimizer_lin)
    if(TRAIN_HAL):
        train_summary_writer_hal, _, logdir_hal = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="hal", dir="tensorboard")
        print('tensorboard --logdir={}'.format(logdir_hal))
        
        """Model initialization"""
        optimizer_hal, train_loss_hal, _ = tf_utils.model_initialization("hal", LEARNING_RATE)
    
        ckpt_hal, ckpt_manager_hal = tf_utils.checkpoint_initialization(
                                        model_name="hal",
                                        pretrained_dirpath=HAL_PRETRAINED_DIR,
                                        model=_hal,
                                        optimizer=optimizer_hal)

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
                deq_loss = tf.multiply(loss,loss_mask)
            
            gradients_deq = deq_tape.gradient(deq_loss, _deq.trainable_variables)
            optimizer_deq.apply_gradients(zip(gradients_deq, _deq.trainable_variables))
            train_loss_deq(deq_loss)

            return [pred]

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
                lin_loss = tf.multiply(tf.add(loss, 0.1*crf_loss),loss_mask)
            
            gradients_lin = lin_tape.gradient(lin_loss, _lin.trainable_variables)
            optimizer_lin.apply_gradients(zip(gradients_lin, _lin.trainable_variables))
            train_loss_lin(lin_loss)

            return [pred_lin_ldr, tf.reduce_mean(crf_loss)]

        ##################
        # Hallucination  #
        ##################
        @tf.function
        def hal_train_step(ds):

            hdr_t, clipped_hdr_t, loss_mask = ds # rgb

            # Equivalent to "get_final(network, x_in)"
            thr = 0.12
            alpha = tf.reduce_max(clipped_hdr_t, axis=[3])
            alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
            alpha = tf.reshape(alpha, [-1, tf.shape(clipped_hdr_t)[1], tf.shape(clipped_hdr_t)[2], 1])
            alpha = tf.tile(alpha, [1, 1, 1, 3])

            with tf.GradientTape() as hal_tape:
                bgr_pred = _hal(clipped_hdr_t, training= True)
                pred = tf_utils.bgr2rgb(bgr_pred)
                y_final = (clipped_hdr_t) + alpha * pred
            
                vgg_pool1, vgg_pool2, vgg_pool3 = vgg(tf.math.log(1.0+10.0*y_final)/tf.math.log(1.0+10.0))
                vgg2_pool1, vgg2_pool2, vgg2_pool3 = vgg2(tf.math.log(1.0+10.0*hdr_t)/tf.math.log(1.0+10.0))
                perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)), axis=[1, 2, 3], keepdims=True)
                perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)), axis=[1, 2, 3], keepdims=True)
                perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)), axis=[1, 2, 3], keepdims=True)

                y_final_gamma = tf.math.log(1.0+10.0*y_final)/tf.math.log(1.0+10.0)
                hdr_t_gamma = tf.math.log(1.0+10.0*hdr_t)/tf.math.log(1.0+10.0)

                loss = tf.reduce_mean(tf.abs(y_final_gamma - hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
                y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
                y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')

                # Image denoising on 2D image of an anisotropic version  (NOT histogram matching loss)
                # https://en.wikipedia.org/wiki/Total_variation_denoising
                tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
                tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
                tv_loss = tv_loss_x + tv_loss_y

                hal_loss = tf.multiply((loss + 0.001 * perceptual_loss + 0.1 * tv_loss),loss_mask)

            gradients_hal = hal_tape.gradient(hal_loss, _hal.trainable_variables)
            optimizer_hal.apply_gradients(zip(gradients_hal, _hal.trainable_variables))
            train_loss_hal(hal_loss)

            return [pred, y_final, alpha]
    
    def train(module="module",
                train_step="train_step",
                train_loss="train_loss",
                train_summary_writer = "train_summary_writer",
                ckpt = "ckpt",
                ckpt_manager = "ckpt_manager"):

        global EPOCHS
        
        dataset_reader = RandDatasetReader(get_train_dataset(HDR_PREFIX), BATCH_SIZE)
        
        # print("hdr len : ", hdr.__len__() , "   hdr shape : ", np.shape(hdr))
        # print("crf len : ", crf.__len__() , "   crf shape : ", np.shape(crf))
        # print("t   len : ", t.__len__() , "   t shape : ", np.shape(t))

        #########################################
        for epoch in range(EPOCHS): # "EPOCHS" means "iteraion", NOT literally "epoch" in this code.

            start = time.perf_counter()

            train_loss.reset_states()

            hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()

            preprocessed_dataset = tf.py_function(_preprocessing, [module, hdr_val, crf_val, t_val], [tf.float32, tf.float32, tf.float32])
            
            if module == "lin":
                preprocessed_dataset.append(invcrf_val)
            
            pred = train_step(preprocessed_dataset)

            print('IN {}, iteration: {}, Train Loss: {}'.format(module, epoch, train_loss.result()))
        
            print("Spends time : {} seconds in global Epoch {}".format(time.perf_counter() - start, int(ckpt.epoch)))
            
            ckpt.epoch.assign_add(1)

            if ckpt.epoch == 1 or ckpt.epoch % 1000 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
                    if module == "deq":
                        ldr            = preprocessed_dataset[0]
                        jpeg_img_float = preprocessed_dataset[1]
                        tf.summary.image('ldr', ldr, step=epoch+1)
                        tf.summary.image('jpeg_img_float', jpeg_img_float, step=epoch+1)
                        tf.summary.image('pred', pred[0], step=epoch+1)

                    if module == "lin":
                        ldr           = preprocessed_dataset[0]
                        clipped_hdr_t = preprocessed_dataset[1]
                        tf.summary.image('ldr', ldr, step=epoch+1)
                        tf.summary.image('pred_lin_ldr', pred[0], step=epoch+1)
                        tf.summary.scalar('crf_loss', pred[1], step=epoch+1)
                        tf.summary.image('clipped_hdr_t', clipped_hdr_t, step=epoch+1)
                    
                    if module == "hal":
                        _hdr_t        = preprocessed_dataset[0]
                        clipped_hdr_t = preprocessed_dataset[1]
                        tf.summary.image('hdr_t', _hdr_t, step=epoch+1)
                        tf.summary.image('y', pred[1], step=epoch+1)
                        tf.summary.image('clipped_hdr_t', clipped_hdr_t, step=epoch+1)
                        tf.summary.image('alpha', pred[2], step=epoch+1)
                        tf.summary.image('y_predict', pred[0], step=epoch+1)
                save_path =  ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.epoch), save_path))
    
    print("Start to train")
    if TRAIN_DEQ:
        train(module="deq",
                train_step=deq_train_step,  
                train_loss=train_loss_deq, 
                train_summary_writer = train_summary_writer_deq,
                ckpt = ckpt_deq,
                ckpt_manager = ckpt_manager_deq)
    
    if TRAIN_LIN:
        train(module="lin",
                train_step=lin_train_step,  
                train_loss=train_loss_lin, 
                train_summary_writer = train_summary_writer_lin,
                ckpt = ckpt_lin,
                ckpt_manager = ckpt_manager_lin)
    
    if TRAIN_HAL:
        train(module="hal",
                train_step=hal_train_step,   
                train_loss=train_loss_hal, 
                train_summary_writer = train_summary_writer_hal,
                ckpt = ckpt_hal,
                ckpt_manager = ckpt_manager_hal)
    print("End of training")
    
if __name__=="__main__":

    import argparse
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

    parser = argparse.ArgumentParser(description="train SingleHDR")
    parser.add_argument('--dir', type=str)
    parser.add_argument('--deq', type=str, default="False")
    parser.add_argument('--lin', type=str, default="False")
    parser.add_argument('--hal', type=str, default="False")
    parser.add_argument('--deq_w', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/deq"))
    parser.add_argument('--lin_w', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/lin"))
    parser.add_argument('--hal_w', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/hal"))
    parser.add_argument('--vgg_w', type=str, default=os.path.join(CURRENT_WORKINGDIR, 'vgg16.npy'))
    
    args = parser.parse_args
    run(args)
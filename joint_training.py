import logging
logging.basicConfig(level=logging.INFO)

import os
import tensorflow as tf
import time

from dataset import get_train_dataset, RandDatasetReader
import tf_utils

import dequantization_net as deq
import linearization_net as lin
import hallucination_net as hal

from vgg16 import Vgg16

AUTO = tf.data.AUTOTUNE

# Hyper parameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
EPOCHS = 5000000 # "EPOCHS" means "iteraion", NOT literally "epoch" in this code. 
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr
CURRENT_WORKINGDIR = os.getcwd()

def _preprocessing(hdr, crf, t):
    b, h, w, c, = tf_utils.get_tensor_shape(hdr)
    
    _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
    sigma_c = 0.005 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
    noise_s_map = sigma_s * _hdr_t
    noise_s = tf.random.normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * tf.random.normal(shape=tf.shape(_hdr_t), seed=1)
    temp_x = temp_x + noise_c
    _hdr_t = tf.nn.relu(temp_x)

    # Dynamic range clipping
    clipped_hdr_t = tf.clip_by_value(_hdr_t, 0, 1)

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

    return [ldr, jpeg_img_float, clipped_hdr_t, _hdr_t, loss_mask]

def run(args):

    # Absolute path
    DEQ_PRETRAINED_DIR = args.deq_ckpt
    LIN_PRETRAINED_DIR = args.lin_ckpt
    HAL_PRETRAINED_DIR = args.hal_ckpt

    _deq  = deq.model()
    _lin = lin.model()
    _hal = hal.model()
    vgg = Vgg16(args.vgg_ckpt)
    vgg2 = Vgg16(args.vgg_ckpt)

    """"Create Output Image Directory"""
    train_summary_writer_jnt, _, logdir_jnt = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="jnt", dir="tensorboard")
    print('tensorboard --logdir={}'.format(logdir_jnt))
    # train_outImgDir_jnt, _ = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="jnt", dir="outputImg")

    optimizer_jnt, train_loss_jnt, _ = tf_utils.model_initialization("jnt", LEARNING_RATE) 

    """Model initialization"""
    optimizer_deq, train_loss_deq, _ = tf_utils.model_initialization("deq", LEARNING_RATE) 

    ckpt_deq, ckpt_manager_deq = tf_utils.checkpoint_initialization(
                                    model_name="deq",
                                    pretrained_dirpath=DEQ_PRETRAINED_DIR,
                                    model=_deq,
                                    optimizer=optimizer_deq)
    
    # train_summary_writer_lin, _, logdir_lin = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="lin", dir="tensorboard")
    # print('tensorboard --logdir={}'.format(logdir_lin))
    # train_outImgDir_lin, _ = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="lin", dir="outputImg")
    
    """Model initialization"""
    optimizer_lin, train_loss_lin, _ = tf_utils.model_initialization("lin", LEARNING_RATE)
    train_crf_loss = tf.keras.metrics.Mean(name= 'train_crf_loss', dtype=tf.float32)
    ckpt_lin, ckpt_manager_lin = tf_utils.checkpoint_initialization(
                                    model_name="lin",
                                    pretrained_dirpath=LIN_PRETRAINED_DIR,
                                    model=_lin,
                                    optimizer=optimizer_lin)

    # train_summary_writer_hal, _, logdir_hal = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="hal", dir="tensorboard")
    # print('tensorboard --logdir={}'.format(logdir_hal))
    # train_outImgDir_hal, _ = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="hal", dir="outputImg")

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
    
    @tf.function
    def train_step(ds, invcrf):
        ldr, jpeg_img_float, clipped_hdr_t, hdr_t, loss_mask = ds

        thr = 0.12
        alpha = tf.reduce_max(clipped_hdr_t, axis=[3])
        alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
        alpha = tf.reshape(alpha, [-1, tf.shape(clipped_hdr_t)[1], tf.shape(clipped_hdr_t)[2], 1])
        alpha = tf.tile(alpha, [1, 1, 1, 3])
        
        with tf.GradientTape() as tape:
            
            # Dequantization
            pred_deq = _deq(jpeg_img_float, training= True)
            C_pred = tf.clip_by_value(pred_deq, 0, 1)
            l2loss_deq = tf_utils.get_l2_loss_with_mask(C_pred, ldr)
            loss_deq = tf.multiply(l2loss_deq,loss_mask)

            # Linearization
            pred_invcrf = _lin(ldr, training= True)
            B_pred = tf_utils.apply_rf(ldr, pred_invcrf)
            crf_loss = tf.reduce_mean(tf.square(pred_invcrf - invcrf), axis=1, keepdims=True)
            l2loss_lin = tf_utils.get_l2_loss_with_mask(B_pred, clipped_hdr_t)
            loss_lin = tf.multiply(10. * l2loss_lin + crf_loss, loss_mask)

            # Hallucination
            bgr_pred_hal = _hal(clipped_hdr_t, training= True)
            pred_hal = tf_utils.bgr2rgb(bgr_pred_hal)
            A_pred = (clipped_hdr_t) + alpha * pred_hal
            vgg_pool1, vgg_pool2, vgg_pool3 = vgg(tf.math.log(1.0+10.0*A_pred)/tf.math.log(1.0+10.0))
            vgg2_pool1, vgg2_pool2, vgg2_pool3 = vgg2(tf.math.log(1.0+10.0*hdr_t)/tf.math.log(1.0+10.0))

            perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)), axis=[1, 2, 3], keepdims=True)
            perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)), axis=[1, 2, 3], keepdims=True)
            perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)), axis=[1, 2, 3], keepdims=True)

            y_final_gamma = tf.math.log(1.0+10.0*A_pred) /tf.math.log(1.0+10.0)
            hdr_t_gamma   = tf.math.log(1.0+10.0*hdr_t)  /tf.math.log(1.0+10.0)

            l1loss_hal = tf.reduce_mean(tf.abs(y_final_gamma - hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
            y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
            y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
            tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
            tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
            tv_loss = tv_loss_x + tv_loss_y
            loss_hal   = tf.multiply((l1loss_hal + 0.001 * perceptual_loss + 0.1 * tv_loss), loss_mask)
            total_loss = loss_deq + loss_lin + loss_hal
        
        gradients = tape.gradient(total_loss, _deq.trainable_variables+_lin.trainable_variables+_hal.trainable_variables)
        optimizer_jnt.apply_gradients(zip(gradients, _deq.trainable_variables+_lin.trainable_variables+_hal.trainable_variables))
        
        train_loss_deq(loss_deq)
        train_loss_lin(loss_lin)
        train_crf_loss(crf_loss)
        train_loss_hal(loss_hal)
        train_loss_jnt(total_loss)

        return [C_pred, B_pred, A_pred, alpha]

    train_loss = [train_loss_deq, train_loss_lin, train_loss_hal, train_loss_jnt]
    ckpt = [ckpt_deq, ckpt_lin, ckpt_hal]
    ckpt_manager = [ckpt_manager_deq, ckpt_manager_lin, ckpt_manager_hal]

    def train(train_step="train_step"):
        
        dataset_reader = RandDatasetReader(get_train_dataset(args.dir), BATCH_SIZE)
        
        # print("hdr len : ", hdr.__len__() , "   hdr shape : ", np.shape(hdr))
        # print("crf len : ", crf.__len__() , "   crf shape : ", np.shape(crf))
        # print("t   len : ", t.__len__() , "   t shape : ", np.shape(t))

        #########################################
        for epoch in range(1, EPOCHS+1): # "EPOCHS" means "iteraion", NOT literally "epoch" in this code. 

            start = time.perf_counter()

            for tl in train_loss:
                tl.reset_states() 
            
            hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()

            preprocessed_dataset = tf.py_function(_preprocessing, [hdr_val, crf_val, t_val], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            # output : [ldr, jpeg_img_float, clipped_hdr_t, _hdr_t, loss_mask]
            
            pred = train_step(preprocessed_dataset, invcrf_val)

            with train_summary_writer_jnt.as_default():
                
                ldr            = preprocessed_dataset[0]
                jpeg_img_float = preprocessed_dataset[1]
                clipped_hdr_t  = preprocessed_dataset[2]
                _hdr_t         = preprocessed_dataset[3]
                loss_mask      = preprocessed_dataset[4]

                tf.summary.scalar('deq loss', train_loss_deq.result(), step=epoch)
                tf.summary.scalar('lin loss', train_loss_lin.result(), step=epoch)
                tf.summary.scalar('hal loss', train_loss_hal.result(), step=epoch)
                tf.summary.scalar('total loss', train_loss_jnt.result(), step=epoch)
                tf.summary.scalar('crf_loss', train_crf_loss.result(), step=epoch)

                tf.summary.scalar('loss_mask 0', tf.squeeze(loss_mask[0]), step=epoch)
                tf.summary.scalar('loss_mask 1', tf.squeeze(loss_mask[1]), step=epoch)
                tf.summary.scalar('loss_mask 2', tf.squeeze(loss_mask[2]), step=epoch)
                
                tf.summary.image('ldr', ldr, step=epoch)

                tf.summary.image('jpeg_img_float', jpeg_img_float, step=epoch)
                tf.summary.image('C_pred', pred[0], step=epoch)

                tf.summary.image('clipped_hdr_t', clipped_hdr_t, step=epoch)
                tf.summary.image('B_pred', pred[1], step=epoch)

                tf.summary.image('hdr_t', _hdr_t, step=epoch)
                tf.summary.image('alpha', pred[3], step=epoch)
                tf.summary.image('A_pred', pred[2], step=epoch)
            
            print(f'[Joint training], iteration: {epoch}, total Loss: {train_loss_jnt.result()}\n \
                    deq Loss: {train_loss_deq.result()} lin Loss: {train_loss_lin.result()}, hal Loss: {train_loss_hal.result()}')
            print("Spends time : {} seconds in Epoch number {}".format(time.perf_counter() - start,epoch))
            
            for c in ckpt:
                c.epoch.assign_add(1)

            if epoch == 1 or epoch % 1000 == 0:
                for cm in ckpt_manager:
                    save_path =  cm.save()
                    print(f"Saved checkpoint for step {epoch}: {save_path}")
        
    print("Start to train")
    
    train( train_step=train_step )
    
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

    parser = argparse.ArgumentParser(description="Joint training of SingleHDR")
    parser.add_argument('--dir', type=str)
    parser.add_argument('--deq_ckpt', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/deq"))
    parser.add_argument('--lin_ckpt', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/lin"))
    parser.add_argument('--hal_ckpt', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/hal"))
    parser.add_argument('--vgg_ckpt', type=str, default=os.path.join(CURRENT_WORKINGDIR, 'vgg16.npy'))
    
    args = parser.parse_args
    run(args)
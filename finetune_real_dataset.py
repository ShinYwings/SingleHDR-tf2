import logging
logging.basicConfig(level=logging.INFO)

import os
import tensorflow as tf
import time

import utils
import tf_utils

import dequantization_net as deq
import linearization_net as lin
import hallucination_net as hal
import refinement_net as ref

AUTO = tf.data.AUTOTUNE

"""
BGR input but RGB conversion in dataset.py (due to tf.image.rgb_to_grayscale and other layers)
"""
# Hyper parameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
THRESHOLD = 0.12

EPOCHS = 1000
IMSHAPE = (256,256,3)

HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

CURRENT_WORKINGDIR = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "tf_records/256_64_b32_tfrecords")

DEQ_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/deq")
LIN_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/lin")
HAL_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/hal")
REF_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/ref")
    
def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'ref_HDR': tf.io.FixedLenFeature([], tf.string),
        'ref_LDR': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    ref_HDR = tf.io.decode_raw(example['ref_HDR'], tf.float32)
    ref_LDR = tf.io.decode_raw(example['ref_LDR'], tf.float32)
    ref_HDR = tf.reshape(ref_HDR, IMSHAPE)
    ref_LDR = tf.reshape(ref_LDR, IMSHAPE)

    ref_HDR = ref_HDR / (1e-6 + tf.reduce_mean(ref_HDR)) * 0.5
    ref_LDR = ref_LDR / 255.0

    distortions = tf.random.uniform([2], 0, 1.0, dtype=tf.float32)

    # flip horizontally
    ref_HDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_HDR), lambda: ref_HDR)
    ref_LDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_LDR), lambda: ref_LDR)

    # rotate
    k = tf.cast(distortions[1] * 4 + 0.5, tf.int32)
    ref_HDR = tf.image.rot90(ref_HDR, k)
    ref_LDR = tf.image.rot90(ref_LDR, k)

    return ref_LDR, ref_HDR

def configureDataset(dirpath):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecords"), shuffle=False)
    tfrecords_list.extend(a)
    
    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)

    ds  = ds.shuffle(buffer_size = len(tfrecords_list)).batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    return ds
        
if __name__=="__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)
    
    ds  = configureDataset(DATASET_DIR)

    """CheckPoint Create"""
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")

    _deq  = deq.model()
    _lin = lin.model()
    _hal = hal.model()
    _ref = ref.model()

    """"Create Output Image Directory"""
    optimizer_deq, _, _ = tf_utils.model_initialization("deq", LEARNING_RATE) 

    ckpt_deq, ckpt_manager_deq = tf_utils.checkpoint_initialization(
                                    model_name="deq",
                                    pretrained_dir=DEQ_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_deq,
                                    optimizer=optimizer_deq)
    
      
    optimizer_lin, _, _ = tf_utils.model_initialization("lin", LEARNING_RATE)

    ckpt_lin, ckpt_manager_lin = tf_utils.checkpoint_initialization(
                                    model_name="lin",
                                    pretrained_dir=LIN_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_lin,
                                    optimizer=optimizer_lin)
    
        
    optimizer_hal, _, _ = tf_utils.model_initialization("hal", LEARNING_RATE)

    ckpt_hal, ckpt_manager_hal = tf_utils.checkpoint_initialization(
                                    model_name="hal",
                                    pretrained_dir=HAL_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_hal,
                                    optimizer=optimizer_hal)
    
    train_summary_writer_ref, test_summary_writer_ref, logdir_ref = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="ref", dir="tensorboard")
    print(f'tensorboard --logdir={logdir_ref}')

    """Model initialization"""
    optimizer_ref, train_loss_ref, test_loss_ref = tf_utils.model_initialization("ref", LEARNING_RATE) 

    ckpt_ref, ckpt_manager_ref = tf_utils.checkpoint_initialization(
                                    model_name="ref",
                                    pretrained_dir=REF_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_ref,
                                    optimizer=optimizer_ref)

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
    
    @tf.function
    def train_step(ldr, hdr):
        
        denominator = 1 / tf.math.log(1.0 + 10.0)
        
        with tf.GradientTape() as tape:

            # Dequantization
            pred_deq = _deq(ldr, training= True)
            C_pred = tf.clip_by_value(pred_deq, 0, 1)

            # Linearization
            pred_invcrf = _lin(C_pred, training= True)
            B_pred = tf_utils.apply_rf(C_pred, pred_invcrf)

            # Hallucination
            alpha = tf.reduce_max(B_pred, axis=[3])
            alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + THRESHOLD) / THRESHOLD)
            alpha = tf.reshape(alpha, [-1, tf.shape(B_pred)[1], tf.shape(B_pred)[2], 1])
            alpha = tf.tile(alpha, [1, 1, 1, 3])

            bgr_B_pred = tf_utils.rgb2bgr(B_pred)

            hal_res = _hal(bgr_B_pred, training= True)
            A_pred = (bgr_B_pred) + alpha * hal_res

            A_pred = tf_utils.bgr2rgb(A_pred)

            hdr_gamma = tf.math.log(1.0 + 10.0 * hdr) * denominator

            # Refinement
            refinement_output = _ref(tf.concat([A_pred, B_pred, C_pred], -1), training=True)
            refinement_output = refinement_output / (1e-6 + tf.reduce_mean(refinement_output, axis=[1, 2, 3], keepdims=True)) * 0.5
            refinement_output_gamma = tf.math.log(1.0 + 10.0 * refinement_output) * denominator
            loss = tf.abs(refinement_output_gamma - hdr_gamma)
        
        gradients = tape.gradient(loss, _deq.trainable_variables+_lin.trainable_variables+_hal.trainable_variables+_ref.trainable_variables)
        optimizer_ref.apply_gradients(zip(gradients, _deq.trainable_variables+_lin.trainable_variables+_hal.trainable_variables+_ref.trainable_variables))
        
        train_loss_ref(loss)

        # return [A_pred_loss, C_pred, B_pred, A_pred, refinement_output]
        return [C_pred, B_pred, A_pred, refinement_output]

    @tf.function
    def test_step(gt):
        # NO USED, NO TYPED
        pred = _deq(gt, training= False)
        l1_loss = tf.square(pred - gt)
        test_loss_ref(l1_loss)
    
    ckpts = [ckpt_deq, ckpt_lin, ckpt_hal, ckpt_ref]
    ckpt_managers = [ckpt_manager_deq, ckpt_manager_lin, ckpt_manager_hal, ckpt_manager_ref]

    print("시작")
    
    for epoch in range(1, EPOCHS+1):

        start = time.perf_counter()

        train_loss_ref.reset_states()

        for (ldr, hdr) in tqdm(ds):
            
            pred = train_step(ldr, hdr)

        with train_summary_writer_ref.as_default():
            
            C_pred, B_pred, A_pred, refinement_output = pred

            tf.summary.scalar('loss', train_loss_ref.result(), step=epoch)
            # tf.summary.scalar('A_pred_loss', tf.reduce_mean(A_pred_loss), step=epoch+1)
            tf.summary.image('hdr', hdr, step=epoch)
            tf.summary.image('ldr', ldr, step=epoch)
            tf.summary.image('C_pred', C_pred, step=epoch)
            tf.summary.image('B_pred', B_pred, step=epoch)
            tf.summary.image('A_pred', A_pred, step=epoch)
            tf.summary.image('refinement_output', refinement_output, step=epoch)
            
            tf.summary.histogram('hdr_histo', hdr, step=epoch)
            tf.summary.histogram('refinement_output_histo', refinement_output, step=epoch)

        print(f'IN ref, epoch: {epoch}, Train Loss: {train_loss_ref.result()}')

        print(f"Spends time : {time.perf_counter() - start} seconds in Epoch number {epoch}")
        
        for ckpt in ckpts:
            ckpt.epoch.assign_add(1)
        
        for ckpt_manager in ckpt_managers:
            save_path =  ckpt_manager.save()
            print(f"Saved checkpoint for step {epoch}: {save_path}")

    print("끝")
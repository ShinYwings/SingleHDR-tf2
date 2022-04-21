import tensorflow as tf
import utils

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
    
def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    def _select_one(aa, bb):
        if type(aa) is int:
            return aa
        else:
            return bb
    return [_select_one(aa, bb) for aa, bb in zip(a, b)]


def pass_net_nx(func, in_img, n):
    b, h, w, c, = get_tensor_shape(in_img)
    def _get_nx(x):
        s, r = x//n, x%n
        s = tf.cond(
            tf.equal(r, 0),
            lambda: s,
            lambda: s + 1,
        )
        return n*s
    nx_h = _get_nx(h)
    nx_w = _get_nx(w)
    def _get_rl_rr(x, nx):
        r = nx - x
        rl = r//2
        rr = r - rl
        return rl, rr
    in_img = tf.pad(in_img, [[0, 0], _get_rl_rr(h, nx_h), _get_rl_rr(w, nx_w), [0, 0]], mode='symmetric')
    in_img = tf.reshape(in_img, [b, nx_h, nx_w, c])
    out_img = func(in_img)
    out_img = tf.image.resize_with_crop_or_pad(out_img, h, w)
    return out_img


def sample_1d(
    img,   # [b, h, c]
    y_idx, # [b, n], 0 <= pos < h, dtpye=int32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y_idx)
    
    b_idx = tf.range(b, dtype=tf.int32) # [b]
    b_idx = tf.expand_dims(b_idx, -1)   # [b, 1]
    b_idx = tf.tile(b_idx, [1, n])      # [b, n]
    
    y_idx = tf.clip_by_value(y_idx, 0, h - 1) # [b, n]
    a_idx = tf.stack([b_idx, y_idx], axis=-1) # [b, n, 2]
    
    return tf.gather_nd(img, a_idx)

def interp_1d(
    img, # [b, h, c]
    y,   # [b, n], 0 <= pos < h, dtype=float32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y)
    
    y_0 = tf.floor(y) # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(
        img,
        tf.cast(y_x, tf.int32)
    )
    y_0_val = _sample_func(y_0) # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y # [b, n]
    w_1 = y - y_0
    
    w_0 = tf.expand_dims(w_0, -1) # [b, n, 1]
    w_1 = tf.expand_dims(w_1, -1)
    
    return w_0*y_0_val + w_1*y_1_val


def apply_rf(
    x,  # [b, s...]
    rf, # [b, k]
):
    b, *s, = get_tensor_shape(x)
    b, k,  = get_tensor_shape(rf)
    x = interp_1d(
        tf.expand_dims(rf, -1),                              # [b, k, 1] 
        tf.cast((k - 1), tf.float32)*tf.reshape(x, [b, -1]), # [b, ?] 
    ) # [b, ?, 1]
    return tf.reshape(x, [b] + s)

def get_l2_loss(a, b):
    return tf.reduce_mean(tf.square(a - b))

def get_l2_loss_with_mask(a, b):
    return tf.reduce_mean(tf.square(a - b), axis=[1, 2, 3], keepdims=True)

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

def createDirectories(path, name="name", dir="dir"):
    
    path = utils.createNewDir(path, dir)
    root_logdir = utils.createNewDir(path, name)
    logdir = utils.createNewDir(root_logdir)

    if dir=="tensorboard":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=False)
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        test_summary_writer = tf.summary.create_file_writer(test_logdir)
        return train_summary_writer, test_summary_writer, logdir

    if dir=="outputImg":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=True)
        return train_logdir, test_logdir

def checkpoint_initialization(model_name : str,
                                pretrained_dir : str,
                                checkpoint_path : str,
                                model="model",
                                optimizer="optimizer",
                                ):
    if pretrained_dir is None:
        checkpoint_path = utils.createNewDir(checkpoint_path, model_name)
    else: checkpoint_path = pretrained_dir
    
    ckpt = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            lin=model,
                           optimizer=optimizer,)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    #  if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest {} checkpoint has restored!!'.format(model_name))

    return ckpt, ckpt_manager

    #######################
    # Dequantization
    #######################
    
    # if DEQ_PRETRAINED_DIR is None:
    #     checkpoint_path_deq = utils.createNewDir(checkpoint_path, "deq")
    # else: checkpoint_path_deq = DEQ_PRETRAINED_DIR
    
    # ckpt_deq = tf.train.Checkpoint(
    #                         epoch = tf.Variable(0),
    #                         deq=_deq,
    #                        optimizer=optimizer_deq,)

    # ckpt_manager_deq = tf.train.CheckpointManager(ckpt_deq, checkpoint_path_deq, max_to_keep=5)

    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager_deq.latest_checkpoint:
    #     ckpt_deq.restore(ckpt_manager_deq.latest_checkpoint)
    #     print('Latest deq checkpoint has restored!!')

    ##################
    # Linearization
    ##################
    # if LIN_PRETRAINED_DIR is None:
    #     checkpoint_path_lin = utils.createNewDir(checkpoint_path, "lin")
    # else: checkpoint_path_lin = LIN_PRETRAINED_DIR
    
    # ckpt_lin = tf.train.Checkpoint(
    #                         epoch = tf.Variable(0),
    #                         lin=_lin,
    #                        optimizer=optimizer_lin,)

    # ckpt_manager_lin = tf.train.CheckpointManager(ckpt_lin, checkpoint_path_lin, max_to_keep=5)

    # #  if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager_lin.latest_checkpoint:
    #     ckpt_lin.restore(ckpt_manager_lin.latest_checkpoint)
    #     print('Latest lin checkpoint has restored!!')

    ####################
    #  Hallucination
    ####################

    # if HAL_PRETRAINED_DIR is None:
    #     checkpoint_path_hal = utils.createNewDir(checkpoint_path, "hal")
    # else: checkpoint_path_hal = HAL_PRETRAINED_DIR
    
    # ckpt_hal = tf.train.Checkpoint(
    #                         epoch = tf.Variable(0),
    #                         hal=_hal,
    #                        optimizer=optimizer_hal,)

    # ckpt_manager_hal = tf.train.CheckpointManager(ckpt_hal, checkpoint_path_hal, max_to_keep=5)

    # #  if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager_hal.latest_checkpoint:
    #     ckpt_hal.restore(ckpt_manager_hal.latest_checkpoint)
    #     print('Latest hal checkpoint has restored!!')

def model_initialization(model_name : str, lr = "lr"):
    optimizer = tf.keras.optimizers.Adam(lr)
    train_loss = tf.keras.metrics.Mean(name= 'train_loss_{}'.format(model_name), dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss_{}'.format(model_name), dtype=tf.float32)

    return optimizer, train_loss, test_loss
    
    # deq
    # optimizer_deq = tf.keras.optimizers.Adam(LEARNING_RATE)
    # train_loss_deq = tf.keras.metrics.Mean(name= 'train_loss_deq', dtype=tf.float32)
    # test_loss_deq = tf.keras.metrics.Mean(name='test_loss_deq', dtype=tf.float32)
    
    # lin
    # optimizer_lin = tf.keras.optimizers.Adam(LEARNING_RATE)
    # train_loss_lin = tf.keras.metrics.Mean(name= 'train_loss_lin', dtype=tf.float32)
    # test_loss_lin = tf.keras.metrics.Mean(name='test_loss_lin', dtype=tf.float32)

    # hal
    # optimizer_hal = tf.keras.optimizers.Adam(LEARNING_RATE)
    # train_loss_hal = tf.keras.metrics.Mean(name= 'train_loss_hal', dtype=tf.float32)
    # test_loss_hal = tf.keras.metrics.Mean(name='test_loss_hal', dtype=tf.float32)

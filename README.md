# Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline (CVPR 2020) \[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Single-Image_HDR_Reconstruction_by_Learning_to_Reverse_the_Camera_Pipeline_CVPR_2020_paper.pdf)\]

<img src="figure/arch.png" width="100%" height="100%">

Reconstructed "Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline" (CVPR 2020) from [the author's official code](https://github.com/alex04072000/SingleHDR) using Tensorflow2.

# Note

1. Most pretrained weight provided from [the official git repository](https://github.com/alex04072000/SingleHDR) can be used in this code.
    > Not support loading "crf-net_v2.npy" in linearization_net.

2. Modified `spatial-aware soft-histogram layer` that does not match the paper's description in Linearization-Net.
    - The histogram bin $b$ in the original code starts at index 0, but it should be started at index 1. ($\because b \in \lbrace 1, \cdots, B \rbrace $)
    - The distance equation in the original code is  $$d = |I(i,j,c)- \frac {b} {B}|$$
      I modified the distance equation in my code as described in the paper (see below figure).
      <img src="figure/lin.png" width="70%" height="70%">

    - linearization_net.py (original code)

      ```diff
      def histogram_layer(img, max_bin):
          # histogram branch
          tmp_list = []

      -   for i in range(max_bin + 1):
      -     histo = tf.nn.relu(1 - tf.abs(img - i / float(max_bin)) * float(max_bin))
            tmp_list.append(histo)
        
          histogram_tensor = tf.concat(tmp_list, -1)
          return histogram_tensor
          # histogram_tensor = tf.layers.average_pooling2d(histogram_tensor, 16, 1, 'same')
      ```

    - linearization_net.py (my code)

      ```diff
      def histogram_layer(self, img, max_bin):
          # histogram branch
          tmp_list = []   
      +   _threshold = 1. / max_bin
      +   condition = lambda x: tf.less(x, _threshold)
      +   max_bin_square = 2.*max_bin

      +   for i in range(1, max_bin + 1):
      +     distance = tf.abs(img - tf.divide((2.*i - 1.), max_bin_square))
      +     histo = tf.where(condition(distance) , tf.subtract(1., tf.multiply(distance, max_bin)), 0)
            tmp_list.append(histo)

          histogram_tensor = tf.concat(tmp_list, -1)
          return histogram_tensor
          # histogram_tensor = tf.layers.average_pooling2d(histogram_tensor, 16, 1, 'same')
      ```

3. To prevent potentially errors, I added a function that converts channel format in the training process of the Hallucination-Net.

    - Hallucination_net in the original code returns an output image in BGR format. However, Vgg16 used for "perceptual loss" in the training process takes an input image in RGB format. So, in my code, I applied the `tf_utils.bgr2rgb` function to the output image of hallucination_net.
    - train_hallucination_net.py (original code)

      ```diff
      299   with tf.variable_scope("Hallucination_Net"):
      230       net, vgg16_conv_layers = hallucination_net.model(clipped_hdr_t, ARGS.batch_size, True)
      231       y_predict = tf.nn.relu(net.outputs)
      -                                                           
      232       y_final = (clipped_hdr_t) + alpha * y_predict # residual

      ...

      242   vgg = Vgg16('vgg16.npy')
      243   vgg.build(tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0))
      244   vgg2 = Vgg16('vgg16.npy')
      245   vgg2.build(tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0))
      246   perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
      247   perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
      248   perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)
      ```

    - train.py (my code)

      ```diff
      215   with tf.GradientTape() as hal_tape:
      216       bgr_pred = _hal(clipped_hdr_t, training= True)
      +         pred = tf_utils.bgr2rgb(bgr_pred)
      218       y_final = (clipped_hdr_t) + alpha * pred
      219        
      220   vgg_pool1, vgg_pool2, vgg_pool3 = vgg(tf.math.log(1.0+10.0*y_final)/tf.math.log(1.0+10.0))
      221   vgg2_pool1, vgg2_pool2, vgg2_pool3 = vgg2(tf.math.log(1.0+10.0*hdr_t)/tf.math.log(1.0+10.0))
      222   perceptual_loss = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)), axis=[1, 2, 3], keepdims=True)
      223   perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)), axis=[1, 2, 3], keepdims=True)
      224   perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)), axis=[1, 2, 3], keepdims=True)
      ```

4. Created "joint_training" that uses Synth-HDR-dataset in order to converge three networks (dequantization-net, linearlization-net, hallucination-net) before fine-tuning.

    ```
      joint_training.py
    ```

# Requirements

- tensorflow >= 2.4
- pickle
- scipy
- opencv
- tqdm

# Usage

# Result

# Training

1. Download the pre-trained weights of [vgg16](https://drive.google.com/file/d/1sNrwJJxCTIJ1G_7kgXkITCXZvmrJvSe5/view?usp=sharing) and [vgg16_places365_weights](https://drive.google.com/file/d/1_onEcNKpMR1R-AzWRrY9FQtHf7NGKTX5/view?usp=sharing)
2. Download the training data of [HDR-Synth and HDR-Real](https://drive.google.com/file/d/1muy49Pd0c7ZkxyxoxV7vIRvPv6kJdPR2/view?usp=sharing)

## Train the Dequantization-Net using HDR-Synth dataset

  ```
  python train.py --deq "True" --deq_ckpt "output/deq/ckpt/path" --dir "hdr/synth/training/data/path"
  ```

## Train the Linearization-Net using HDR-Synth dataset

  ```
  python train.py --lin "True" --lin_ckpt "output/lin/ckpt/path" --dir "hdr/synth/training/data/path"
  ```

## Train the Hallucination-Net using HDR-Synth dataset

  ```
  python train.py --hal "True" --hal_ckpt "output/hal/ckpt/path" --dir "hdr/synth/training/data/path"
  ```

## Joint training of the entire pipeline using HDR-Synth dataset

  ```
  python joint_training.py --deq_ckpt "pretrained/deq/ckpt" --lin_ckpt "pretrained/lin/ckpt" --hal_ckpt "pretrained/hal/ckpt" --vgg_ckpt "pretrained/vgg/ckpt" --dir "hdr/synth/training/data/path" 
  ```

## Fine-tuning the entire pipeline with Refinement-Net using HDR-real dataset

  1. Convert the real HDR-jpg paired data into tfrecords for training.

      ```
      python convert_to_tf_record.py --dir "hdr/real/training/data/path"
      ```

  2. Fine-tuning the entire pipeline with Refinement-Net

      ```
      python finetune_real_dataset.py --deq_ckpt "pretrained/deq/ckpt" --lin_ckpt "pretrained/lin/ckpt" --deq_ckpt "pretrained/hal/ckpt" --ref_ckpt "pretrained/ref/ckpt"  
      ```

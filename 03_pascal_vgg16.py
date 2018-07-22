from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
from os import listdir
import os.path as osp
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
# from skimage import io
import pdb
from eval import compute_map
# import models
import pickle
from tensorflow.core.framework import summary_pb2

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

trainval_data_dir = 'VOCdevkit_trainVal/VOC2007'
test_data_dir = 'VOCdevkit_test/VOC2007'
size = 224
full=0

reader = tf.train.NewCheckpointReader('vgg_16.ckpt')

def conv2d(inputs,filters, kernel_size,padding,activation,name):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=[1,1],
        padding=padding,
        activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name=name
        )

def dense(inputs,units,activation, name):
    return tf.layers.dense(
        inputs=inputs, 
        units=units,
        activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        name=name
        )


def cnn_model_fn(features, labels, mode, num_classes=20):
   
    # pdb.set_trace()

    input_layer = tf.reshape(features["x"], [-1, size, size, 3])
    resize = lambda x:tf.image.resize_images(x, size=[280,280])
    rand_crop = lambda x:tf.random_crop(x,size=[size,size,3])
    cen_crop = lambda x: tf.image.central_crop(x, central_fraction=0.8)
    rand_flip = lambda x:tf.image.random_flip_left_right(x)


    if mode == tf.estimator.ModeKeys.TRAIN:
        input_aug = tf.map_fn(fn=resize, elems=input_layer, name='resize_train')
        input_aug = tf.map_fn(fn=rand_crop, elems=input_aug, name='random_crop')
        input_aug = tf.map_fn(fn=rand_flip, elems=input_aug, name='random_flip')

    elif mode == tf.estimator.ModeKeys.PREDICT:
        input_aug = tf.map_fn(fn=resize, elems=input_layer, name='resize_test')
        input_aug = tf.map_fn(fn=cen_crop, elems=input_aug, name='center_crop')

    else:
        input_aug = input_layer
    
    # pdb.set_trace()
    # Convolutional Layer #1
    with tf.variable_scope('conv1'):
        conv1 = conv2d(input_aug,64,[3, 3],"same",tf.nn.relu,'conv1_1')
        conv2 = conv2d(conv1,64,[3, 3],"same",tf.nn.relu,'conv1_2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    with tf.variable_scope('conv2'):
        conv3 = conv2d(pool1,128,[3, 3],"same",tf.nn.relu,'conv2_1')
        conv4 = conv2d(conv3,128,[3, 3],"same",tf.nn.relu,'conv2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv3'):
        conv5 = conv2d(pool2,256,[3, 3],"same",tf.nn.relu,'conv3_1')
        conv6 = conv2d(conv5,256,[3, 3],"same",tf.nn.relu,'conv3_2')
        conv7 = conv2d(conv6,256,[3, 3],"same",tf.nn.relu,'conv3_3')
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv4'):
        conv8 = conv2d(pool3,512,[3, 3],"same",tf.nn.relu,'conv4_1')
        conv9 = conv2d(conv8,512,[3, 3],"same",tf.nn.relu,'conv4_2')
        conv10 = conv2d(conv9,512,[3, 3],"same",tf.nn.relu,'conv4_3')
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv5'):
        conv11 = conv2d(pool4,512,[3, 3],"same",tf.nn.relu,'conv5_1')
        conv12 = conv2d(conv11,512,[3, 3],"same",tf.nn.relu,'conv5_2')
        conv13 = conv2d(conv12,512,[3, 3],"same",tf.nn.relu,'conv5_3')
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    dense1 = dense(pool5_flat, 4096,tf.nn.relu,'fc6')
    
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = dense(dropout1, 4096,tf.nn.relu,'fc7')
    
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20,name='fc8')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
    multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        global_step=tf.train.get_global_step()
        decayed_learning_rate = tf.train.exponential_decay(
            learning_rate=0.001, 
            global_step=global_step,
            decay_steps=10000,
            decay_rate=0.5)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=decayed_learning_rate, 
            momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)
        grads_and_vars=optimizer.compute_gradients(loss)
        for g, v in grads_and_vars:
            if g is not None:
                tf.summary.histogram(v.name[:-2],v)
                tf.summary.histogram(v.name[:-2]+'_grad', g)
        tf.summary.image('my_image', input_layer, max_outputs=10)
        tf.summary.scalar('train_loss', loss)
        tf.summary.scalar('learning_rate', decayed_learning_rate)
        # summary_hook = tf.train.SummarySaverHook(
        #     save_steps=400,
        #     output_dir='mnistParams',
        #     summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss, 
            train_op=train_op)
            # training_hooks=[summary_hook])

def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
    """
    filename = osp.join(data_dir,'ImageSets/Main/'+split+".txt")
    with open(filename) as f:
        image_list = f.read().splitlines()
    # pdb.set_trace()
    image_list.sort()
    n_images = len(image_list)
    num_classes = len(CLASS_NAMES)
    if full:
        images = np.zeros((n_images,size,size,3))
        labels = np.zeros((n_images, num_classes))
        weights = np.zeros((n_images, num_classes))
    else:
        images = np.zeros((20,size,size,3))
        labels = np.zeros((20, num_classes))
        weights = np.zeros((20, num_classes))

    counter = 0
    # Read Image JPGs
    # for image in image_list:
    if full: 
        im_list = image_list
    else:
        im_list = image_list[:20]
    for image in im_list:
        imageJpgFile = osp.join(data_dir,'JPEGImages/'+image+'.jpg')
        img = Image.open(imageJpgFile)
        img = img.resize((size,size), Image.NEAREST)
        imageNp = np.array(img)
        images[counter,:,:,:] = imageNp
        counter+=1
    # Assign labels and weights
    cat_index = 0
    for cat in CLASS_NAMES:
        filename = osp.join(data_dir,'ImageSets/Main/'+cat+'_'+split+'.txt')
        with open(filename) as f:
            cat_list = f.read().splitlines()
        cat_list.sort()
        img_index = 0
        if full:
            c_list = cat_list
        else:
            c_list = cat_list[:20]
        for line in c_list:
        # for line in cat_list:
            # print(cat_index)
            if line[-2:]==' 1':
                labels[img_index][cat_index]=1
                weights[img_index][cat_index]=1
            elif line[-2:]=='-1':
                labels[img_index][cat_index]=0
                weights[img_index][cat_index]=1
            else:
                labels[img_index][cat_index]=0
                weights[img_index][cat_index]=0
            img_index+=1
        cat_index+=1
    print("##### Data Loaded #####")
    return np.float32(images),np.float32(labels),np.float32(weights)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr

def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()

def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        trainval_data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        test_data_dir, split='test')
    
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        keep_checkpoint_max = 2,
        save_summary_steps=1000,
        log_step_count_steps=1000)


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="VGGParamsTest",
        config=my_checkpoint_config)
    tensors_to_log = {"loss": "loss"}
    # pdb.set_trace()
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir='VGGParamsTest',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
  
    n_iter = []
    mAP_list = []
    randAP_list = []
    gtAP_list = []
    for i in range(50):
        n_iter.append(i)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data, "w": train_weights},
            y=train_labels,
            batch_size=1,
            num_epochs=None,
            shuffle=True)
            
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=800,
            hooks=[logging_hook, summary_hook])
        Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        pdb.set_trace()
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        randAP_list.append(np.mean(rand_AP))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        gtAP_list.append(np.mean(gt_AP))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        mAP_list.append(np.mean(AP))
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        if (i+1)%10==0:
            for cid, cname in enumerate(CLASS_NAMES):
                print('{}: {}'.format(cname, _get_el(AP, cid)))
        summary_var('VGGParamsFinal', 'mAP', np.mean(AP), i+1)

    with open('randAP_VGG', 'wb') as fp:
        pickle.dump(randAP_list, fp)
    with open('gtAP_VGG', 'wb') as fp:
        pickle.dump(gtAP_list, fp)
    with open('mAP_VGG', 'wb') as fp:
        pickle.dump(mAP_list, fp)
    
    plt.plot(n_iter, mAP_list)
    plt.ylabel('Test mAP')
    plt.xlabel('Iterations')
    plt.show()
    plt.savefig('mAP_Pascal_VGG.png')



if __name__ == "__main__":
    main()

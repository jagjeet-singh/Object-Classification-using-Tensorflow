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
size = 256

def cnn_model_fn(features, labels, mode, num_classes=20):
   
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    resize = lambda x:tf.image.resize_images(x, size=[320,320])
    rand_crop = lambda x:tf.random_crop(x,size=[256,256,3])
    cen_crop = lambda x: tf.image.central_crop(x, central_fraction=0.8)
    rand_flip = lambda x:tf.image.random_flip_left_right(x)


    if mode == tf.estimator.ModeKeys.TRAIN:
        input_aug = tf.map_fn(fn=resize, elems=input_layer)
        input_aug = tf.map_fn(fn=rand_crop, elems=input_aug)
        input_aug = tf.map_fn(fn=rand_flip, elems=input_aug)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        input_aug = tf.map_fn(fn=resize, elems=input_layer)
        input_aug = tf.map_fn(fn=cen_crop, elems=input_aug)

    else:
        input_aug = input_layer
  
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_aug,
        filters=96,
        kernel_size=[11, 11],
        strides=[4, 4],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer()
        )
    pdb.set_trace()
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)


    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])
    dense1 = tf.layers.dense(
        inputs=pool3_flat, 
        units=4096,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())

    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(
        inputs=dropout1, 
        units=4096,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer())

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
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
        # print("#########Loss:{}############".format(loss))
        # pdb.set_trace()
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # accuracy = tf.metrics.accuracy(
    #         labels=labels, predictions=predictions["classes"])
    # metrics = {'accuracy': accuracy}
    # # tf.summary.scalar('accuracy', accuracy[1])
    # # tf.summary.scalar('accuracy', loss[1])

    # if mode == tf.estimator.ModeKeys.EVAL:
    #     return tf.estimator.EstimatorSpec(
    #         mode, loss=loss, eval_metric_ops=metrics)

    # # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    # pdb.set_trace()
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
    # images = np.zeros((10,size,size,3))
    # labels = np.zeros((10, num_classes))
    # weights = np.zeros((10, num_classes))

    images = np.zeros((n_images,size,size,3))
    labels = np.zeros((n_images, num_classes))
    weights = np.zeros((n_images, num_classes))
    counter = 0
    # Read Image JPGs
    # for image in image_list:
    for image in image_list[:10]:

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
        for line in cat_list[:10]:
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


def main():
    args = parse_args()
    # Load training and eval data
    # train_data, train_labels, train_weights = load_pascal(
    #     args.data_dir, split='trainval')
    # eval_data, eval_labels, eval_weights = load_pascal(
    #     args.data_dir, split='test')
    train_data, train_labels, train_weights = load_pascal(
        trainval_data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        test_data_dir, split='test')


    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="/tmp/pascal_model_scratch")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # Train the model
    # pdb.set_trace()

    n_iter = []
    mAP_list = []
    randAP_list = []
    gtAP_list = []
    for i in range(5):
        n_iter.append(i)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=4,
        num_epochs=None,
        shuffle=True)
        
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=2,
            hooks=[logging_hook])
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        # pdb.set_trace()
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
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))

    with open('randAP_Alex', 'wb') as fp:
        pickle.dump(randAP_list, fp)
    with open('gtAP_Alex', 'wb') as fp:
        pickle.dump(gtAP_list, fp)
    with open('mAP_Alex', 'wb') as fp:
        pickle.dump(mAP_list, fp)
    
    plt.plot(n_iter, mAP_list)
    plt.ylabel('Test mAP')
    plt.xlabel('Iterations')
    plt.show()
    plt.savefig('mAP_Pascal_Alex.png')



if __name__ == "__main__":
    main()

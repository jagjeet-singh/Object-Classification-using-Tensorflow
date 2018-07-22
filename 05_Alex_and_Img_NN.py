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
from IPython import embed
from tensorflow.core.framework import summary_pb2
from sklearn.decomposition import PCA
import pickle
from random import randint
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

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
full=1
# reader = tf.train.NewCheckpointReader('vgg_16.ckpt')

def conv2d(inputs,filters, kernel_size,padding,activation,name):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=[1,1],
        padding=padding,
        activation=activation,
        # kernel_initializer=kernel_initializer,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        # bias_initializer=tf.zeros_initializer(),
        # bias_initializer=bias_initializer,
        name=name
        )

def dense(inputs,units,activation,name):
    return tf.layers.dense(
        inputs=inputs, 
        units=units,
        activation=activation,
        # kernel_initializer=kernel_initializer,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
        # bias_initializer=tf.zeros_initializer(),
        # bias_initializer=bias_initializer,
        name=name
        )


def ImageNet_model_fn(features, labels, mode, num_classes=20):
   
    # pdb.set_trace()


    input_layer = tf.reshape(features["x"], [-1, size, size, 3])
    resize = lambda x:tf.image.resize_images(x, size=[280,280])
    rand_crop = lambda x:tf.random_crop(x,size=[size,size,3])
    cen_crop = lambda x: tf.image.central_crop(x, central_fraction=0.8)
    rand_flip = lambda x:tf.image.random_flip_left_right(x)
    # reader = tf.train.NewCheckpointReader('vgg_16.ckpt')

    if mode == tf.estimator.ModeKeys.TRAIN:
        input_aug = tf.map_fn(fn=resize, elems=input_layer, name='resize_train')
        input_aug = tf.map_fn(fn=rand_crop, elems=input_aug, name='random_crop')
        input_aug = tf.map_fn(fn=rand_flip, elems=input_aug, name='random_flip')

    elif mode == tf.estimator.ModeKeys.PREDICT:
        input_aug = tf.map_fn(fn=resize, elems=input_layer, name='resize_test')
        input_aug = tf.map_fn(fn=cen_crop, elems=input_aug, name='center_crop')

    else:
        input_aug = input_layer
    
    with tf.variable_scope('conv1'):
        conv1 = conv2d(
        	input_aug,64,[3, 3],"same",tf.nn.relu,'conv1_1')
        conv2 = conv2d(
        	conv1,64,[3, 3],"same",tf.nn.relu,'conv1_2') 
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    with tf.variable_scope('conv2'):
    	conv3 = conv2d(
        	pool1,128,[3, 3],"same",tf.nn.relu,'conv2_1') 
    	conv4 = conv2d(
    		conv3,128,[3, 3],"same",tf.nn.relu,'conv2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv3'):
    	conv5 = conv2d(
        	pool2,256,[3, 3],"same",tf.nn.relu,'conv3_1') 
    	conv6 = conv2d(
        	conv5,256,[3, 3],"same",tf.nn.relu,'conv3_2') 
    	conv7 = conv2d(
        	conv6,256,[3, 3],"same",tf.nn.relu,'conv3_3')
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv4'):
    	conv8 = conv2d(
        	pool3,512,[3, 3],"same",tf.nn.relu,'conv4_1')
    	conv9 = conv2d(
        	conv8,512,[3, 3],"same",tf.nn.relu,'conv4_2') 
    	conv10 = conv2d(
        	conv9,512,[3, 3],"same",tf.nn.relu,'conv4_3') 
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv5'):
    	conv11 = conv2d(
        	pool4,512,[3, 3],"same",tf.nn.relu,'conv5_1')
    	conv12 = conv2d(
        	conv11,512,[3, 3],"same",tf.nn.relu,'conv5_2') 
    	conv13 = conv2d(
        	conv12,512,[3, 3],"same",tf.nn.relu,'conv5_3')
    
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)
    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    dense1 = dense(
    	pool5_flat, 4096,tf.nn.relu,'fc6')    	
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = dense(
    	dropout1, 4096,tf.nn.relu,'fc7')
    
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(
    	inputs=dropout2, units=20,name='fc8')

    predictions = {
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "pool5" : pool5,
        "fc7": dense2}


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def AlexNet_model_fn(features, labels, mode, num_classes=20):
   
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
    # pdb.set_trace()
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
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "pool5": pool3,
        "fc7": dense2
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
def load_pascal(data_dir, split, size):

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
    return np.float32(images),np.float32(labels),np.float32(weights), image_list

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

def showNN(AlexNet_classifier, fc7A, pool5A, ImageNet_classifier, fc7I, pool5I, eval_labels , Aeval_data, Ieval_data, num_Images, image_list):

    fc7ANeigh = NearestNeighbors()
    pool5ANeigh = NearestNeighbors()
    fc7INeigh = NearestNeighbors()
    pool5INeigh = NearestNeighbors()

    fc7ANeigh.fit(fc7A)
    pool5ANeigh.fit(pool5A)
    fc7INeigh.fit(fc7I)
    pool5INeigh.fit(pool5I)

    for i in range(num_Images):
        idx = randint(0,eval_labels.shape[0])
        # pdb.set_trace()
        AtestImage = np.expand_dims(Aeval_data[idx,:,:,:], axis=0)
        ItestImage = np.expand_dims(Ieval_data[idx,:,:,:], axis=0)
        # testWeight = np.expand_dims(eval_weights[idx,:], axis=0)
        testLabel = np.expand_dims(eval_labels[idx,:], axis=0)
        Aeval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": AtestImage},
            y=testLabel,
            num_epochs=1,
            shuffle=False)
        Ieval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": ItestImage},
            y=testLabel,
            num_epochs=1,
            shuffle=False)
        pred_AlexNet = list(AlexNet_classifier.predict(input_fn=Aeval_input_fn))
        fc7ATest = np.stack([f['fc7'] for f in pred_AlexNet])
        pool5ATest = np.stack([f['pool5'] for f in pred_AlexNet])
        pool5ATest = pool5ATest.reshape((pool5ATest.shape[0],pool5ATest.shape[1]*pool5ATest.shape[2]*pool5ATest.shape[3]))

        pred_ImgNet = list(ImageNet_classifier.predict(input_fn=Ieval_input_fn))
        fc7ITest = np.stack([f['fc7'] for f in pred_ImgNet])
        pool5ITest = np.stack([f['pool5'] for f in pred_ImgNet])
        pool5ITest = pool5ITest.reshape((pool5ITest.shape[0],pool5ITest.shape[1]*pool5ITest.shape[2]*pool5ITest.shape[3]))
    
        print("For image {0}".format(idx))
        
        fc7Aknn = fc7ANeigh.kneighbors(fc7ATest)
        print(fc7Aknn)
        fc7Iknn = fc7INeigh.kneighbors(fc7ITest)
        print(fc7Iknn)
        pool5Aknn = pool5ANeigh.kneighbors(pool5ATest)
        print(pool5Aknn)
        pool5Iknn = pool5INeigh.kneighbors(pool5ITest)
        print(pool5Iknn)

        fig = plt.figure(figsize=(15,10))
        fig.add_subplot(3,2,1)
        orig = np.array(Aeval_data[idx,:,:,:], dtype=np.uint8)
        plt.imshow(orig)
        plt.xlabel("Original Image")
        fig.add_subplot(3,2,3)
        fc7Img = np.array(Aeval_data[fc7Aknn[1][0][1],:,:,:], dtype=np.uint8)
        plt.imshow(fc7Img)
        plt.xlabel("AlexNet:FC7")
        fig.add_subplot(3,2,4)
        pool5Img = np.array(Aeval_data[pool5Aknn[1][0][1],:,:,:], dtype=np.uint8)
        plt.imshow(pool5Img)
        plt.xlabel("AlexNet:Pool5")
        fig.add_subplot(3,2,5)
        fc7Img = np.array(Ieval_data[fc7Iknn[1][0][1],:,:,:], dtype=np.uint8)
        plt.imshow(fc7Img)
        plt.xlabel("ImageNet:FC7")
        fig.add_subplot(3,2,6)
        pool5Img = np.array(Ieval_data[pool5Iknn[1][0][1],:,:,:], dtype=np.uint8)
        plt.imshow(pool5Img)
        plt.xlabel("ImageNet:Pool5")
        fig.savefig('Results/{0}_NN.png'.format(image_list[idx]))
        plt.show()



def main():
    args = parse_args()
    # # Load training and eval data
    # train_data, train_labels, train_weights = load_pascal(
    #     trainval_data_dir, split='trainval')
    Aeval_data, Aeval_labels, Aeval_weights, Aimage_list = load_pascal(
        test_data_dir, 'test', 256)

    Ieval_data, Ieval_labels, Ieval_weights, Iimage_list = load_pascal(
        test_data_dir, 'test', 224)    
    
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        keep_checkpoint_max = 1,
        save_summary_steps=100,
        log_step_count_steps=100)

    AlexNet_classifier = tf.estimator.Estimator(
        model_fn=partial(AlexNet_model_fn,
                         num_classes=Aeval_labels.shape[1]),
        model_dir="AlexParamsFinal",
        config=my_checkpoint_config)
    tensors_to_log = {"loss": "loss"}

    ImageNet_classifier = tf.estimator.Estimator(
        model_fn=partial(ImageNet_model_fn,
                         num_classes=Ieval_labels.shape[1]),
        model_dir="ImageNetParams",
        config=my_checkpoint_config)
    tensors_to_log = {"loss": "loss"}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=2)

    Aeval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Aeval_data, "w": Aeval_weights},
        y=Aeval_labels,
        num_epochs=1,
        shuffle=False)

    Ieval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Ieval_data, "w": Ieval_weights},
        y=Ieval_labels,
        num_epochs=1,
        shuffle=False)

    pred_AlexNet = list(AlexNet_classifier.predict(input_fn=Aeval_input_fn))
    pred_ImgNet = list(ImageNet_classifier.predict(input_fn=Ieval_input_fn))
    

    fc7A = np.stack([f['fc7'] for f in pred_AlexNet])
    pool5A = np.stack([f['pool5'] for f in pred_AlexNet])
    pool5A = pool5A.reshape((pool5A.shape[0],pool5A.shape[1]*pool5A.shape[2]*pool5A.shape[3]))

    fc7I = np.stack([f['fc7'] for f in pred_ImgNet])
    pool5I = np.stack([f['pool5'] for f in pred_ImgNet])
    pool5I = pool5I.reshape((pool5I.shape[0],pool5I.shape[1]*pool5I.shape[2]*pool5I.shape[3]))

    showNN(AlexNet_classifier, fc7A, pool5A, ImageNet_classifier, fc7I, pool5I, Aeval_labels , Aeval_data, Ieval_data, 25, Aimage_list)

if __name__ == "__main__":
    main()

# MNIST CNN

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Helper Functions

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    # x --> [batch,H,W,Channels]
    # W --> [filter H, filter W, Channels IN, Channels OUT]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# POOLING
def max_pool_2by2(x):
    # x --> [batch,H,W,Channels]
    # ksize: A list of `ints` that has length ` = 4`.  
    #        The size of the window for each dimension of the input tensor. 
    #        ksize参数里的第一个1：The only that basically pooling that I want to do is along the height and width 
    #        of an individual image which means as far as the size if concerned. I'm going to do a one.
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]]) # the bias just going to run along the third dimension ??
    return tf.nn.relu(conv2d(input_x, W) + b)

# NORMAL LAYER (FULLY CONNECTED)                        
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# Layers
# -1:表示同x的图片数量 1: gray scale
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Using a 6by 6 filer here, used 5by5 in video, you can play around with the filer size
# You can change the 32 output, that essentially represents the amout of filters used
# You need to pass in 32 to the next input though, the 1 comes from the original input of 
# a single image
# 1：gray scale
convo_1 = convolutional_layer(x_image, shape=[6,6,1,32])
convo_1_pooling = max_pool_2by2(convo_1)

# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can actually change the 64 output if you want, you can think of that as a rpresentation
# of the amout of 6by6 filters used.
# 32: 就是上一层的那个32
convo_2 = convolutional_layer(convo_1_pooling, shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# Why 7 by 7 image? Because we did 2 pooling layers, so (28/2)/2 = 7
# 64 then just comes from the output fo the previous Convolution
# 7*7：貌似是最终的图片尺寸
convo_2_flat = tf.reshape(convo_2_pooling, [-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# NOTE THE PLACEHGOLDER HERE!
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)

# Loss Functions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

# Initialize Variables
init = tf.global_variables_initializer()

# Session
#steps = 5000
steps = 1000
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        # 0.5: we are disigning that dropout layer appear. We have this whole probability which is a
        #   placeholder and that basically is the probability that a neuron is held during dropout
        #   So during training we'ss go ahead and say 50 percent. So each each neuron has 50 percent chance of
        #   being held essentially randomly droppin out half our neurons.
        sess.run(train, feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # print out a message every 100 steps
        if i%100 == 0:
            print('Currently on step {}'.format(i))

            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print('Accuracy is:', sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            
            print('\n')
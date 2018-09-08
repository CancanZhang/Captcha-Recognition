import tensorflow as tf
from utils.config import *

def cnn_model(input_tensor,dropout_rate):

    input_tensor = tf.reshape(input_tensor, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    
    conv1 = tf.layers.conv2d(
          inputs=input_tensor,
          filters=32,
          kernel_size=[5, 5],
          strides = [1,1],
          padding="same",
          activation=tf.nn.relu,
          name = 'conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    drop1 = tf.layers.dropout(inputs=pool1, rate=dropout_rate)
    
    conv2 = tf.layers.conv2d(
          inputs=drop1,
          filters=64,
          kernel_size=[5, 5],
          strides = [1,1],
          padding="same",
          activation=tf.nn.relu,
          name = 'conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    drop2 = tf.layers.dropout(inputs=pool2, rate=dropout_rate)
    
    conv3 = tf.layers.conv2d(
          inputs=drop2,
          filters=64,
          kernel_size=[5, 5],
          strides = [1,1],
          padding="same",
          activation=tf.nn.relu,
          name = 'conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool3')
    drop3 = tf.layers.dropout(inputs=pool3, rate=dropout_rate)
    
    pool_flat = tf.layers.flatten(drop3)
        
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu, name='dense1')
    drop = tf.layers.dropout(inputs=dense, rate=dropout_rate)
        
    logits = tf.layers.dense(inputs=drop, units=CHAR_SET_LEN*MAX_CAPTCHA, name='logits')
    return logits
    
def get_opt_op(logits,labels_tensor): 
    logits = tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    labels_tensor = tf.reshape(labels_tensor, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    with tf.variable_scope('loss'):
        #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_tensor, logits=logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_tensor))
    with tf.variable_scope('optimizer'):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(0.001, step, 100000, 0.96)
        opt_op = tf.train.AdamOptimizer(rate).minimize(loss)
    return opt_op,loss

def get_accuracy(logits,labels):
    logits = tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    labels = tf.reshape(labels, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        
    class_predict = tf.argmax(logits, 2, name='predict_class')
    class_label = tf.argmax(labels, 2)
    correct_pred = tf.equal(class_predict, class_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return class_predict,accuracy
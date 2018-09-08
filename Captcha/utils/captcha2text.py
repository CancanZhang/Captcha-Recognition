import tensorflow as tf
from utils.config import *
from utils.pre_process import *
import utils.inference as inference
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def captcha2text(address):
    binary_image = process_image(address)
    binary_image = binary_image.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH)
    with tf.Graph().as_default():

        address = address_latest_model
        input_tensor = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT , IMAGE_WIDTH])
        dropout_rate = tf.placeholder(tf.float32,name='dropout_rate')
        logits = inference.cnn_model(input_tensor,dropout_rate)

        saver = tf.train.Saver()
        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(address)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)

            logits = tf.reshape(logits, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            prediction = tf.argmax(logits, 2)
            class_pred = sess.run(prediction,feed_dict={input_tensor:binary_image, dropout_rate: 0.0})
            #print(class_pred)
            return index2text(class_pred[0])
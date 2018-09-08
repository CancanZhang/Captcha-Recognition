import tensorflow as tf

from utils.config import *
from utils.generate_data import *
import utils.inference as inference
#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

with tf.Graph().as_default():

    input_tensor = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT , IMAGE_WIDTH],name='input')
    labels_tensor = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    dropout_rate = tf.placeholder(tf.float32,name='dropout_rate')

    logits = inference.cnn_model(input_tensor,dropout_rate)
    opt_op,loss = inference.get_opt_op(logits,labels_tensor)
    class_predict,accuracy = inference.get_accuracy(logits,labels_tensor)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1 
        train_accuracy = 0
        test_accuracy = 0
        while True:
            data_train = Generate_Data(BATCH_SIZE_TRAIN)
            batch_x,batch_y = data_train.xs, data_train.ys
            _,value_loss= sess.run([opt_op,loss],feed_dict={input_tensor:batch_x, labels_tensor:batch_y, dropout_rate:DROPOUT_RATE})
            step += 1
            if step % 50 == 0:
                data_test = Generate_Data(BATCH_SIZE_TEST)
                test_x,test_y = data_test.xs, data_test.ys
                train_accuracy = sess.run(accuracy,feed_dict={input_tensor:batch_x,labels_tensor:batch_y, dropout_rate:0})
                test_accuracy = sess.run(accuracy,feed_dict={input_tensor:test_x,labels_tensor:test_y, dropout_rate:0})
                print('step {} | Loss: {} | Train Accuracy: {} | Test Accuracy {}'.format(step,value_loss,train_accuracy, test_accuracy))

                # show example in training set
                example_x = batch_x[0].reshape(-1,IMAGE_HEIGHT , IMAGE_WIDTH)
                example_y = batch_y[0].reshape(-1,MAX_CAPTCHA * CHAR_SET_LEN)
                predict_y = sess.run(class_predict,feed_dict={input_tensor:example_x,dropout_rate:0})
                predict_y = index2text(predict_y[0])
                example_y = vec2text(example_y[0])
                print ('Example in Training Set: Predict: {} | Real: {}').format(predict_y,example_y)

                # show example in training set
                example_x = test_x[0].reshape(-1,IMAGE_HEIGHT , IMAGE_WIDTH)
                example_y = test_y[0].reshape(-1,MAX_CAPTCHA * CHAR_SET_LEN)
                predict_y = sess.run(class_predict,feed_dict={input_tensor:example_x,dropout_rate:0})
                predict_y = index2text(predict_y[0])
                example_y = vec2text(example_y[0])
                print ('Example in Test Set: Predict: {} | Real: {}').format(predict_y,example_y)

                saver.save(sess, address_saved_model, global_step=step)
            if test_accuracy > 0.95:
                break
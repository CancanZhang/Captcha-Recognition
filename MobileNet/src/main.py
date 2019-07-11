%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.applications import MobileNet
from keras.utils import multi_gpu_model
import os
import datetime

from config import *
from utils import *
from generate_data import *
from evaluate import *
from mycbk import *

np.random.seed(seed=1992)
tf.set_random_seed(seed=1992)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "12"

def draw_hist(hists):
    hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists])
    hist_df.index = np.arange(1, len(hist_df)+1)
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
    axs[0].plot(hist_df.val_accuracy, lw=5, label='Validation Accuracy')
    axs[0].plot(hist_df.accuracy, lw=5, label='Training Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid()
    axs[0].legend(loc=0)
    axs[1].plot(hist_df.categorical_crossentropy, lw=5, label='Validation MLogLoss')
    axs[1].plot(hist_df.val_categorical_crossentropy, lw=5, label='Training MLogLoss')
    axs[1].set_ylabel('MLogLoss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid()
    axs[1].legend(loc=0)
    fig.savefig(address_hist, dpi=300)
    plt.show();

img_gen = Generate_Data(train_num = 128*4)
[char_len,char_num] = get_char_length_and_number()  

input_shape = (IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL)
model = MobileNet(input_shape=input_shape,alpha=1.,weights=None,classes=char_num*char_len)

#parallel_model = multi_gpu_model(model, 2)
adam = keras.optimizers.Adam(lr = 0.005, beta_1=0.9, beta_2=0.999,decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[accuracy,categorical_accuracy,categorical_crossentropy])
cbk = MyCbk(model)
tensorboard = TensorBoard(log_dir=address_tensorboard,histogram_freq=0)
eva = Evaluate(img_gen.test())
moniter = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, mode='max', cooldown=3, verbose=1)

hists = []
hist = model.fit_generator(generator=img_gen.next_train(),
                    steps_per_epoch = 600,
                    epochs = 16,
                    validation_data = img_gen.next_val(),
                    validation_steps = 1,
                    verbose = 1,
                    callbacks = [cbk,eva,moniter,tensorboard])
hists.append(hist)
draw_hist(hists)


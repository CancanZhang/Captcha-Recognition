from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, LSTM,GRU, Bidirectional, Dropout, Flatten, Wrapper, Activation, TimeDistributed
from keras.layers import Reshape, Lambda, RepeatVector, Concatenate, Permute, Dot, Multiply, Permute,merge,BatchNormalization 
from keras.models import Model
from keras import backend as K
from keras.engine import InputSpec
from keras import initializers, regularizers, constraints

from config import *
from utils import *

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    time_steps = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def cnn_lstm_attention_model():
    
    [char_len,char_num] = get_char_length_and_number()  
    conv_filters = CONV_FILTERS
    kernel_size = (KERNEL_SIZE, KERNEL_SIZE)
    pool_size = POOL_SIZE
    
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL)

    input_data = Input(name='the_input',shape=input_shape,dtype='float32')
          
    inner = input_data
    for i in range(0,NUM_CNN_LAYERS):
        inner = Conv2D(conv_filters,
                       kernel_size,
                       padding='same',
                       activation = None, 
                       kernel_initializer='he_normal',
                       name=('conv'+str(i+1)))(inner)
        inner = BatchNormalization(name=('cnn_norm'+str(i+1)))(inner)
        inner = Activation('relu',name=('cnn_relu'+str(i+1)))(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name=('max'+str(i+1)))(inner)
       
    conv_to_rnn_dims = (IMAGE_WIDTH // (POOL_SIZE ** NUM_CNN_LAYERS), 
                        (IMAGE_HEIGHT // (POOL_SIZE ** NUM_CNN_LAYERS)) * CONV_FILTERS)
    
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    attention_mul = attention_3d_block(inner)
    
    inner = Bidirectional(GRU(GRU_SIZE,return_sequences=True,dropout=0.25,recurrent_dropout=0.1), name='gru1')(attention_mul)
    inner = BatchNormalization(name=('gru_norm1'))(inner)
    inner = Bidirectional(GRU(GRU_SIZE,return_sequences=True,dropout=0.25,recurrent_dropout=0.1), name='gru2')(inner)
    inner = BatchNormalization(name=('gru_norm2'))(inner)
    
    outputs = TimeDistributed(Dense(char_num,activation='sigmoid'),name='dense')(inner)   
    
    Model(inputs=input_data, outputs=outputs).summary()
    model = Model(inputs=input_data, outputs=outputs)
    
    return model
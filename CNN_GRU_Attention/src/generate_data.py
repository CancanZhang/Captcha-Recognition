import numpy as np

from config import *
from utils import *

class Generate_Data():
    
    def __init__(self,train_num=512,val_num=128):
        self.train_num = train_num
        self.val_num = val_num
        [self.char_len,self.char_num] = get_char_length_and_number()    
        
    def next_train(self):
        while True:
            ret = self.get_next_batch(self.train_num)
            yield ret[1:]
            
    def next_val(self):
        while True:
            ret = self.get_next_batch(self.val_num)
            yield ret[1:]
            
    def test(self):
        ret = self.get_next_batch(128)
        image = ret[0]
        x = ret[1]
        y = ret[2]
        return [image,x,y]
    
    def get_next_batch(self,batch_size=128):
        images = np.zeros([batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,3])
        batch_x = np.zeros([batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL])
        batch_y = np.zeros([batch_size,self.char_len*self.char_num],dtype='int32')
        flag = 0
        
        while True:
            if FLAG_CHAR == 0:
                text = make_rand_text(self.char_len)
            elif FLAG_CHAR == 1:
                random_len = np.random.randint(MIN_CHAR_LEN,MAX_CHAR_LEN+1)
                text = make_rand_text(random_len)
            image = make_rand_image(text)
            images[flag,:,:,:] = np.reshape(np.array(image),[IMAGE_HEIGHT,IMAGE_WIDTH,3])
            if CHANNEL == 1:
                image = rgb2gray(np.array(image))  
            batch_x[flag,:,:,:] = np.reshape(image,[IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL])
            batch_y[flag,:] = text2vec(text)
            flag += 1
            if flag >= batch_size:
                break
        batch_x = np.transpose(batch_x,(0,2,1,3))
        batch_y = np.reshape(batch_y,[batch_size,self.char_len,self.char_num])
        inputs = batch_x
        outputs = batch_y
        return images,inputs,outputs
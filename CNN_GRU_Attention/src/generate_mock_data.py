import numpy as np
import random
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import math
import time 
import keras



class Generate_Mock_Data():


    def __init__(self,img_w_min=160,img_w_max=160,img_h=60,channel=1,char_num=62,min_char_len=4,max_char_len=8,train_size=512,val_size=128):
        
        self.img_w_min = img_w_min
        self.img_w_max = img_w_max
        self.img_h = img_h
        self.channel = channel
        self.char_num = char_num
        self.min_char_len = min_char_len
        self.max_char_len = max_char_len
        self.train_size = train_size
        self.val_size = val_size
        
    def text2vec(self,text):
        text_len = len(text)

        if text_len > self.char_num:
            raise ValueError(u'Text length does not fit')

        vector = np.zeros(self.max_char_len * self.char_num)
        for i in range(self.max_char_len):
            if i < text_len:
                idx = i * self.char_num + char2pos(text[i])
            else:
                idx = i * self.char_num + char2pos('')
            vector[idx] = 1        
        return vector
            
    def get_next_batch(self,batch_size=128):  
        imgs = np.zeros([batch_size, self.img_h, self.img_w_max, 3])
        batch_x = np.zeros([batch_size, self.img_h, self.img_w_max, self.channel])
        batch_y = np.zeros([batch_size, self.char_num * self.max_char_len],dtype='int32')
            
        i = 0 
        while True:
            start = time.clock()
            if i > 500 and i % 500 == 0:
                print ( '%d / %d, time: %.2f\n' % (i,batch_size,time.clock()-start))
                
            text, image = self.produce_mock_captcha()
            image = np.array(image)
            origin = image
            if len(image.shape) == 3:
                img_h = image.shape[0]
                img_w = image.shape[1]
                if self.channel == 1:
                    # RGB to gray 
                    image= rgb2gray(image)
                if img_h == self.img_h and img_w <= self.img_w_max and img_w >= self.img_w_min:
                    image = np.reshape(image,[img_h,img_w,self.channel])
                    imgs[i,0:img_w,:,:] = np.array(origin) 
                    batch_x[i,0:img_w,:,:] = np.array(image)
                    batch_y[i,:] = self.text2vec(text)
                    i += 1
                else:
                    print (image.shape)
                    print ('Image size does not fit...')
            else:
                print ('image does have dimension 3')
            if int(i) == int(batch_size):
                break
        #batch_y = batch_y.reshape([batch_size, MAX_CHAR_LEN , NUM_CHAR])  
        #batch_y = list(batch_y.swapaxes(0,1))
            
        #batch_x,batch_y = shuffle_data(batch_x,batch_y)
        batch_x = np.transpose(batch_x,(0,2,1,3))
        inputs = batch_x
        outputs = batch_y
            
        return (imgs,inputs,outputs)        

    
    @staticmethod
    def veticle_move(x,image_width):
        # contrale the verticle distance distort by rotate text
        if x <= 0:
            dis = 0
        elif x > 0:
            dis = int(image_width * math.tan(math.pi/(180/x)))
        dis += np.random.randint(-5,5)
        return dis
    
    @staticmethod
    def GaussieNoisy(img,mean,sigma):
        img = np.array(img)
        gauss = np.random.normal(mean,sigma,img.shape)
        img = img + gauss
        return img
    
    @staticmethod
    def SaltPepperNoisy(img,n):
        # n: percent of pixles changed to 0 or 255
        img = np.array(img)
        m = int(img.shape[0]*img.shape[1]*n)
        for a in range(m):
            i = np.random.randint(0,img.shape[0])
            j = np.random.randint(0,img.shape[1])
            if len(img) == 2:
                img[i,j] = 255
            elif len(img) == 3:
                img[i,j,0] = 255
                img[i,j,1] = 255
                img[i,j,2] = 255
        for b in range(m):
            i = np.random.randint(0,img.shape[0])
            j = np.random.randint(0,img.shape[1])
            if len(img) == 2:
                img[i,j] = 0
            elif len(img) == 3:
                img[i,j,0] = 0
                img[i,j,1] = 0
                img[i,j,2] = 0
        return img    
    
    def produce_mock_captcha(self):
        
        # random char num
        if self.min_char_len < self.max_char_len:
            char_len = random.randint(self.min_char_len,self.max_char_len)
        else:
            char_len = self.max_char_len
            
        # random image width
        if self.img_w_min < self.img_w_max:
            while True:
                img_w = char_len * 40 + random.randint(-30,30)
                if img_w < self.img_w_max and img_w > self.img_w_min:
                    break
        else:
            img_w = self.img_w_min
            
        # random background color
        random_background = np.random.rand()
        if random_background < 0.5:
            image = Image.new('RGB',[img_w,self.img_h],(0,0,0))# black background
        else:
            image = Image.new('RGB',[img_w,self.img_h],(255,255,255))# white background
            
        # choose random font
        char = ''
        total_dis = 0
        root_address = 'font/'
        num_fonts = len(os.listdir(root_address))                         
        while True:
            random_font = np.random.randint(num_fonts)
            
            if os.listdir(root_address)[random_font][-4:] in ['.otf','.ttf']:
                random_font = root_address + os.listdir(root_address)[random_font]
                break
            else:
                continue    
                
        # draw each character in the image one by one
        for i in range(char_len):

            # generate random character
            rand_char = ''
            r_char = make_rand_char()
            rand_char += r_char
            char += rand_char
            
            # generate random fontsize, color, rotate angle, distance from the last character
            standard_fontsize = 45
            fontsize = int(60 - char_len**1.5)
            font = ImageFont.truetype(random_font, fontsize)
            
            fontcolor = (np.random.randint(5,250),np.random.randint(5,250),np.random.randint(5,250))
            rand_rotate = np.random.randint(-15,15)
            distance = char_len - self.max_char_len #np.random.randint(-2,0) # contral overlapping 

            txt=Image.new('RGBA',[img_w,self.img_h],(0,0,0,1))
            d = ImageDraw.Draw(txt)
            d.text((0,0),rand_char,font=font,fill=fontcolor)
            w=txt.rotate(rand_rotate, expand=1)#.convert('RGBA')
            mask = Image.fromarray(np.uint8(255*(np.array(w) > 50))).convert('RGBA')
            
            verticle_dis = 2 - Generate_Mock_Data.veticle_move(rand_rotate,img_w)
            image.paste(w,(total_dis,verticle_dis),mask)  
            total_dis += (int(img_w/char_len-1)+ distance)
            
        # add noisy
        random_noise = np.random.rand()
        if random_noise < 0.5:
            n = np.random.rand()
            image = Generate_Mock_Data.SaltPepperNoisy(image,n) # add salt pepper noisy
        else:
            mean = np.random.randint(255/4.0,255/2.0)
            sigma = np.random.randint(5,20)
            image = Generate_Mock_Data.GaussieNoisy(image,mean,sigma) # add gauss noisy
            
        image = Image.fromarray(image.astype(np.uint8))
        return char.lower(),image
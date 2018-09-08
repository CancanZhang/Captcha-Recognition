import numpy as np
import random
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image

from utils.config import *
from utils.pre_process import *

class Generate_Data():
    
    def __init__(self,batch_size = 128):
        self.xs, self.ys = Generate_Data.get_next_batch(batch_size)
        
    @staticmethod
    def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text
    
    @staticmethod
    def gen_gary_captcha_text_and_image():
        image = ImageCaptcha()

        captcha_text = Generate_Data.random_captcha_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)

        gray_image = rgb2gray(captcha_image)
        binary_image = gray2binary(gray_image,200)

        return captcha_text, binary_image
    
    @staticmethod
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = Generate_Data.gen_gary_captcha_text_and_image()
            if image.shape != (60, 160):
                continue
            return text, image
        
    @staticmethod
    def get_next_batch(batch_size=128):
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

        for i in range(batch_size):
            text, image = Generate_Data.wrap_gen_captcha_text_and_image()
            batch_x[i, :] = image
            batch_y[i, :] = text2vec(text)
        return batch_x, batch_y
    
    @staticmethod
    def gen_and_show():
        text, image = Generate_Data.wrap_gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()
    
import numpy as np
import cv2
from utils.config import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2binary(gray,threshold):
    binary = np.full(gray.shape,0,dtype=int) # all white
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] < threshold: # black
                binary[i][j] = 1
    return binary

def char2pos(c):
    if c == '_':
        k = 62
    elif c >= '0' and c <= '9':
        k = ord(c) - ord('0')
    elif c >= 'a' and c <= 'z':
        k = ord(c) - ord('a') + 10
    elif c >= 'A' and c <= 'Z':
        k = ord(c) - ord('A') + 36
    else:
        print ('character not in lists')
    return k

def pos2char(char_idx):

    if char_idx < 10:
        char_code = char_idx + ord('0')
    elif char_idx < 36:
        char_code = char_idx - 10 + ord('a')
    elif char_idx < 62:
        char_code = char_idx - 36 + ord('A')
    elif char_idx == 62:
        char_code = ord('_')
    else:
        raise ValueError('error')

    return chr(char_code)

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError(u'Max Length: 4')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN

        char_code = pos2char(char_idx)

        text.append(char_code)
    return "".join(text)

def index2text(vec):
    text = []
    for i in vec:
        char_code = pos2char(i)
        text.append(char_code)
    return "".join(text)

def reshape(img,threshold1, threshold2):
    i = 0
    while True:
        if sum(img[i]) > threshold1:
            break
        i += 1
    if i - 2 > 0:
        i = i - 2
    
    j = img.shape[0] - 1
    while True:
        if sum(img[j]) > threshold1:
            break
        j -= 1
    if j + 5 < img.shape[0] - 1:
        j = j + 5
    
    k = 0
    while True:
        if sum(img[:,k]) > threshold2:
            break
        k += 1
    if k - 2 > 0:
        k = k - 2   
        
    img = img[i:j,k:]
    
    return i,j,k

def process_image(address):
    
    captcha_image = cv2.imread(address)
    captcha_image = np.array(captcha_image)
    captcha_image = cv2.medianBlur(captcha_image,3)
    gray_image = rgb2gray(captcha_image)
    binary_image = gray2binary(gray_image,120)
    binary_image = 1 - binary_image
    i,j,k = reshape(binary_image,5,3)
    captcha_image = captcha_image[i:j,k:]
    captcha_image = cv2.resize(captcha_image,(IMAGE_WIDTH,IMAGE_HEIGHT)) 
    gray_image = rgb2gray(captcha_image)
    binary_image = gray2binary(gray_image,120)
    binary_image = 1 - binary_image
    
    return binary_image

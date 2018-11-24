%matplotlib inline
import matplotlib.pyplot as plt

from config import *
from utils import *
from generate_data import *
from evaluate import *
from mycbk import *
from inference import *

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "11"

def draw_predict(img,y_pred):
    plt.rcParams['figure.figsize'] = [16, 10]
    plt.rcParams['font.size'] = 14

    n = 5
    fig,axs = plt.subplots(nrows=n,ncols=n,sharex=True,sharey=True,figsize=(16,6))
    for i in range(n**2):
        ax = axs[i // n, i % n]
        ax.imshow(img[i].astype(np.uint8))
        ax.text(135,6,pred2text(y_pred[i]),fontsize=15,color = 'blue',
                bbox=dict(boxstyle="square",facecolor='wheat'))
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(address_predict, dpi=300)
    plt.show()
    
[char_len,char_num] = get_char_length_and_number()  
input_shape = (IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL)
model = cnn_lstm_attention_model()
model.load_weights(address_model)

[img,x,y] = Generate_Data().test()
y_pred = model.predict(x)

draw_predict(img,y_pred)
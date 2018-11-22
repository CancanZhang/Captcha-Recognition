number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
CHAR_NUM = len(number) + len(alphabet) + len(ALPHABET)

FLAG_CHAR = 0 # 0:fixed length; 1:variable lenght
CHAR_LEN = 4
MIN_CHAR_LEN = 4
MAX_CHAR_LEN = 6

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

CHANNEL = 1

NUM_CNN_LAYERS = 3
CONV_FILTERS = 32
KERNEL_SIZE = 3
POOL_SIZE = 2

DENSE_SIZE = 256
DROPOUT_RATE = 0.5

address_tensorboard = '../show/'
address_model = '../best_model.hdf5'
address_hist = '../img/hist.png'
address_predict = '../img/predict.png'
address_font = '../font/'

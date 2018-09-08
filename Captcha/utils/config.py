number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
CHAR_SET_LEN = len(number) + len(alphabet) + len(ALPHABET)

address_saved_model = 'tmp/captha.try'
address_latest_model = 'tmp/captha.try'

DROPOUT_RATE = 0.4

BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 100

GPU_ID = "11"
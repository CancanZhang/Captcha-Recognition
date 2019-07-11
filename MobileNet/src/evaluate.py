import keras
from utils import *

class Evaluate(keras.callbacks.Callback):
    
    # random show a result in test set
    def __init__(self,val):
        self.x = val[1]
        self.y = val[2]
        self.random_index = np.random.randint(self.y.shape[0])
        
    def on_epoch_end(self,epoch,logs={}):
        
        y_pred = self.model.predict(self.x)
        true = vec2text(self.y[self.random_index,:])
        pred = pred2text(y_pred[self.random_index,:])
        print ('True: ',true)
        print ('Pred: ',pred)
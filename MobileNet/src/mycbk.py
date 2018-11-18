import keras

class MyCbk(keras.callbacks.Callback):
    # for multi-gpu
    def __init__(self, model,logs={}):
        self.model_to_save = model
        self.min_loss = float('Inf')

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')
        if current_loss < self.min_loss:
            self.model_to_save.save(address_model)
            self.min_loss = current_loss
            print (u'val loss improved, new model saved')
        else:
            print (u'val loss does not improve in this epoch')
        
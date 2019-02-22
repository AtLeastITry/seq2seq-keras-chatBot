from tensorflow.keras.callbacks import Callback
import sys
import os
sys.path.append("..")
script_dir = os.path.dirname(__file__)
from Config import BATCH_TO_SAVE

class BatchSaver(Callback):
    def ___init___(self):
        self.__batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.__batch & BATCH_TO_SAVE == 0:
            name = script_dir + '/../weights/model_weights_on_batch%08d.h5' % self.__batch
            self.model.save_weights(name)
        self.__batch += 1
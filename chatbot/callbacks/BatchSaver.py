from tensorflow.keras.callbacks import Callback
import sys
import os
sys.path.append("..")
script_dir = os.path.dirname(__file__)
from Config import BATCH_TO_SAVE
from tensorflow.keras.models import save_model

class BatchSaver(Callback):
    def ___init___(self):
        self.N = BATCH_TO_SAVE

    def on_batch_end(self, batch, logs={}):
        if batch % BATCH_TO_SAVE == 0:
            model_weights_name = script_dir + '/../weights/model_weights_on_batch%08d.h5' % batch
            model_name = script_dir + '/../weights/model_on_batch%08d.h5' % batch
            self.model.save_weights(model_weights_name)
            save_model(self.model, model_name)
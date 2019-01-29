from tensorflow.keras.layers import Input, LSTM, Dense, CuDNNLSTM
from tensorflow.test import is_gpu_available
import sys
sys.path.append("..")

from Config import UNIT_SIZE # pylint: disable=E0401

class Encoder:
    def __init__(self, input_data):
        self.inputs = Input(shape=(None, input_data.num_tokens))
        if is_gpu_available(cuda_only = True, min_cuda_compute_capability=3.7):   
            self.lstm = CuDNNLSTM(units=UNIT_SIZE, return_state=True, name='encoder_lstm', stateful=False)
        else:
            self.lstm = LSTM(units=UNIT_SIZE, return_state=True, name='encoder_lstm', stateful=False)
            
        encoder_outputs, state_h, state_c = self.lstm(self.inputs)

        self.encoder_outputs = encoder_outputs
        self.states = [state_h, state_c]

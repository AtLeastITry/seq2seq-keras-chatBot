from tensorflow.keras.layers import Input, LSTM, Dense, CuDNNLSTM
from tensorflow.test import is_gpu_available
import sys
sys.path.append("..")

from Config import UNIT_SIZE # pylint: disable=E0401

class Decoder: 
    def __init__(self, output_data, encoder):
        self.inputs = Input(shape=(None, output_data.num_tokens))
        if is_gpu_available(cuda_only = True, min_cuda_compute_capability=3.7):   
            self.lstm = CuDNNLSTM(units=UNIT_SIZE, return_sequences=True, return_state=True, name='decoder_lstm', stateful=False)
        else:
            self.lstm = LSTM(units=UNIT_SIZE, return_sequences=True, return_state=True, name='decoder_lstm', stateful=False)
        

        outputs, _, _ = self.lstm(self.inputs,initial_state=encoder.states)
        
        self.dense = Dense(output_data.num_tokens, activation='softmax')
        self.outputs = self.dense(outputs)

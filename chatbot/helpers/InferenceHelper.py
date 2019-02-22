# pylint: disable=E0401,E0611

import sys
sys.path.append("..")

import os
from .DataService import DataService
from models.InputData import InputData
from models.OutputData import OutputData
from models.DataResult import DataResult
from models.Encoder import Encoder
from models.Decoder import Decoder
from Config import BATCH_SIZE, EPOCHS, UNIT_SIZE

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

class InferenceHelper:
    def __init__(self, data_result, model = None):
        self.data_result = data_result

        if (model == None):
            temp_encoder = Encoder(self.data_result.input_data)
            temp_decoder = Decoder(self.data_result.output_data, temp_encoder)
            temp_model = Model([temp_encoder.inputs, temp_decoder.inputs], temp_decoder.outputs)
            temp_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
            temp_model.load_weights(os.path.dirname(__file__) + '/../model_weights.h5')
            self.model = temp_model
        else:
            self.model = model

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.data_result.input_data.chars)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.data_result.output_data.chars)])
        

        self.encoder_inputs = self.model.input[0]   # input_1
        self.encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output   # lstm_1
        self.encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        self.decoder_inputs = self.model.input[1]   # input_2
        self.decoder_state_input_h = Input(shape=(UNIT_SIZE,), name='input_3')
        self.decoder_state_input_c = Input(shape=(UNIT_SIZE,), name='input_4')
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_lstm = self.model.layers[3]
        self.decoder_outputs, self.state_h_dec, self.state_c_dec = self.decoder_lstm(self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [self.state_h_dec, self.state_c_dec]
        self.decoder_dense = self.model.layers[4]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def __serialize_text(self, text):
        encoder_input_data = np.zeros((1, self.data_result.input_data.max_len, self.data_result.input_data.num_tokens),dtype='float32')
        for t, char in enumerate(text):
            encoder_input_data[0, t, self.input_token_index[char]] = 1.
        return encoder_input_data

    def __decode_input(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.data_result.output_data.num_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > self.data_result.output_data.max_len):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.data_result.output_data.num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def predict(self, text):
        input_seq = self.__serialize_text(text)
        text = self.__decode_input(input_seq)
        return text

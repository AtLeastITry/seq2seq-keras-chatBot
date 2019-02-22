# pylint: disable=E0401,E0611

import os
import json
script_dir = os.path.dirname(__file__)

from helpers.DataService import DataService
from models.InputData import InputData
from models.OutputData import OutputData
from models.DataResult import DataResult
from models.Encoder import Encoder
from models.Decoder import Decoder
from callbacks.BatchSaver import BatchSaver
from Config import BATCH_SIZE, EPOCHS, LIMIT_GPU_USAGE
from generator.TrainingGenerator import TrainingGenerator

from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import backend as ktf

def get_session(gpu_fraction=0.3):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if (LIMIT_GPU_USAGE):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ktf.set_session(get_session())
# Process the dataset
print('STARTING: loading_data')
data_result = DataResult(None, None)

with open(script_dir + './temp/processed_data.json', 'r') as output:
    json_data = json.load(output)
    data_result.loadJSON(json_data)
print('END: loading_data')
print('')

# Create the encoder
print('STARTING: create encoder')
encoder = Encoder(data_result.input_data)
print('END: create encoder')
print('')

# Create the decoder
print('STARTING: create decoder')
decoder = Decoder(data_result.output_data, encoder)
print('STARTING: create decoder')
print('')

# Create the model
print('STARTING: create model')
model = Model([encoder.inputs, decoder.inputs], decoder.outputs)
print('END: create model')
print('')

# Compile the model
print('STARTING: compile model')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print('END: compile model')
print('')

# Train the model
print('STARTING: train model')
print(' Training with ' + str(data_result.input_data.num_lines) + ' lines')

generator = TrainingGenerator(data_result, BATCH_SIZE)
model.fit_generator(generator, epochs=EPOCHS, verbose=1, callbacks=[BatchSaver()])
# model.fit([token_result.encoder_input, token_result.decoder_input], token_result.decoder_output, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
print('END: train model')
print('')

#Save the entire model
save_model(model, 'model.h5')

#Save the weights for cpu compatibility
model.save_weights('model_weights.h5')

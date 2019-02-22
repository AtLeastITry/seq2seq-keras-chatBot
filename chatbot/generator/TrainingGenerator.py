import sys
import os
sys.path.append("..")
script_dir = os.path.dirname(__file__)
from helpers.DataService import DataService
from models.InputData import InputData
from models.OutputData import OutputData
from models.DataResult import DataResult
from tensorflow.keras.utils import Sequence

import numpy as np
import mmap

class TrainingGenerator(Sequence):
    def __init__(self, data_result, batch_size):
        self.input_file = script_dir + "/../data/input.txt"
        self.output_file = script_dir + "/../data/output.txt"

        self.temp_input_batch_file = script_dir + '/../temp/_input_batch_file.txt'
        self.temp_output_batch_file = script_dir + '/../temp/_output_batch_file.txt'

        self.batch_size = batch_size
        self.current_batch_index = 0
        self.internal_index = 0

        self.data_result = data_result
        self.total_lines = self.data_result.input_data.num_lines

        self.input_index = dict()
        self.output_index = dict()

        self.input_file_stream = open(self.input_file, 'r+')
        self.output_file_stream = open(self.output_file, 'r+')

        batch_input_file_stream = open(self.temp_input_batch_file, 'r+')
        batch_output_file_stream = open(self.temp_output_batch_file, 'r+')

        self.batch_input_mmap = mmap.mmap(batch_input_file_stream.fileno(), 0)
        self.batch_output_mmap = mmap.mmap(batch_output_file_stream.fileno(), 0)

        self.__process_index()
    
    def __len__(self):
        return int(np.floor(self.total_lines / self.batch_size))

    def __getitem__(self, index):
        x, y = self.__data_generation()
        self.current_batch_index = self.current_batch_index + 1
        return x, y

    def on_epoch_end(self):
        self.internal_index = 0

    def __process_index(self):
        for i, char in enumerate(self.data_result.input_data.chars):
            self.input_index[char] = i

        for i, char in enumerate(self.data_result.output_data.chars):
            self.output_index[char] = i

    def __get_input_seek(self):
        result = 0
        i = 0
        
        self.batch_input_mmap.seek(0)

        while True:
            line = self.batch_input_mmap.readline().decode("utf-8")
            
            if not line:
                break

            if (i == self.current_batch_index):
                result = float(line.rstrip())
                break
            i += 1  

        return result


    def __get_output_seek(self):
        result = 0
        i = 0
        
        self.batch_output_mmap.seek(0)

        while True:
            line = self.batch_output_mmap.readline().decode("utf-8")
            
            if not line:
                break

            if (i == self.current_batch_index):
                result = float(line.rstrip())
                break
            i += 1  
            
        return result

    def __process_batch_data(self):
        encoder_input = np.zeros((self.batch_size, self.data_result.input_data.max_len, self.data_result.input_data.num_tokens), dtype='float32')
        decoder_input = np.zeros((self.batch_size, self.data_result.output_data.max_len, self.data_result.output_data.num_tokens), dtype='float32')
        decoder_output = np.zeros((self.batch_size, self.data_result.output_data.max_len, self.data_result.output_data.num_tokens), dtype='float32')

        temp_index = 0
        input_seek = self.__get_input_seek()
        output_seek = self.__get_output_seek()

        self.input_file_stream.seek(input_seek)
        self.output_file_stream.seek(output_seek)

        for i, (input_line, output_line) in enumerate(zip(self.input_file_stream, self.output_file_stream)):
            if temp_index == self.batch_size:
                break

            output_line = "\t" + output_line + "\n"

            for j, char in enumerate(input_line):
                try:
                    encoder_input[temp_index, j, self.input_index[char]] = 1
                except Exception as e: 
                    print(e)
                    print('input_line: ' + str(input_line))
                    print('output_line: ' + str(output_line))
                    print('temp_index: ' + str(temp_index))
                    print('j: ' + str(j))
                    print('char: ' + str(char))

            for j, char in enumerate(output_line):
                try:
                    decoder_input[temp_index, j, self.output_index[char]] = 1
                except Exception as e: 
                    print(e)
                    print('input_line: ' + str(input_line))
                    print('output_line: ' + str(output_line))
                    print('temp_index: ' + str(temp_index))
                    print('j: ' + str(j))
                    print('char: ' + str(char))

                if j > 0:
                    try:
                        decoder_output[temp_index, j -1, self.output_index[char]] = 1
                    except Exception as e: 
                        print(e)
                        print('input_line: ' + str(input_line))
                        print('output_line: ' + str(output_line))
                        print('temp_index: ' + str(temp_index))
                        print('j: ' + str(j))
                        print('char: ' + str(char))
                    
            temp_index += 1
            
        return (encoder_input, decoder_input, decoder_output)

    def __data_generation(self):
        encoder_input, decoder_input, decoder_output = self.__process_batch_data()
        self.internal_index = self.internal_index + self.batch_size

        return [encoder_input, decoder_input], decoder_output
# pylint: disable=E0401,E0611

import sys
import os
sys.path.append("..")
script_dir = os.path.dirname(__file__)

from models.InputData import InputData
from models.OutputData import OutputData
from models.DataResult import DataResult
from Config import BATCH_SIZE

class DataService:
    def __init__(self, input_batch_file, output_batch_file):
        self.input_file = script_dir + "/../data/input.txt"
        self.output_file = script_dir + "/../data/output.txt"
        self.input_batch_file = script_dir + '/../temp/' + input_batch_file
        self.output_batch_file = script_dir + '/../temp/' + output_batch_file

        with open(self.input_batch_file, 'w') as temp_input_batch:
            temp_input_batch.write('0\n')

        with open(self.output_batch_file, 'w') as temp_output_batch:
            temp_output_batch.write('0\n')

    def __process_input(self, input_data):
        with open(self.input_batch_file, 'a') as temp_input_batch:
            with open(self.input_file, 'r', encoding="utf8") as f:
                # Loop through each of the lines in the file
                temp_index = 0
                temp_batch_index = 0
                batch_num = 0
                while(True):
                    line = f.readline()
                    
                    if not line:
                        break
                    
                    line_length = len(line)

                    if (temp_batch_index == (BATCH_SIZE - 1) or (temp_index + 1) == input_data.num_lines):
                        position = str(f.tell())
                        if batch_num > 0:
                            position = '\n' + position

                        temp_input_batch.write(position)
                        
                        temp_batch_index = 0
                        batch_num += 1

                    else:
                        temp_batch_index = temp_batch_index + 1

                    if (input_data.max_len < line_length):
                        input_data.max_len = line_length

                    # loop through each char in the line
                    for char in line:
                        # If the char doesnt already exist then add it to the input chars collection
                        if char not in input_data.chars:
                            input_data.chars.add(char)
                    
                    temp_index += 1
            
        # Sort the input chars
        input_data.chars = sorted(list(input_data.chars))

        # Count the number of input chars
        input_data.num_tokens = len(input_data.chars)

        return input_data

    def __process_output(self, output_data):                
        with open(self.output_batch_file, 'a') as temp_output_batch:
            with open(self.output_file, 'r', encoding="utf8") as f:
                # Loop through each of the lines in the file
                temp_batch_index = 0                
                temp_index = 0
                batch_num = 0
                while(True):
                    line = f.readline()
                    
                    if not line:
                        break

                    if (temp_batch_index == (BATCH_SIZE - 1) or (temp_index + 1) == output_data.num_lines):
                        position = str(f.tell())
                        if batch_num > 0:
                            position = '\n' + position

                        temp_output_batch.write(position)
                        
                        temp_batch_index = 0
                        batch_num += 1
                    else:
                        temp_batch_index = temp_batch_index + 1

                    output_line = "\t" + line + "\n"

                    line_length = len(output_line)                
                    if (output_data.max_len < line_length):
                        output_data.max_len = line_length

                    # loop through each char in the line
                    for char in output_line:
                        # If the char doesnt already exist then add it to the output chars collection
                        if char not in output_data.chars:
                            output_data.chars.add(char)

                    temp_index += 1
            
        # Sort the output chars
        output_data.chars = sorted(list(output_data.chars))

        # Count the number of output chars
        output_data.num_tokens = len(output_data.chars)

        return output_data

    def __pre_process_input(self):
        input_data = InputData()
        with open(self.input_file, 'r', encoding="utf8") as f:
            for i, line in enumerate(f):
                input_data.num_lines = input_data.num_lines + 1
        
        return input_data
    
    def __pre_process_output(self):
        output_data = OutputData()
        with open(self.output_file, 'r', encoding="utf8") as f:
            for i, line in enumerate(f):
                output_data.num_lines = output_data.num_lines + 1
        
        return output_data

    def process(self):
        print('    STARTING: pre_processing_input')
        input_data = self.__pre_process_input()
        print('    END: pre_processing_input')
        print('')

        print('    STARTING: pre_processing_output')
        output_data = self.__pre_process_output()
        print('    END: pre_processing_output')
        print('')

        # Process the inputs
        print('    STARTING: processing_input')
        input_data = self.__process_input(input_data)
        print('    END: processing_input')
        print('')

        # Process the outputs
        print('    STARTING: processing_output')
        output_data = self.__process_output(output_data)
        print('    END: processing_output')
        
        return DataResult(input_data, output_data)
        


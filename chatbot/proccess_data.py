# pylint: disable=E0401,E0611

import os
from os.path import exists
script_dir = os.path.dirname(__file__)

from helpers.DataService import DataService
temp_dir = script_dir + './temp/'
input_batch_file = '_input_batch_file.txt'
output_batch_file = '_output_batch_file.txt'

if exists(temp_dir + input_batch_file):
    os.remove(temp_dir + input_batch_file)

if exists(temp_dir + output_batch_file):
    os.remove(temp_dir + output_batch_file)

data_service = DataService(input_batch_file, output_batch_file)

print('STARTING: processing')
data_result = data_service.process()
print('END: processing')
print('')

data_file =  temp_dir + 'processed_data.json'

if exists(data_file):
    os.remove( data_file)

with open(data_file, 'w') as output:
    data = data_result.toJSON()
    output.write(data)
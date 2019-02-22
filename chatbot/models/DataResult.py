import json

from .InputData import InputData
from .OutputData import OutputData

class DataResult:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def loadJSON(self, json_data):
        temp_input_data = InputData()
        temp_output_data = OutputData()

        temp_input_data.loadJSON(json_data['input_data'])
        temp_output_data.loadJSON(json_data['output_data'])

        self.input_data = temp_input_data
        self.output_data = temp_output_data

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
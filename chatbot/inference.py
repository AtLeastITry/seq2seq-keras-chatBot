# pylint: disable=E0401,E0611
import json
import os
script_dir = os.path.dirname(__file__)

from sys import stdin
from helpers.InferenceHelper import InferenceHelper
from models.DataResult import DataResult


print ("initializing")
print('STARTING: loading_data')
data_result = DataResult(None, None)

with open(script_dir + './temp/processed_data.json', 'r') as output:
    json_data = json.load(output)
    data_result.loadJSON(json_data)
print('END: loading_data')
print('')

helper = InferenceHelper(data_result)


print("say something")
end = False
while end == False:
    userinput = str(stdin.readline())
    if (userinput == 'e'):
        end = True
    else:
        prediction = helper.predict(userinput)
        print(prediction)

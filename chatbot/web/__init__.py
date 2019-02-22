from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
wsgi_app = app.wsgi_app #Registering with IIS


# pylint: disable=E0401,E0611,E1129
import numpy as np
import flask
import os
import json
import sys
sys.path.append("..")
script_dir = os.path.dirname(__file__)

from helpers.InferenceHelper import InferenceHelper
from helpers.DataService import DataService
from tensorflow.keras.models import load_model
from tensorflow import Graph, Session
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from models.Encoder import Encoder
from models.Decoder import Decoder
from models.DataResult import DataResult


def load():
    global session
    global graph
    global model
    global data_result

    data_result = DataResult(None, None)

    with open(script_dir + '/../temp/processed_data.json', 'r') as output:
        json_data = json.load(output)
        data_result.loadJSON(json_data)

    graph = Graph()
    with graph.as_default():
        session = Session(graph = graph)
        with session.as_default():
            temp_encoder = Encoder(data_result.input_data)
            temp_decoder = Decoder(data_result.output_data, temp_encoder)
            temp_model = Model([temp_encoder.inputs, temp_decoder.inputs], temp_decoder.outputs)
            temp_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
            temp_model.load_weights(os.path.dirname(__file__) + '/../model_weights.h5')
            
            model = temp_model
	

@app.route("/chat", methods=["POST"])
def predict():
    if session == None:
        load()
        
    K.set_session(session)
    with graph.as_default():
        reply = InferenceHelper(data_result, model).predict(flask.request.json['message'])
    
    return flask.jsonify(reply)

load()
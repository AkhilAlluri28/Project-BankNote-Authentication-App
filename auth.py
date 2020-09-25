# -*- coding: utf-8 -*-
"""
Created on May 22 18:11:00 2020

@author Akhil Reddy Alluri
"""
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier_model = pickle.load(pickle_in)

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    
    y_predictions = classifier_model.predict([[variance, skewness, curtosis, entropy]])
    print(y_predictions)
    return str(y_predictions)


@app.route('/predict_file', methods=["POST"])
def predict_note_authentication_file():
    
    
    """Predict File
    Predictions are made on provided file data (csv only)
    ---
    parameters:
      - name: uploadFile
        in: formData
        type: file
        required: true
          
    responses:
        200:
            description: The output values
    
    """
    
    df = pd.read_csv(request.files.get("uploadFile"));
    y_predictions = classifier_model.predict(df)
    return str(list(y_predictions))

# __name__ set to __main__ only if file is ran directly by python.
# if file is ran indirectly i.e by import then __name is set to __fileName__
if __name__ == '__main__':
    app.run()

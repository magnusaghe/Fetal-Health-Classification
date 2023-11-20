from flask import Flask, render_template, request
import joblib
from joblib import dump, load
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

filename = 'file_fetalhealth.pkl'
model = joblib.load('file_fetalhealth.pkl')
dump(model, 'filename.joblib')      # save the model
model = load('filename.joblib')     # load the model

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    Prolongued_Decelerations = request.form['prolongued_decelerations']
    Percentage_of_Time_with_Abnormal_Long_Term_Variability = request.form['percentage_of_time_with_abnormal_long_term_variability']
    Accelerations = request.form['accelerations']
    Histogram_Variance = request.form['histogram_variance']
    Uterine_Contractions = request.form['uterine_contractions']
    pred = model.predict(np.array([[float(Prolongued_Decelerations), float(Percentage_of_Time_with_Abnormal_Long_Term_Variability), float(Accelerations), float(Histogram_Variance), float(Uterine_Contractions) ]])) #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(host="0.0.0.0", port= port)

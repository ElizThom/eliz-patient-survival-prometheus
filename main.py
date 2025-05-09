# Load your trained model
import gradio
import joblib
from fastapi import FastAPI, Request, Response
import prometheus_client as prom

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import random
import numpy as np
import pandas as pd

app = FastAPI()

save_file_name = "xgboost-model.pkl"
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
                        'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
target = 'DEATH_EVENT'

xgb_clf = joblib.load(save_file_name)

acc_metric = prom.Gauge('patient_survival_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('patient_survival_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('patient_survival_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('patient_survival_recall_score', 'Recall score for few random 100 test samples')
 
test_data = []

def load_test_data():
    global test_data
    # Load test data from a file or database
    data = pd.read_csv("./train/heart_failure_clinical_records_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(                   # divide into train and test set
        data[features],
        data[target],
        test_size=0.2,
        random_state=42,
    )
    test_data = X_test.copy()
    test_data['target'] = y_test.values

def update_metrics():
    global test_data
    # Performance on test set
    size = random.randint(30, 40)
    test = test_data.sample(size, random_state = random.randint(0, 1e6))       # sample few 100 rows randomly
    y_pred = xgb_clf.predict(test.iloc[:, :-1])                           # prediction
    acc = round(accuracy_score(test['target'], y_pred) , 3)                   # accuracy score
    f1 = round(f1_score(test['target'], y_pred), 3)                           # F1 score
    precision = round(precision_score(test['target'], y_pred),3)             # Precision score
    recall = round(recall_score(test['target'], y_pred),3)                   # Recall score
    
    acc_metric.set(acc)
    f1_metric.set(f1)
    precision_metric.set(precision)
    recall_metric.set(recall)
# Function for prediction

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                        platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    # Prepare the input as a 2D array for prediction
    features = [[
        int(age),
        int(anaemia),
        float(creatinine_phosphokinase),
        int(diabetes),
        float(ejection_fraction),
        int(high_blood_pressure),
        float(platelets),
        float(serum_creatinine),
        float(serum_sodium),
        int(sex),
        int(smoking),
        int(time)
    ]]
    pred = xgb_clf.predict(features)[0]
    return f"{pred}"

"""For categorical user input, user [Radio](https://www.gradio.app/docs/radio) button component.

For numerical user input, user [Slider](https://www.gradio.app/docs/slider) component.
"""
load_test_data()

# Inputs from user
age = gradio.Slider(minimum = 30, maximum = 110, value = 60, label = "Age")
anaemia = gradio.Radio(choices = [0, 1], label = "Anaemia", value=0)
creatinine_phosphokinase = gradio.Slider(minimum = 10, maximum = 10000, value = 1000, label = "Creatinine Phosphokinase")
diabetes = gradio.Radio(choices = [0, 1], label = "Diabetes", value=0)
ejection_fraction = gradio.Slider(minimum = 0, maximum = 100, value = 20, label = "Ejection Fraction")
high_blood_pressure = gradio.Radio(choices = [0,1], label = "High Blood Pressure", value=0)
platelets = gradio.Slider(minimum = 10000, maximum = 1000000, value = 300000, label = "Platelets")
serum_creatinine = gradio.Slider(minimum = 0.5, maximum = 10.0, value = 1.0, label = "Serum Creatinine")
serum_sodium = gradio.Slider(minimum = 100, maximum = 200, value = 100, label = "Serum Sodium")
sex = gradio.Radio(choices = [0,1], label = "Sex", value=0)
smoking = gradio.Radio(choices = [0,1], label = "Smoking", value=0)
time = gradio.Slider(minimum = 0, maximum = 500, value = 100, label = "Time")
inputs = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]

# Output response
outputs = gradio.Textbox(label = "Death Event")

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = inputs,
                         outputs = outputs,
                         title = title,
                         description = description,
                         allow_flagging='never')

 #Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 

# iface.launch(server_name="0.0.0.0", server_port=8001)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
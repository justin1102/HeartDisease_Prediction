import email
from pyexpat import model
from tkinter import X
from turtle import xcor
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import smtplib
import ssl
from email.message import EmailMessage
import os

app = Flask(__name__)

def get_algorithm(x):
    x=int(x)
    if x == 0:
        model = pickle.load(open('../HeartDisease/algorithm/model_rf.pkl', 'rb'))
    elif x == 1:
        model = pickle.load(open('../HeartDisease/algorithm/model_gb.pkl', 'rb'))
    elif x == 2:
        model = pickle.load(open('../HeartDisease/algorithm/model_knn.pkl', 'rb'))
    elif x == 3:
        model = pickle.load(open('../HeartDisease/algorithm/model_svm.pkl', 'rb'))
    elif x == 4:
        model = pickle.load(open('../HeartDisease/algorithm/model_dt.pkl', 'rb'))
    elif x == 5:
        model = pickle.load(open('../HeartDisease/algorithm/model_log.pkl', 'rb'))
    elif x == 6:
        model = pickle.load(open('../HeartDisease/algorithm/model_gnb.pkl', 'rb'))
    return model


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# @app.route("/login_page", methods=['GET', 'POST'])
# def login():
#     return render_template('login.html')

@app.route("/login_page", methods=['GET', 'POST'])
def login():
    return render_template('login.html')



@app.route("/about_us_page", methods=['GET', 'POST'])
def about_us():
    return render_template('about_us.html')

@app.route("/signup_page", methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

@app.route("/predict_page", methods=['GET', 'POST'])
def predict_page():
    # redirect(url_for('success'))
    return render_template('main.html')


@app.route("/graph_page", methods=['GET', 'POST'])
def graph_page():
    return render_template('graph.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    age = request.form.get("age")
    sex = request.form.get("sex")
    cp = request.form.get("cp")
    trestbps = request.form.get("trestbps")
    chol = request.form.get("chol")
    fbs = request.form.get("fbs")
    restecg = request.form.get("restecg")
    thalach = request.form.get("thalach")
    oldpeak = request.form.get("oldpeak")
    exang = request.form.get("exang")
    slope = request.form.get("slope")
    ca = request.form.get("ca")
    thal = request.form.get("thal")
    algorithm = request.form.get("algorithm")
    data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    data = list(map(float, data))
    result = get_algorithm(algorithm)
    scaler2 = StandardScaler()
    ##CHANGE THE INPUT TO NUMPY ARRAY
    input_data_as_numpy_array = np.asarray(data)
    #RESHAPE THE NUMPY ARRAY BECAUSE WE NEED TO PREDICT THE TARGET
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler2.fit_transform(input_data_reshaped)
    prediction = result.predict(input_data_reshaped)

    if prediction[0] == 0:
        return render_template('main.html', prediction_text='does not heart disease')
    else:
        return render_template('main.html', prediction_text=' heart disease')


if __name__ == "__main__":
    app.run(debug=True)

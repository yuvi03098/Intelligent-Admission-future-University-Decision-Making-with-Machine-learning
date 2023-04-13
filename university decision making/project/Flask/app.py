import numpy as np
from flask import Flask, request, jsonify, render_template 
import pickle
app = Flask(__name__)
from tensorflow.keras.models import load_model
model=load_model('C:/university decision making/project/training/model.h5')
@app.route('/')
def home():
    return render_template('C:/university decision making/project/Flask/Demo2.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    for rendering results on HTML GUI
    '''
    #min max scaling 
    min1=[290.0, 92.0, 1.0, 1.0, 1.0, 6.8, 0.0]
    max1=[340.0, 120.0, 5.0, 5.0, 5.0, 9.92, 1.0]
    k=[float(x) for x in request.form.values()]
    p=[]
    for i in range(7):
        l=(k[i]-min1[i])/(max1[i]-min[i])
        p.appand(l)
        prediction = model.predict([p])
        print(prediction)
        output=prediction[0]
        if(output==False):
            return render_template('noChance.html',prediction_text='you dont have a chance of gettin')
        else:
            return render_template('Chance.html',prediction_text='you have a chance of getting admit')
    if __name__ == "__main__":
        app.run(debug=True)
            
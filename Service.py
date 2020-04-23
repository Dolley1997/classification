# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:02:11 2020

@author: Dolley
"""
import logging

import pickle
from flask import Flask, render_template,request,jsonify
app = Flask(__name__)

from sklearn.preprocessing import LabelEncoder
@app.route('/')
def home():
    return render_template('home.html')
##predict AsseType
@app.route('/predict',methods=['POST'])
def predict():

    targetEncoder=LabelEncoder()
    
    #pickle file of dependent features
    file = open("labelEncoder.pkl",'rb')
    targetEncoder.classes_ = pickle.load(file)
    
    #pickle file of independent and text features together
    with open('textFeatureNClassfier.pkl', 'rb') as f:
        vect, classifier = pickle.load(f)
        
    if request.method == 'POST':
        try:
            message = request.form['message']
            #newVal = ['K4000 Mobile Workstation Cart']
            message=[message]
            print('message for AssetType',message)
            logger.info('Input from AGServer for AssetType api')
            logger.info(message)
            newVal = vect.transform(message).toarray()
            y_pred=classifier.predict(newVal)
            y_pred = targetEncoder.inverse_transform(y_pred)
            y_pred = y_pred.astype('U')
            temp={"prediction":y_pred.tolist(),
                  "stausCode":'200'}
            print('temp',temp)
            logger.info('Temp Json of AssetType')
            logger.info(temp)
        except Exception as e:
            print('Exception in AssetType Classification',e)
            logger.error('Exception in AssetType Classification')
            logger.error(e)
            temp={"prediction":'Unavailable',
                  "stausCode":'500'}
    return jsonify(temp)

##predict Parent AssetType
@app.route('/predictParentAssetType',methods=['POST'])
def predictParentAssetType():

    targetEncoder=LabelEncoder()
    #pickle file of dependent features for parentAssetType
    file = open("parentLabelEncoder.pkl",'rb')
    targetEncoder.classes_ = pickle.load(file)
    
    
    #pickle file of independent and text features together
    with open('parentClassifier.pkl', 'rb') as f:
        vect, classifier = pickle.load(f)
        
        
        
    if request.method == 'POST':
        try:
            message = request.form['message']
            #newVal = ['10031 chatillon dfi100 digital force gauge 100lb load cell']
            message=[message]
            logger.info('Input from AGServer')
            logger.info(message)
            newVal = vect.transform(message).toarray()
            y_pred=classifier.predict(newVal)
            y_pred = targetEncoder.inverse_transform(y_pred)
            y_pred = y_pred.astype('U')
            #temp=createResponseData(message,y_pred,'200')
            temp={"prediction":y_pred.tolist(),
                  "stausCode":'200'}
            logger.info('Temp of ParentAssetType')
            logger.info(temp)
        except Exception as e:
            logger.error('Exception in ParentAssetType Classification')
            logger.error(e)
            temp={"prediction":'Unavailable',
                  "stausCode":'500'}
    return jsonify(temp)


'''
def createResponseData(inputData,prediction,statusCode):
    temp={}
    temp['inputData']=inputData.tolist(),
    temp['prediction']=prediction.tolist(),
    temp['statusCode']=statusCode
    json_data=json.dumps(temp)
    print(json_data)
    return json_data
'''

if __name__ == '__main__':
    logger = logging.getLogger('werkzeug')# WSGI logger
    handler = logging.FileHandler('PythonService.log')
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
    #handler = RotatingFileHandler('service.log', maxBytes=10000000, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    app.run()
'''

if __name__ == '__main__':
    logger = logging.getLogger('werkzeug')# WSGI logger
    handler = logging.FileHandler('PythonService.log')
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
    #handler = RotatingFileHandler('service.log', maxBytes=10000000, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    app.run(host='0.0.0.0',port=8080)
'''
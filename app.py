import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    tenth = float(request.args.get('tenth'))
    twelth=float(request.args.get('twelth'))
    btech=float(request.args.get('btech'))
    sevsem=float(request.args.get('7sem'))
    sixsem=float(request.args.get('6sem'))
    fivesem=float(request.args.get('5sem'))
    final=float(request.args.get('final'))
    medium=float(request.args.get('medium'))
    model1=float(request.args.get('model1'))

    if model1==0:
        model=pickle.load(open('project6_decision_model.pkl','rb'))
        accr="58.33%"
    elif model1==1:
        model=pickle.load(open('project6_svm.pkl','rb'))
        accr="75.00%"
    elif model1==2:
        model=pickle.load(open('project6_random_forest.pkl','rb'))
        accr="83.33%"
    elif model1==3:
        model=pickle.load(open('project6_knn.pkl','rb'))
        accr="70.83%"
    elif model1==4:
        model=pickle.load(open('project6_naive.pkl','rb'))
        accr="91.67%"
      

    dataset= pd.read_excel('DATASET education.xlsx')
    X = dataset.iloc[:, 0:8].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[tenth,twelth,btech,sevsem,sixsem,fivesem,final,medium]]))
    if prediction==0:
         message="Student not Placed"
    else:
        message="Student will be placed"
    
        
    return render_template('index.html', prediction_text='Model  has predicted : {}'.format(message), accuracy_text='Accuracy of Model :{}'.format(accr))


if __name__ == "__main__":
    app.run(debug=True)

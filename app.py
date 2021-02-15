from flask import Flask, request, make_response, redirect, render_template
from flask import Flask, render_template, request
from flask import jsonify
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
       no_year = int(request.form['no_year'])
       Present_Price=int(request.form['Present_Price'])
       owner=request.form['owner']
       Km_driven=int(request.form['Km_driven'])
       Km_driven2=np.log(Km_driven)
       
       Seller_Type_Individual=request.form['Seller_Type_Individual']
       
       if(Seller_Type_Individual=='Individual'):
         Seller_Type_Individual=1
       else:
         Seller_Type_Individual=0
       
       prediction=model.predict([[Present_Price, Km_driven,Km_driven2, no_year, Seller_Type_Individual,owner]])
       output=round(prediction[0],2)
       if output<0:
           return render_template('index.html',prediction_texts="Sorry you cannot sell this Bike")
       else:
            return render_template('index.html',prediction_text="You Can Sell The Bike at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)



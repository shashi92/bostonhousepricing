import pickle
from flask import Flask,request,app, jsonify,url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# load model and scalar
regmodel=pickle.load(open('linear_reg.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    row = np.array(list(data.values())).reshape(1, -1)
    row_scaled = scalar.fit_transform(row)
    output=regmodel.predict(row_scaled)
    print(f'prdicted: {output[0]}')
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data=[float(i) for i in request.form.values()]
    print(data)
    row = np.array(data).reshape(1, -1)
    final_input = scalar.fit_transform(row)
    final_out=regmodel.predict(final_input)[0]
    return render_template("home.html", text="Predicted home price is : {}".format(final_out))


if __name__=="__main__":
    app.run(debug=True)


import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if(request.method=="GET"):

          data ="hello world"
          return jsonify({'data': data})

@app.route('/predict/')
def salary_predict():
    model = pickle.load(open('model.pkl','rb'))
    years =  request.args.get('years')

    test_df=pd.DataFrame({'years':[years]})

    pred_salary = model.predict(test_df)
    return jsonify({'salary': str(pred_salary)})

if __name__=="__main__":
    app.run(debug=True)
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET','POST']) 
def fun():
    if request.method=='POST':
        data = request.json
        arr = np.array([int(data['Glucose']),int(data['BloodPressure']),int(data['SkinThickness']),int(data['Insulin']),float(data['BMI']),int(data['Age'])])
        filename='diab_pred_model'
        load_model= pickle.load(open(filename,'rb'))
        pred= load_model.predict([arr])
        resp={"Prediction":""}
        if(pred[0]==0):
            resp['Prediction']="No"
        else:
            resp['Prediction']="Yes"
        return jsonify(resp)
    else:
        return "Connected"

if __name__=='__main__':
    app.run(debug=True)







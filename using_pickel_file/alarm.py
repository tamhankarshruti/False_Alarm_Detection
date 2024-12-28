
from flask import Flask,request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/train')
def train():
    data = pd.read_excel('Historical Alarm Cases_copy.xlsx')
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']

    logm = LogisticRegression()
    logm.fit(x, y)
    joblib.dump(logm, 'train.pkl')
    return "Model trained successfully"


@app.route('/test', methods=['POST'])
def test():

    pkl_file = joblib.load('train.pkl')
    test_data = request.get_json()
    f1 = test_data['Ambient Temperature']
    f2 = test_data['Calibration']
    f3 = test_data['Unwanted substance deposition']
    f4 = test_data['Humidity']
    f5 = test_data['H2S Content']
    f6 = test_data['detected by']
    my_test_data = [f1, f2, f3, f4, f5, f6]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 6)
    df_test = pd.DataFrame(test_array,columns=['Ambient Temperature( deg C)', 'Calibration(days)', 'Unwanted substance deposition(0/1)', 'Humidity(%)','H2S Content(ppm)', 'detected by(% of sensors)'])
    y_pred = pkl_file.predict(df_test)

    if y_pred == 1:
        return "False Alarm, 1"
    else:
        return "True Alarm, 0"



app.run(port=5001)

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from flask import *
import json
from jinja2 import Environment, BaseLoader
import os

portnumber = int(os.environ.get('PORT', 5000))

app = Flask(__name__)


@app.route('/')
def quality():
    data = request.args.get('my_dict')
    stringdict = format(json.dumps(data))
    my_dataDict = json.loads(stringdict)
    myData = json.loads(my_dataDict)

    coffee = pd.read_csv("merged_data_cleaned_new.csv")
    coffee_data = coffee[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "CleanCup", "Sweetness", "CupperPoints",
                          "Moisture", "Quakers", "CategoryOneDefects", "CategoryTwoDefects", "altitude_mean_meters", "Total.Cup.Points"]]

    x = coffee_data["altitude_mean_meters"].mean()
    coffee_data_copy = coffee_data.copy()
    coffee_data_copy["altitude_mean_meters"].fillna(x, inplace=True)
    X = coffee_data.drop(columns='Total.Cup.Points', axis=1)
    Y = coffee_data['Total.Cup.Points']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    model = XGBRegressor()
    model.fit(X_train, Y_train)
    test_data_prediction = model.predict(X_test)
    a = {
        "Aroma": float(myData['aroma']),
        "Flavor": float(myData['Flavor']),
        "Aftertaste": float(myData['Aftertaste']),
        "Acidity": float(myData['Acidity']),
        "Body": float(myData['Body']),
        "Balance": float(myData['Balance']),
        "Uniformity": float(myData['Uniformity']),
        "CleanCup": float(myData['CleanCup']),
        "Sweetness": float(myData['Sweetness']),
        "CupperPoints": float(myData['CupperPoints']),
        "Moisture": float(myData['Moisture']),
        "Quakers": float(myData['Quakers']),
        "CategoryOneDefects": float(myData['CategoryOneDefects']),
        "CategoryTwoDefects": float(myData['CategoryTwoDefects']),
        "altitude_mean_meters": float(myData['altitude_mean_meters'])
    }
    # A = np.array([[a['Aroma'], 8.83, 8.75, 8.5, 8.42, 10.0, 10.0,10.0, 8.75, 0.12, 0.0, 0, 0, 2075.0]])
    A = np.array([[a['Aroma'], a["Flavor"], a['Aftertaste'], a["Acidity"], a["Body"], a["Balance"], a["Uniformity"], a["CleanCup"], a["Sweetness"],
                 a["CupperPoints"], a["Moisture"], a["Quakers"], a["CategoryOneDefects"], a["CategoryTwoDefects"], a["altitude_mean_meters"]]])
    single_pred = model.predict(A)
    print(single_pred)
    return render_template('/index.html', quality=single_pred[0])
    # output = {'quality': "{}".format(single_pred[0])}
    # return jsonify(output)


if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0', port=portnumber)
# 127.0.0.1:5501

# http://localhost:5000?my_dict={"aroma":8.67,"Flavor":8.83}
# http://localhost:5000?my_dict={"aroma":8.67,"Flavor":8.83,"Aftertaste":8.6,"Acidity":8.75,"Body":8.5,"Balance":8.42,"Uniformity":10.0,"CleanCup":10.0,"Sweetness":10.0,"CupperPoints":8.75,"Moisture":0.12,"Quakers":0.0,"CategoryOneDefects":0.0,"CategoryTwoDefects":0,"altitude_mean_meters":2075.0}

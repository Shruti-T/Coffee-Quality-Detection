from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from flask import Flask, request
import json

app = Flask(__name__)
# http://localhost:5000?my_dict={"aroma":8.67,"Flavor":8.83}
# http://localhost:5000?my_dict={"aroma":8.67,"Flavor":8.83,"Acidity":8.75,"Body":8.5,"Balance":8.42,"Uniformity":10.0,"Clean.Cup":10.0,"Sweetness":10.0,"Cupper.Points":8.75,"Moisture":0.12,"Quakers":0.0,"Category.One.Defects":0.0,"Category.Two.Defects":0,"altitude_mean_meters":2075.0}
# http://127.0.0.1:5000/?aroma=8.67,Flavor=8.83,Acidity=8.75,Body=8.5,Balance=8.42,Uniformity=10.0,Clean.Cup=10.0,Sweetness=10.0,Cupper.Points=8.75,Moisture=0.12,Quakers=0.0,Category.One.Defects=0.0,Category.Two.Defects=0,altitude_mean_meters=2075.0


@app.route('/')
def quality():
    data = request.args.get('my_dict')
    stringdict = format(json.dumps(data))
    my_dataDict = json.loads(stringdict)
    myData = json.loads(my_dataDict)

    coffee = pd.read_csv("merged_data_cleaned.csv")
    coffee_data = coffee[["Aroma", "Flavor", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points",
                          "Moisture", "Quakers", "Category.One.Defects", "Category.Two.Defects", "altitude_mean_meters", "Total.Cup.Points"]]

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
        "Acidity": float(myData['Acidity']),
        "Body": float(myData['Body']),
        "Balance": float(myData['Balance']),
        "Uniformity": float(myData['Uniformity']),
        "Clean.Cup": float(myData['Clean.Cup']),
        "Sweetness": float(myData['Sweetness']),
        "Cupper.Points": float(myData['Cupper.Points']),
        "Moisture": float(myData['Moisture']),
        "Quakers": float(myData['Quakers']),
        "Category.One.Defects": float(myData['Category.One.Defects']),
        "Category.Two.Defects": float(myData['Category.Two.Defects']),
        "altitude_mean_meters": float(myData['altitude_mean_meters'])
    }
    # print(a)
    # A = np.array([[a['Aroma'], 8.83, 8.75, 8.5, 8.42, 10.0, 10.0,10.0, 8.75, 0.12, 0.0, 0, 0, 2075.0]])
    A = np.array([[a['Aroma'], a["Flavor"], a["Acidity"], a["Body"], a["Balance"], a["Uniformity"], a["Clean.Cup"], a["Sweetness"],
                 a["Cupper.Points"], a["Moisture"], a["Quakers"], a["Category.One.Defects"], a["Category.Two.Defects"], a["altitude_mean_meters"]]])
    single_pred = model.predict(A)
    print(single_pred)

    return "this is the coffee quality you are getting from python route: {}".format(single_pred[0])


if __name__ == '__main__':
    app.run()

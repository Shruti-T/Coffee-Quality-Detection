from sklearn.metrics import r2_score
from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
# import json


app = Flask(__name__)
a = {
    "Aroma": 8.67,
    "Flavor": 8.83,
    "Acidity": 8.75,
    "Body": 8.5,
    "Balance": 8.42,
    "Uniformity": 10.0,
    "Clean.Cup": 10.0,
    "Sweetness": 10.0,
    "Cupper.Points": 8.75,
    "Moisture": 0.12,
    "Quakers": 0.0,
    "Category.One.Defects": 0,
    "Category.Two.Defects": 0,
    "altitude_mean_meters": 2075.0}


def hello_world():
    return 'Hello, World!'


@app.route('/<Aro>/<Flavor>')
def quality(Aro, Flavor):
    # x = json.dumps(x)
    # xx = json.loads(x)
    # print(xx)
    print(Aro, Flavor)
    coffee = pd.read_csv("merged_data_cleaned.csv")
    coffee_data = coffee[["Aroma", "Flavor", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points",
                          "Moisture", "Quakers", "Category.One.Defects", "Category.Two.Defects", "altitude_mean_meters", "Total.Cup.Points"]]

    x = coffee_data["altitude_mean_meters"].mean()
    coffee_data["altitude_mean_meters"].fillna(x, inplace=True)
    X = coffee_data.drop(columns='Total.Cup.Points', axis=1)
    Y = coffee_data['Total.Cup.Points']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    test_data_prediction = model.predict(X_test)
    mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
    score = r2_score(Y_test, test_data_prediction)
    # A = np.array([[8.67, 8.83, 8.75, 8.5, 8.42, 10.0, 10.0,10.0, 8.75, 0.12, 0.0, 0, 0, 2075.0]])
    aro = 8.44
    A = [[aro, a["Flavor"], a["Acidity"], a["Body"], a["Balance"], a["Uniformity"], a["Clean.Cup"], a["Sweetness"],
                 a["Cupper.Points"], a["Moisture"], a["Quakers"], a["Category.One.Defects"], a["Category.Two.Defects"], a["altitude_mean_meters"]]]
    # A = np.array([[Aroma, Flavor, Acidity, Body, Balance, Uniformity, CleanCup, Sweetness, CupperPoints,
    #         Moisture, Quakers, CategoryOneDefects, CategoryTwoDefects, altitude_mean_meters]])
    single_pred = model.predict(A)
    print(single_pred)
    return 'hello'


if __name__ == '__main__':
    app.run()

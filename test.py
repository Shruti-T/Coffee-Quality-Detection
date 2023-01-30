from sklearn.metrics import r2_score
from flask import Flask

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


app = Flask(__name__)


# @app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/')
def quality():
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
    A = np.array([[8.67, 8.83, 8.75, 8.5, 8.42, 10.0, 10.0,
                   10.0, 8.75, 0.12, 0.0, 0, 0, 2075.0]])
    single_pred = model.predict(A)
    print(single_pred)
    return 'hello'


if __name__ == '__main__':
    app.run()

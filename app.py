from flask import Flask, render_template, request
# from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
# cors = CORS(app)
car = pd.read_csv("cleaned_car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route("/")
def index():
    # Getting each data only single time (unique()) with the help of pandas.
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = car["fuel_type"].unique()
    companies.insert(0, "Select Company")
    return render_template("index.html", companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route("/predict", methods=["POST"])
# @cross_origin()
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_model")
    year = request.form.get("year")
    fuel_type = request.form.get("fuel_type")
    kilo_driven = int(request.form.get("kilo_driven"))
    # print(company, car_model, year, fuel_type, kilo_driven)
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kilo_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    # print(prediction[0])
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)

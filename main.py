from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def build_plot(file_path, x, y):
    data = pd.read_csv(Path(__file__).parent / file_path)
    plt.scatter(data[x], data[y], c='blue', marker='x')
    plt.xlabel(x)
    plt.ylabel(y)
    return plt

# build_plot('resources/CO2_emission.csv', 'Engine_Size', 'CO2_Emissions').show()

def build_regression_plot(x_data, y_data, y_pred, xlabel, ylabel) -> None:
    plt.scatter(x_data, y_data, color="green")

    plt.plot(x_data, y_pred, color="blue", linewidth=4, label='Regression Line')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

def extract_texts_from_file(file_path):
    data_path = Path(__file__).parent / file_path
    
    dataframe = pd.read_csv(data_path, delimiter=',', quotechar='"')
    # selected_records = dataframe.values.tolist()
    return dataframe 

data = extract_texts_from_file('resources/heart.csv')
print(data[:50])
def CO2_emissions(data):
    # en_size = 'Engine_Size'
    # co2_emiss = 'CO2_Emissions'
    
    en_size = data['Engine_Size']
    co2_emiss = data['CO2_Emissions']
    
    # en_train, en_test, co2_train, co2_test = train_test_split(en_size, co2_emiss, test_size=0.1, shuffle=True)
    
    model = LinearRegression(fit_intercept=True)
    model.fit(en_size.values.reshape(-1, 1), co2_emiss.values.reshape(-1, 1))
    co2_pred = model.predict(en_size.values.reshape(-1, 1))
    
    build_regression_plot(en_size, co2_emiss, co2_pred, 'Engine_Size', 'CO2_Emissions')
    
    # print(f"R2: {r2_score(en_test, co2_pred)}")
    # print(f"MSE: {mean_squared_error(en_test, co2_pred)}")
    
    return model.coef_.tolist() 

data = extract_texts_from_file('resources/CO2_emission.csv')
theta = CO2_emissions(data[::5])
def ice_cream_selling(data):
    sells = data["Ice Cream Sales (units)"].values
    temperature = data["Temperature (°C)"].values
    
    poly = PolynomialFeatures(degree=2)
    temperature_poly = poly.fit_transform(sells.reshape(-1, 1))
    
    
    model = LinearRegression()
    model.fit(temperature_poly, sells)
    sells_pred = model.predict(temperature_poly)
    
    build_regression_plot(temperature, sells,  sells_pred, 'Ice Cream Sales (units)', 'Temperature (°C)')
        
    return model.coef_.tolist() 

# build_plot('resources/ice_cream_selling_data.csv', 'Temperature (°C)', 'Ice Cream Sales (units)').show()
data = extract_texts_from_file('resources/ice_cream_selling_data.csv')
ice_cream_selling(data)
def power_consumption(data):
    weather = data[["Temperature", "Humidity", "WindSpeed"]]
    
    consumption = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
    
    weather_train, weather_test, consumption_train, consumption_test = train_test_split(weather, consumption, test_size=0.2, shuffle=True)
    
    model = LinearRegression()
    model.fit(weather_train, consumption_train)
    consumption_pred = model.predict(weather_test)
    
    print(f"R2: {r2_score(consumption_test, consumption_pred)}")
    print(f"MSE: {mean_squared_error(consumption_test, consumption_pred)}")
    
    # plt.scatter(temperature, consumption1, color='blue')
    # plt.plot(temperature, consumption1, 'g.', linewidth=0.001, markersize=12)
    # plt.xlabel('Temperature (°C)')
    # plt.ylabel('Power Consumption')
    # plt.title('Power Consumption vs Temperature')
    # plt.show()
    
    return model.coef_.tolist() 

# build_plot('resources/powerconsumption.csv', "Temperature", "Humidity", "WindSpeed", "PowerConsumption_Zone1", 
#                "PowerConsumption_Zone2", "PowerConsumption_Zone3").show()
data = extract_texts_from_file('resources/powerconsumption.csv')
power_consumption(data)
def heart_classification(data):
    X = data.iloc[:, :-1] 
    y = data.iloc[:, -1]   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: {conf_matrix}")
    print(f"Classification Report:{class_report}")
    
    return model.coef_.tolist() 

data = extract_texts_from_file('resources/heart.csv')
heart_classification(data)

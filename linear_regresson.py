import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import StringIO

def get_user_data():
    data = "X,Y\n"
    print("Enter your data in the format: X,Y (each entry in a new line)")
    print("End input with an empty line.")
    while True:
        line = input()
        if line == "":
            break
        data += line + "\n"
    return data

def perform_linear_regression(data):
    # Create a DataFrame
    df = pd.read_csv(StringIO(data))
    # Prepare the data
    X = df['X'].values.reshape(-1, 1)  # features
    y = df['Y'].values  # target variable
   
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
   
    # Make predictions
    y_pred = model.predict(X)
   
    # Display the coefficients
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
   
    # Plot the data and the regression line
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

data = get_user_data()
perform_linear_regression(data)
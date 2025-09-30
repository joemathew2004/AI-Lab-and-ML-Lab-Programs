import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# Display the first few rows
print("Iris Dataset:")
print(df_iris.head())

# Set the target variable
target_feature = 'petal length (cm)'

def linear_regression():
    # Predict 'petal length (cm)' using 'sepal length (cm)' as the independent variable
    X = df_iris[['sepal length (cm)']]
    y = df_iris[target_feature]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n---Linear Regression Results---")
    print(f"Testing Mean Squared Error: {test_mse}")
    print(f"Testing R² Score: {test_r2}")

    # Plot the linear regression fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Fitted line')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Linear Regression: Sepal Length vs Petal Length')
    plt.legend()
    plt.grid(True)
    plt.show()

def polynomial_regression():
    # Predict 'petal length (cm)' using 'sepal length (cm)' with polynomial regression
    X = df_iris[['sepal length (cm)']]
    y = df_iris[target_feature]
    
    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n---Polynomial Regression Results---")
    print(f"Testing Mean Squared Error: {test_mse}")
    print(f"Testing R² Score: {test_r2}")

    # Plot the polynomial regression fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')

    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Polynomial (degree={degree})')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Polynomial Regression: Sepal Length vs Petal Length')
    plt.legend()
    plt.grid(True)
    plt.show()

def multiple_linear_regression():
    # Predict 'petal length (cm)' using multiple features
    X = df_iris[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
    y = df_iris[target_feature]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n---Multiple Linear Regression Results---")
    print(f"Testing Mean Squared Error: {test_mse}")
    print(f"Testing R² Score: {test_r2}")

# Call the functions to perform regressions and display results
linear_regression()
polynomial_regression()
multiple_linear_regression()

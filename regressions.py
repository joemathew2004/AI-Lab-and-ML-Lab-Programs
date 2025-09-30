import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

csv_file_path = 'C:/Users/LENOVO/Downloads/BostonHousing.csv'
df_boston = pd.read_csv(csv_file_path)

student_data_path = 'C:/Users/LENOVO/Downloads/Student_Performance.csv'
df_student = pd.read_csv(student_data_path)

print("Boston Housing Data:")
print(df_boston.head())

def linear_regression():
    X = df_boston[['rm']]  # Independent variable (number of rooms)
    y = df_boston['medv']  # Dependent variable (median home value)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n---Linear Regression Results---")
    print(f"Testing Mean Squared Error: {test_mse}")
    print(f"Testing R² Score: {test_r2}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Fitted line')
    plt.xlabel('rm (Average Number of Rooms)')
    plt.ylabel('medv (Median Value of Homes in $1000s)')
    plt.title('Linear Regression: RM vs MEDV')
    plt.legend()
    plt.grid()
    plt.show()

    def predict_new_prices_linear():
        user_input = input("Enter values for RM: ")
        data = [list(map(float, user_input.split(',')))]
        new_samples = pd.DataFrame(data, columns=['rm'])
        predictions = model.predict(new_samples)

      #  new_samples = pd.DataFrame({
    # 'rm': [5.0, 6.0, 7.0]  # Average number of rooms
    #    })

       # predictions = model.predict(new_samples)
                
        print("Predictions for new sample:")
        for rm_value, price in zip(data[0], predictions):
            print(f"RM: {rm_value}, Predicted Price: ${price:.2f}")
                    

    predict_new_prices_linear()

def polynomial_regression():
    X = df_boston[['rm']]  # Independent variable (number of rooms)
    y = df_boston['medv']  # Dependent variable (median home value)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_test_pred = model.predict(X_test_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')

    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Polynomial (degree={degree})')
    plt.xlabel('rm (Average Number of Rooms)')
    plt.ylabel('medv (Median Value of Homes in $1000s)')
    plt.title('Polynomial Regression: RM vs MEDV')
    plt.legend()
    plt.grid(True)
    plt.show()

    def predict_new_prices_poly():
        while True:
            try:
                user_input = input("Enter values for RM: ")
                data = [list(map(float, user_input.split(',')))]

                new_samples = pd.DataFrame(data, columns=['rm'])
                new_samples_poly = poly_features.transform(new_samples)
                
                predictions = model.predict(new_samples_poly)

                exists_in_dataset = df_boston[df_boston['rm'].isin(data[0])]
                
                print("Predictions for new samples:")
                for rm_value, price in zip(data[0], predictions):
                    print(f"RM: {rm_value}, Predicted Price: ${price:.2f}")
                    if not exists_in_dataset.empty:
                        actual_price = df_boston[df_boston['rm'] == rm_value]['medv'].values[0]
                        print(f"Actual Price in Dataset: ${actual_price:.2f}")

                another = input("Do you want to enter another set of data? (yes/no): ").strip().lower()
                if another != 'yes':
                    break
            except ValueError:
                print("Invalid input. Please enter numerical values for RM.")

    predict_new_prices_poly()


linear_regression()
polynomial_regression()


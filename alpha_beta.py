import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston dataset
boston = load_boston()
X = boston.data
y = boston.target

# Step 2: Linear Regression (Using only one feature)
X_room = X[:, 5].reshape(-1, 1)  # Number of rooms

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_room, y, test_size=0.2, random_state=42)

# Apply Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

# Print Linear Regression results
print("Linear Regression (Number of Rooms):")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

# Step 3: Multivariate Regression (Using all features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Multivariate Regression
multi_regressor = LinearRegression()
multi_regressor.fit(X_train, y_train)
y_pred_multi = multi_regressor.predict(X_test)

# Print Multivariate Regression results
print("\nMultivariate Regression (All features):")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_multi))
print("R2 Score:", r2_score(y_test, y_pred_multi))

# Step 4: Polynomial Regression (Using one feature and generating polynomial terms)
# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_room)

# Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Apply Polynomial Regression
poly_regressor = LinearRegression()
poly_regressor.fit(X_train, y_train)
y_pred_poly = poly_regressor.predict(X_test)

# Print Polynomial Regression results
print("\nPolynomial Regression (Degree 2):")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))

# Step 5: Visualize the regression results (Optional)
plt.figure(figsize=(14, 5))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(X_room, y, color='blue')
plt.plot(X_room, linear_regressor.predict(X_room), color='red')
plt.title("Linear Regression (Rooms vs Price)")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")

# Polynomial Regression
plt.subplot(1, 3, 2)
plt.scatter(X_room, y, color='blue')
plt.plot(X_room, poly_regressor.predict(poly.fit_transform(X_room)), color='red')
plt.title("Polynomial Regression (Rooms vs Price)")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")

plt.tight_layout()
plt.show()

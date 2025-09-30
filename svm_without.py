'''import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        # Training using gradient descent
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                if y[idx] * (np.dot(x_i, self.w) + self.b) >= 1:
                    # Correctly classified
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# Example usage
if __name__ == "__main__":
    # Generate a simple dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])  # Labels for XOR

    # Train SVM
    svm = SVM(learning_rate=0.1, lambda_param=0.01, epochs=1000)
    svm.fit(X, y)

    # Make predictions
    predictions = svm.predict(X)
    print("Predictions:", predictions)
    
    
    

'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                if y[idx] * (np.dot(x_i, self.w) + self.b) >= 1:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

iris = load_iris()
X = iris.data
y = iris.target

y = np.where(y > 0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM(learning_rate=0.1, lambda_param=0.01, epochs=1000)
svm.fit(X_train, y_train)

# Function to get user input


# Get user input
user_input = [4,5,6,7]

# Make prediction
prediction = svm.predict(user_input)
print("Prediction:", prediction)
if prediction == 1:
    print("Iris-versicolor or Iris-virginica")
else:
    print("Iris-setosa")

# Plot SVM decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

x_values = np.linspace(x_min, x_max, 100)
y_values = (-svm.b - svm.w[0] * x_values) / svm.w[1]

plt.plot(x_values, y_values, 'k-')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('SVM Decision Boundary')
plt.show()

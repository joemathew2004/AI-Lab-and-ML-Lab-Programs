import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load California Housing dataset
california_housing = fetch_california_housing()

df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['target'] = california_housing.target
print(df.head())
# Define Naive Bayes class for continuous target (Gaussian Naive Bayes)
class NaiveBayes:
    def __init__(self):
        self.mean = {}
        self.variance = {}
        self.priors = {}

    def fit(self, X, y):
        self.mean = np.mean(y)
        self.variance = np.var(y)
        self.priors = 1  # Since we're predicting continuous values, assume equal prior probability for simplicity

    def calculate_likelihood(self, x, mean, var, epsilon=1e-10):
        var = np.maximum(var, epsilon)  # Ensure variance is non-zero
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        likelihood = (1 / np.sqrt(2 * np.pi * var)) * exponent
        return likelihood

    def calculate_posterior(self, x):
        posterior = np.log(self.priors)
        likelihood = self.calculate_likelihood(x, self.mean, self.variance, epsilon=1e-10)
        posterior += np.sum(np.log(np.clip(likelihood, 1e-9, 1 - 1e-9)))  # Clip likelihood
        return posterior

    def predict(self, X):
        # Predict by calculating posterior probability for each instance
        return np.array([self.calculate_posterior(x) for x in X])

# Split data into training and testing sets

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train.values, y_train.values)

# Make predictions
predictions = nb.predict(X_test.values)

# Evaluate model - Since target is continuous, use RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((predictions - y_test.values) ** 2))
print("Root Mean Squared Error:", rmse)
m=mean_squared_error(y_test.values, predictions)
print(m)


















from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

titanic = fetch_openml(name='titanic', version=1)

df = pd.DataFrame(titanic.data, columns=titanic.feature_names)
df['target'] = titanic.target

df = df.dropna()  # Remove rows with missing values, you could also use df.fillna() to fill them with a specific value

# Ensure that all categorical columns are encoded
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 3: Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Naive Bayes Classifier (Gaussian Naive Bayes for simplicity)
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.variance[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def calculate_likelihood(self, x, mean, var, epsilon=1e-10):
        var = np.maximum(var, epsilon)  # Ensure variance is non-zero
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        likelihood = (1 / np.sqrt(2 * np.pi * var)) * exponent
        return likelihood

    def calculate_posterior(self, x):
        posteriors = {}
        for cls in self.classes:
            posterior = np.log(self.priors[cls])
            likelihood = self.calculate_likelihood(x, self.mean[cls], self.variance[cls])
            posterior += np.sum(np.log(likelihood))
            posteriors[cls] = posterior
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self.calculate_posterior(x) for x in X])

nb = NaiveBayes()
nb.fit(X_train.values, y_train.values)

predictions = nb.predict(X_test.values)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")




import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features (sepal length and sepal width)
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM classifier with a linear kernel
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plotting the results (simple scatter plot of test data points)
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.title("SVM Classifier on Iris Dataset (Test Data)")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.colorbar(label='Class')
plt.show()















'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'C:/Users/LENOVO/Downloads/loan_data_set.csv'
data = pd.read_csv(file_path)
data.dropna(inplace=True)

label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the label encoder for later use

X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)  # Drop Loan_ID and Loan_Status columns
y = data['Loan_Status']  # Loan_Status is the target

# Standardize the numeric columns (optional, but useful for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def get_user_input():
    Gender = input("Enter Gender (Male/Female): ")
    Married = input("Are you Married? (Yes/No): ")
    Dependents = input("Number of Dependents (0, 1, 2, 3+): ")
    Education = input("Education Level (Graduate/Not Graduate): ")
    Self_Employed = input("Are you Self Employed? (Yes/No): ")
    ApplicantIncome = float(input("Enter Applicant Income: "))
    CoapplicantIncome = float(input("Enter Coapplicant Income: "))
    LoanAmount = float(input("Enter Loan Amount: "))
    Loan_Amount_Term = float(input("Enter Loan Amount Term: "))
    Credit_History = int(input("Credit History (0 or 1): "))
    Property_Area = input("Property Area (Urban/Semiurban/Rural): ")
    
    return {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

def predict_loan_status(user_input):
    user_df = pd.DataFrame([user_input])
    
    # Encode categorical columns using the stored label encoders
    for col in categorical_columns[:-1]:  # Exclude 'Loan_Status'
        le = label_encoders[col]
        user_df[col] = le.transform(user_df[col])
    
    # Standardize the input using the previously fitted scaler
    user_df_scaled = scaler.transform(user_df)
    
    prediction = svm_model.predict(user_df_scaled)
    loan_status = label_encoders['Loan_Status'].inverse_transform(prediction)  # Inverse transform to get original labels
    
    return loan_status[0]

user_input = get_user_input()
loan_status = predict_loan_status(user_input)
print(f"The predicted loan status is: {loan_status}")







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
file_path = 'C:/Users/LENOVO/Downloads/loan_data_set.csv'
data = pd.read_csv(file_path)
data.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare data for training
X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

# Standardize numeric columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hardcoded user input
user_input = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000.0,
    'CoapplicantIncome': 3000.0,
    'LoanAmount': 200000.0,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1,
    'Property_Area': 'Urban'
}

# Predict loan status
def predict_loan_status(user_input):
    user_df = pd.DataFrame([user_input])
    for col in categorical_columns[:-1]:
        le = label_encoders[col]
        user_df[col] = le.transform(user_df[col])
    user_df_scaled = scaler.transform(user_df)
    prediction = svm_model.predict(user_df_scaled)
    loan_status = label_encoders['Loan_Status'].inverse_transform(prediction)
    return loan_status[0]

# Get predicted loan status
loan_status = predict_loan_status(user_input)
print(f"The predicted loan status is: {loan_status}")'''
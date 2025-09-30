import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

file_path = 'C:/Users/LENOVO/Downloads/loan_data_set.csv'
data = pd.read_csv(file_path)
data.dropna(inplace=True)       # drop missing values

print("Dataset Preview:")
print(data.head())

label_encoders = {}             # Encode categorical columns
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Apply LabelEncoder on categorical columns and store encoders
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le    # Save the encoder for each column

# Encode Loan_Status separately for prediction
loan_status_encoder = LabelEncoder()
data['Loan_Status'] = loan_status_encoder.fit_transform(data['Loan_Status'])

X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
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
    
    for col in categorical_columns:
        user_df[col] = label_encoders[col].transform(user_df[col])
    
    prediction = nb_model.predict(user_df)
    
    # Decode the prediction using the loan_status_encoder
    loan_status = loan_status_encoder.inverse_transform(prediction)
    return loan_status[0]

user_input = get_user_input()
loan_status = predict_loan_status(user_input)
print(f"The predicted loan status is: {loan_status}")

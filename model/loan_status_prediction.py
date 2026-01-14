import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Data Preprocessing"""

# Loading the Dataset to Pandas DataFrame
loan_dataset = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Loan Status Prediction/data/Loan Status Data.csv')

# Dropping the missing values
loan_dataset = loan_dataset.dropna()

# LABEL ENCODING
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# Dependent column Values
loan_dataset['Dependents'] = loan_dataset['Dependents'].replace(to_replace='3+', value=4)

# Convert Categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0, 'Yes':1}, 
                      'Gender':{'Male':1,'Female':0}, 
                      'Self_Employed':{'No':0,'Yes':1}, 
                      'Property_Area':{'Rural':0, 'Semiurban':1, 'Urban':2}, 
                      'Education':{'Graduate':1, 'Not Graduate':0}}, inplace=True)

# Separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

"""SPLITTING THE DATA IN TRAINING AND TESTING"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

"""TRAINING THE MODEL: SUPPORT VECTOR MACHINE CLASSIFIER"""

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

"""MODEL EVALUATION"""

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on testing data: ", test_data_accuracy)

"""MAKING THE PREDICTIVE SYSTEM WITH USER-FRIENDLY INPUT"""

def preprocess_user_input(gender, married, dependents, education, self_employed, 
                          applicant_income, coapplicant_income, loan_amount, 
                          loan_amount_term, credit_history, property_area):
    """
    Convert user-friendly categorical inputs to numerical values
    """
    # Create a mapping dictionary
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    }
    
    # Handle dependents (convert '3+' to 4)
    if dependents == '3+':
        dependents = 4
    else:
        dependents = int(dependents)
    
    # Convert categorical values to numerical
    gender_encoded = mappings['Gender'][gender]
    married_encoded = mappings['Married'][married]
    education_encoded = mappings['Education'][education]
    self_employed_encoded = mappings['Self_Employed'][self_employed]
    property_area_encoded = mappings['Property_Area'][property_area]
    
    # Create the input array in the correct order
    # Order: Gender, Married, Dependents, Education, Self_Employed, 
    #        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
    #        Credit_History, Property_Area
    input_array = np.array([gender_encoded, married_encoded, dependents, 
                           education_encoded, self_employed_encoded,
                           applicant_income, coapplicant_income, loan_amount,
                           loan_amount_term, credit_history, property_area_encoded])
    
    return input_array.reshape(1, -1)


def predict_loan_status(gender, married, dependents, education, self_employed,
                       applicant_income, coapplicant_income, loan_amount,
                       loan_amount_term, credit_history, property_area):
    """
    Predict loan status based on user inputs
    
    Parameters:
    - gender: 'Male' or 'Female'
    - married: 'Yes' or 'No'
    - dependents: '0', '1', '2', '3+'
    - education: 'Graduate' or 'Not Graduate'
    - self_employed: 'Yes' or 'No'
    - applicant_income: numerical value
    - coapplicant_income: numerical value
    - loan_amount: numerical value
    - loan_amount_term: numerical value (in days)
    - credit_history: 1.0 (good) or 0.0 (bad)
    - property_area: 'Rural', 'Semiurban', or 'Urban'
    """
    # Preprocess the input
    processed_input = preprocess_user_input(gender, married, dependents, education, 
                                           self_employed, applicant_income, 
                                           coapplicant_income, loan_amount, 
                                           loan_amount_term, credit_history, 
                                           property_area)
    
    # Make prediction
    prediction = classifier.predict(processed_input)
    
    if prediction[0] == 0:
        return 'Loan Not Approved'
    else:
        return 'Loan Approved'


# Interactive user input
print("\n" + "="*50)
print("LOAN PREDICTION SYSTEM")
print("="*50)
print("\nPlease enter the following details:\n")

# Get user inputs
gender = input("Gender (Male/Female): ").strip()
married = input("Married (Yes/No): ").strip()
dependents = input("Number of Dependents (0/1/2/3+): ").strip()
education = input("Education (Graduate/Not Graduate): ").strip()
self_employed = input("Self Employed (Yes/No): ").strip()
applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_amount_term = float(input("Loan Amount Term (in days): "))
credit_history = float(input("Credit History (1.0 for good, 0.0 for bad): "))
property_area = input("Property Area (Rural/Semiurban/Urban): ").strip()

# Make prediction
try:
    result = predict_loan_status(
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_amount_term=loan_amount_term,
        credit_history=credit_history,
        property_area=property_area
    )
    
    print("\n" + "="*50)
    print(f"PREDICTION RESULT: {result}")
    print("="*50)
    
except KeyError as e:
    print(f"\nError: Invalid input value. Please check your entries.")
    print(f"Make sure you enter values exactly as specified (e.g., 'Male' not 'male')")
except Exception as e:
    print(f"\nError occurred: {e}")
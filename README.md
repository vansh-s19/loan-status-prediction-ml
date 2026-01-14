# Loan Status Prediction using Machine Learning

A machine learning–based system that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details such as income, credit history, education, and property area.  
The model is trained using a Support Vector Machine (SVM) classifier on a real-world banking dataset.

---

## Project Structure
Loan Status Prediction/
│
├── data/
│   └── Loan Status Data.csv
│
├── model/
│   ├── Loan_Status_Prediction.ipynb
│   ├── backupcode.py
│   └── loan_status_prediction.py
│
├── requirements.txt
└── README.md

---

## Features

- Uses **Support Vector Machine (SVM)** for classification  
- Handles missing values and categorical encoding  
- Accepts **human-readable user input** (Male, Yes, Graduate, Urban, etc.)  
- Provides real-time prediction from the terminal  
- Includes exploratory data analysis (EDA) in Jupyter Notebook  

---

## Dataset

The dataset contains loan application details such as:

- Gender  
- Marital Status  
- Education  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Term  
- Credit History  
- Property Area  
- Loan Status (Target Variable)

The dataset is located in the `data/` folder.

---

## Installation

Clone the repository:
git clone <your-repository-url>
cd Loan-Status-Prediction

## Install Dependencies
python3 -m pip install -r requirements.txt

---

## Running the model
Run the prediction system:
python3 loan_status_prediction.py

You will be prompted to enter details such as:
Gender (Male/Female):
Married (Yes/No):
Number of Dependents (0/1/2/3+):
Education (Graduate/Not Graduate):
Self Employed (Yes/No):
Applicant Income:
Coapplicant Income:
Loan Amount:
Loan Amount Term:
Credit History (1.0 for good, 0.0 for bad):
Property Area (Rural/Semiurban/Urban):

The system will then output:
PREDICTION RESULT: Loan Approved 
or 
PREDICTION RESULT: Loan Not Approved

---

## MODEL PERFORMANCE
The model achieves approximately:
	•	Training Accuracy: ~79%
	•	Testing Accuracy: ~83%

This indicates good generalization on unseen data.

---

## Technologies Used
	•	Python
	•	Pandas
	•	NumPy
	•	Scikit-learn
	•	Matplotlib
	•	Seaborn
	•	Jupyter Notebook

---
## Use Cases
	•	Bank loan pre-screening
	•	Fintech applications
	•	Credit risk analysis
	•	Student ML portfolio projects

---

Author

Vansh
Machine Learning & Python Developer

---

License

This project is open-source and free to use for educational and learning purposes.
If you want, I can also suggest **repository names** that look professional on GitHub (for example: `Loan-Approval-Predictor`, `ML-Loan-Status`, etc.).



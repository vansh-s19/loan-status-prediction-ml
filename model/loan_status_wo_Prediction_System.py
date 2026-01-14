import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Data Preprocessing"""

#Loading the Dataset to Pandas DataFrame
loan_dataset = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Loan Status Prediction/data/Loan Status Data.csv')

#printing the first 5 rows of dataset
#loan_dataset.head()

# number of rows and columns
#loan_dataset.shape

#loan_dataset.describe()

#number of missing values in each column

loan_dataset.isnull().sum()

#Dropping the missing values
loan_dataset = loan_dataset.dropna()

loan_dataset.isnull().sum()

#LABEL ENCODING
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

loan_dataset.head()

#Dependent column Values

loan_dataset['Dependents'].value_counts()

loan_dataset['Dependents'] = loan_dataset['Dependents'].replace(to_replace='3+', value=4)

loan_dataset['Dependents'].value_counts()

"""
DATA VISUALIZATION
"""

#EDUCATION AND STATUS LOAN
#sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
#plt.show()

#Martial Status and Loan Status

#sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
#plt.show()

# Convert Categorical columns to numericals values
loan_dataset.replace({'Married':{'No':0, 'Yes':1}, 'Gender':{'Male':1,'Female':0}, 'Self_Employed':{'No':0,'Yes':1}, 'Property_Area':{'Rural':0, 'Semiurban':1, 'Urban':2}, 'Education':{'Graduate':1, 'Not Graduate':0}}, inplace=True)

#loan_dataset.head()

#Separating the data and label

X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

#print(X)
#print(Y)

"""SPLITING THE DATA IN TRAINING AND TESTING"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

"""TRAINING THE MODEL:

SUPPORT VECTOR MACHINE CLASSIFIER
"""

classifier = svm.SVC(kernel='linear')

"""####training the support vector machine model"""

classifier.fit(X_train, Y_train)

"""MODEL EVALUATION"""

#Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ',training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy on testing data: ", test_data_accuracy)

"""MAKING THE PREDICTIVE SYSTEM"""

input_data=(1,1,0,1,0,3500,1667,114,360,1,1)
#changing th array to numpy array
input_data_as_numpy_array=np.asarray(input_data)
prediction=classifier.predict(input_data_as_numpy_array.reshape(1,-1))
print(prediction)
if (prediction[0]==0):
  print('not approved')
else:
  print('approved')
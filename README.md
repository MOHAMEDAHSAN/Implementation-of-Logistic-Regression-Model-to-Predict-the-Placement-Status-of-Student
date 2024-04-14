# EXP4-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
   
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S MOHAMED AHSAN
RegisterNumber: 212223240089
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
print(data.head())

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column


print(data1.isnull().sum())
print(data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
print(data1) 

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Y_Prediction : ",y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix : \n",confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
![Screenshot 2024-04-14 161653](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/f64d8a66-2ef3-495d-8f6b-eb98e2fa788f)
![Screenshot 2024-04-14 161658](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/076df101-200f-4b62-b5d7-1ed7dd9298b3)
![Screenshot 2024-04-14 161705](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/c0ff2184-93f5-4b9c-b809-b32c4b9413a4)
![Screenshot 2024-04-14 161717](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/2106edad-06ac-42dc-b6f4-330e6c5cecae)
![Screenshot 2024-04-14 161433](https://github.com/MOHAMEDAHSAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331378/dad11dba-8390-44e0-9eb2-6edd1cbeac55)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

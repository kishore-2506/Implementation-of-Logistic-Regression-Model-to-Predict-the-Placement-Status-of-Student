# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kishore M
RegisterNumber: 212223040100
```

```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull()

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
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Prediction Array : \n", y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy : ",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix : \n", confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification Report : \n\n",classification_report1)


from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```

## Output:

### DATA SET:
![image](https://github.com/user-attachments/assets/5ab1a5fb-efb4-4a32-9220-3d6398d90ddf)

![image](https://github.com/user-attachments/assets/55a64baa-d600-4c32-96a6-3de014a37c90)


### CHECKING DUPLICATE:
![image](https://github.com/user-attachments/assets/5e80b410-e83e-4068-bc74-a0f4599da429)

### ENCODED DATA:
![image](https://github.com/user-attachments/assets/d65e1c1a-dfff-4960-9144-d2b137033f00)


### DATA STATUS:
![image](https://github.com/user-attachments/assets/5f57e878-33b6-4620-b616-3e0c942489f4)


### PREDICTION ARRAY:
![image](https://github.com/user-attachments/assets/7716ee42-b59f-494d-9ac5-ab1e4edf7ac1)


### ACCURACY:
![image](https://github.com/user-attachments/assets/32effa88-3eda-457b-aa71-9b6d8d89620d)


### CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/898fd35f-be25-4736-acb3-cae16652e55f)


### CLASSIFICATION REPORT:
![image](https://github.com/user-attachments/assets/cf3c8c0d-13e4-48c9-a1c2-759c37c31797)


### CONFUSION MATRIX DISPLAY:
![image](https://github.com/user-attachments/assets/4fc4b6be-b45f-49ff-8e4a-0363372711f5)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

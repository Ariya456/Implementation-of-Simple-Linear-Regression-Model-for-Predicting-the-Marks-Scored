# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ariya Viniya.G
RegisterNumber: 212223080005

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/44fbcd5e-0c8f-486e-806e-9486a3eed64f)
![image](https://github.com/user-attachments/assets/c3242ecb-c5ce-4d83-a616-b97e31348be5)
![image](https://github.com/user-attachments/assets/49b5e62d-6307-49c7-9211-6f4d737528ab)
![image](https://github.com/user-attachments/assets/e8726cb6-9c4d-45f8-8407-417d3b569906)
![image](https://github.com/user-attachments/assets/1b320c86-abef-4afa-aa8e-667e1567b88a)
![image](https://github.com/user-attachments/assets/daf28e3a-c969-4b1b-bec0-a7a0c6ba499f)
![image](https://github.com/user-attachments/assets/09006691-876f-4f58-b9eb-5e7e35a18ef5)
![image](https://github.com/user-attachments/assets/38b05c78-de97-4857-8996-9c9e74873224)
![image](https://github.com/user-attachments/assets/bd72e8fe-d3bf-4041-b097-5d16f4e911f1)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2. Import the dataset to operate on.

3. Split the dataset.

4. Predict the required output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VARNIKA.P
RegisterNumber:  212223240170
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Output Result:
![ml ou 1](https://github.com/Sakthimurugavel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707246/f91b1187-467c-41ed-bbc3-f8971e2109a1)

### Data.head(): 
![ml out 2](https://github.com/Sakthimurugavel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707246/b6c99fc9-70f7-418b-8a20-d82288f48a94)

### Data.info():
![ml out 3](https://github.com/Sakthimurugavel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707246/8d5c7c4c-6f2a-4406-b5b8-27cb53ebdd2f)

### Y_prediction value:
![ml out 4](https://github.com/Sakthimurugavel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707246/2274017c-c031-4424-8350-25cf159e4b18)

### Accuracy value:
![ml out 5](https://github.com/Sakthimurugavel/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707246/7bc9befc-3692-4711-98c3-6c22f6729955)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

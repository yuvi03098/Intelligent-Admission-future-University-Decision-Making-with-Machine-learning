import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('C:/university decision making/project/Dataset/Admission_Predict.csv')
data.head()
data.info()
data.describe()
data.isnull().any()
sns.distplot(data['GRE Score'])
sns.pairplot(data=data,hue= 'Research',markers=["^","v"],palette='inferno')
sns.scatterplot(x='University Rating',y='CGPA',data=data,color='black',s=100)
category=['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
color=['yellowgreen','gold','lightskyblue','pink','red','purple','orange','gray']
start= True
i=0
i in np.arange(4)
fig = plt.figure(figsize=(14,8))
plt.subplot2grid((4,2),(i,0))
data[category[2*i]].hist(color=color[2*i],bins=10)
plt.title(category[2*i])
plt.subplot2grid((4,2),(i,1))
data[category[2*i+1]].hist(color=color[2*i+1],bins=10)
plt.title(category[2*i+1])
plt.subplots_adjust(hspace =0.7, wspace =0.2)
plt.show()
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x=data.iloc[:,0:7].values
x
y=data.iloc[:,8].values
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=101)
y_train=(y_train>0.5)
y_train
y_test=(y_test>0.5)
y_test
k=y_test.astype(int)
print("shape of independent training data is{}".format(x_train.shape))
print("shape of dependent training data is{}".format(y_train.shape))
from sklearn.linear_model import LogisticRegression
cls=LogisticRegression()
lr=cls.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
m=y_pred.astype(int)
from sklearn.metrics import r2_score
#r2_score(y_test,y_pred,force_finite=True)
r2_score(k,m)
lr.predict([[350,103,3,4.5,2.5,8.5,1]])
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
model=keras.Sequential()
model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='softmax'))
model.summary()
#model.fit(x_train,y_train,batch_size=20,epochs=100)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size=20,epochs=100)
from sklearn.metrics import accuracy_score
train_predictions = model.predict(x_train)
print(train_predictions)
train_acc = model.evaluate(x_train,y_train,verbose=0)[1]
print(train_acc)
test_acc = model.evaluate(x_test,y_test,verbose=0)[1]
print(test_acc)
pred=model.predict(x_test)
pred = (pred>0.5)
pred
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
print("Accuracy score :/n%f " %(accuracy_score(y_test,y_pred) * 100))
print("Recall score : %f" %(recall_score(y_test,y_pred) * 100))
print("ROC score : %f/n" %(roc_auc_score(y_test,y_pred) * 100))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print(classification_report(y_test,pred))
model.save('model.h5')

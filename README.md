#!/usr/bin/env python
# coding: utf-8

# # KNN

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd


# In[2]:


tcs=pd.read_csv('TCS.csv')
tcs.head()


# In[3]:


X = tcs.iloc[:, [0,1,2,3,4]].values
y = tcs.iloc[:, -1].values


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)


# In[5]:


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)


# In[6]:


knn_accuracy=(knn.score(X_test, y_test))


# In[7]:


print("KNN model accuracy(in %):",knn_accuracy*100)


# # NAIVE BAIYES

# In[8]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[9]:


y_pred = gnb.predict(X_test)


# In[12]:


from sklearn import metrics
print("NB model accuracy(in %):",metrics.accuracy_score(y_test, y_pred)*100)


# # DECISION TREE

# In[13]:


import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[14]:


def importdata():
 balance_data = pd.read_csv('TCS.csv')

 # Printing the dataswet shape
 print ("Dataset Length: ", len(balance_data))
 print ("Dataset Shape: ", balance_data.shape)

 # Printing the dataset obseravtions
 print ("Dataset: ",balance_data.head())
 return balance_data


# In[15]:


def splitdataset(balance_data):

 # Separating the target variable
 X = balance_data.iloc[:, [0,1,2,3,4]].values
 Y = balance_data.iloc[:, -1].values

 # Splitting the dataset into train and test
 X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
 print(X)

 return X, Y, X_train, X_test, y_train, y_test


# In[16]:


def train_using_gini(X_train, X_test, y_train):

 # Creating the classifier object
 clf_gini = DecisionTreeClassifier(criterion = "gini",
 random_state = 100,max_depth=3, min_samples_leaf=5)

 # Performing training
 clf_gini.fit(X_train, y_train)
 return clf_gini


# In[17]:



def tarin_using_entropy(X_train, X_test, y_train):

 # Decision tree with entropy
 clf_entropy = DecisionTreeClassifier(
 criterion = "entropy", random_state = 100,
 max_depth = 3, min_samples_leaf = 5)

 # Performing training
 clf_entropy.fit(X_train, y_train)
 return clf_entropy


# In[18]:


def prediction(X_test, clf_object):

 # Predicton on test with giniIndex
 y_pred = clf_object.predict(X_test)
 print("Predicted values:")
 print(y_pred)
 return y_pred


# In[19]:


def cal_accuracy(y_test, y_pred):

 print("Confusion Matrix: ",
 confusion_matrix(y_test, y_pred))

 print ("Accuracy : ",
 accuracy_score(y_test,y_pred)*100)

 print("Report : ",
 classification_report(y_test, y_pred))


# In[20]:


def main():

 # Building Phase
 data = importdata()
 X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
 clf_gini = train_using_gini(X_train, X_test, y_train)
 clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

 # Operational Phase
 print("Results Using Gini Index:")
 # Prediction using gini
 y_pred_gini = prediction(X_test, clf_gini)
 cal_accuracy(y_test, y_pred_gini)

 print("Results Using Entropy:")
 # Prediction using entropy
 y_pred_entropy = prediction(X_test, clf_entropy)
 cal_accuracy(y_test, y_pred_entropy)

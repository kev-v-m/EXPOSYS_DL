#!/usr/bin/env python
# coding: utf-8

# # DIABETES CLASSIFICATION IN PIMA INDIAN DIABETES DATASET
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score    


# In[2]:


df=pd.read_csv('C:\\Users\Kevin\Desktop\EXPOSYS\diabetes.csv')
x=df.drop(["Outcome"],axis=1)
y=df["Outcome"]

df.describe()


# In[3]:




x[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] =x[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

x['Glucose'].fillna(x['Glucose'].mean(), inplace = True)
x['BloodPressure'].fillna(x['BloodPressure'].median(), inplace = True)
x['SkinThickness'].fillna(x['SkinThickness'].median(), inplace = True)
x['Insulin'].fillna(x['Insulin'].median(), inplace = True)
x['BMI'].fillna(x['BMI'].median(), inplace = True)


# ## Data Preprocesing

# #### Imputing Outliers

# #### Skewness Elimination

# # # Training and Accuracy with elbow Implementation of KNN

# In[4]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,stratify=y,random_state=56)

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)'''

sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)




x_train

 


# In[5]:


from sklearn.decomposition import PCA
 
pca = PCA(n_components = 2)
 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
 
explained_variance = pca.explained_variance_ratio_


# In[6]:


lst2=[]
maxpos=[]

for i in range(3,10):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=(10-i)/10 ,stratify=y,random_state=56)

    lst=[]

    lstidx=[]
    for j in range(1,20):
        clf=KNN(n_neighbors=j)
        clf.fit(x_train,y_train)
        pred=clf.predict(x_test)
        lst.append(clf.score(x_test,y_test))
        

    lst2.append(max(lst))
    maxpos.append(lst.index(max(lst))+1)


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,stratify=y,random_state=56)

clf=KNN(n_neighbors=maxpos[len(maxpos)-1])
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
clf.score(x_test,y_test)


# In[8]:


from sklearn.metrics import confusion_matrix
    #let us get the predictions using the classifier we had fit above

confusion_matrix(y_test,pred)
pd.crosstab(y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[9]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:





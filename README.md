# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE AND OUTPUT

## DATA PREPROCESSING BEFORE FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/2ea63aec-e8e6-4d49-a1fc-8b641b82f82a)

## checking data

```
df.isnull().sum()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/bd8a4586-4dbd-42da-bf38-aa93904004da)

## removing unnecessary data variables
```
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/4b00b9a0-c562-486f-bd51-47e348d70b83)

## cleaning data
```
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/0c9f21b0-c0e4-404f-9aeb-b2b2bd619362)

## removing outliers 
```
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/2760e195-63b2-4d91-a65f-c61a23871d7b)

```
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/5a7d5df5-a1aa-4636-a531-87c9dcc134ca)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/ad718c4a-55dc-4636-b8c4-849ce7239315)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/81dfcc2e-4dfc-4f78-bacc-898f8bdd7a45)

```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/3e5980e1-83f6-414f-befd-68a52d20dcad)

```
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/ba1d5e43-9e69-4b6f-a191-e6faf15441e0)

```
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/27577584-87f5-454d-8148-549ccb2ce949)

##  FEATURE SELECTION:
##  FILTER METHOD:
```
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/4a19eaf1-0084-4561-85bd-1ed1a2b6d57c)

## HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/10162331-27d4-45d9-8a31-1e16a0effb9e)

## BACKWARD ELIMINATION:
```
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/99a2ac5c-b905-47a0-88bc-5f63b269f77c)

## RFE (RECURSIVE FEATURE ELIMINATION):
```
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/3805fcb5-cfcd-4b6b-9cde-14513edf4b34)

## OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/0bcea6c1-3d5c-4b76-a48b-1ffc9d6a482a)

## FINAL SET OF FEATURE:
```
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/a0b8d877-bd72-4d0a-a6fa-f4c9e7a6d447)

## EMBEDDED METHOD:
```
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/b0ae90d9-204e-4164-87da-8ed14ffa2ab7)


# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.

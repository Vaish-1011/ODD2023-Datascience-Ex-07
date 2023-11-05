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

# CODE
## titanic_dataset.csv :
```
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from google.colab import files
upload = files.upload()
df = pd.read_csv('titanic_dataset.csv')
df
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/01969cd4-b742-433c-8706-bf0ec799b83b)
```
df.isnull().sum()
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/7f363e8c-bb33-4b26-b606-22ae262fab68)
```
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
print("Feature selection")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)
print(X_new)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/d193b9b5-a309-4b18-a365-41bf6c81460b)
```
df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])
df_new['Survived'] = y.values
df_new.to_csv('titanic_transformed.csv', index=False)
print(df_new)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/9dc9581e-8ad5-4e62-8515-a74da6adea88)

## CarPrice.csv:
```
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("CarPrice.csv")
df
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/b2580cae-f9a2-4e1a-aad0-8ede5ce88a54)
```
df = df.drop(['car_ID', 'CarName'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Univariate Selection")
selector = SelectKBest(score_func=f_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]
model = ExtraTreesRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
selected_features = X_train.columns[indices][:10]
df_new = pd.concat([X_train[selected_features], y_train], axis=1)
df_new.to_csv('CarPrice_new.csv', index=False)
print(df_new)
```
![image](https://github.com/Vaish-1011/ODD2023-Datascience-Ex-07/assets/135130074/fbc2de53-1a58-4e50-8668-652f944e9a49)

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.

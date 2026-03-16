# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
# Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("income.csv")

print("Dataset Preview:")
print(df.head())

# Encode Categorical Variables
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)

# Encode Target Variable
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes

# Separate Features and Target
X = df.drop(columns=['SalStat'])
y = df['SalStat']

# Scale Data for Chi-Square
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Filter Method: Chi-Square
selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

# Filter Method: ANOVA
selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

# Wrapper Method: RFE
logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))

# Embedded Method
rf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))

# Accuracy using Embedded Features
X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)

rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)
print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))
```
<img width="518" height="234" alt="Screenshot 2026-03-16 214625" src="https://github.com/user-attachments/assets/690439da-e3aa-47b7-9c5f-fbbc0a7c5fba" />
<img width="521" height="676" alt="Screenshot 2026-03-16 214634" src="https://github.com/user-attachments/assets/780f7244-302f-42e3-b9df-6bfc183cadf0" />


# RESULT:
The important features were selected using Chi-Square, ANOVA, RFE and Embedded methods, and the model achieved an accuracy of 0.83.
# PDF FILE:
[vertopal.com_DS EXP 4.pdf](https://github.com/user-attachments/files/26030281/vertopal.com_DS.EXP.4.pdf)

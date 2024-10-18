# EXNO:4-DS
## Name: Krishna Kumar R
## Reg no: 212223230107

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
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2024-10-18 155154](https://github.com/user-attachments/assets/5ab3d3a5-a402-4541-87fb-113ed783c66d)

```
df.dropna()
```
![Screenshot 2024-10-18 155202](https://github.com/user-attachments/assets/f0bef26e-04e6-415f-bcea-45babfc8434e)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-10-18 155208](https://github.com/user-attachments/assets/33eac300-b943-4e4a-8baf-a271caf4c832)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-18 155217](https://github.com/user-attachments/assets/3f70b141-dcf9-48d8-b09c-e6dda18ae009)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-18 155225](https://github.com/user-attachments/assets/616843ee-29bd-447e-b618-31dc4710bcde)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-18 155232](https://github.com/user-attachments/assets/0cfa25be-91fc-4a07-9312-cd72aa5d2d4d)

```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![Screenshot 2024-10-18 155237](https://github.com/user-attachments/assets/c094ea4d-7139-451f-9a61-c18aecf1d37c)

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![Screenshot 2024-10-18 155243](https://github.com/user-attachments/assets/c3e70696-b972-49c7-844f-0981429f9930)

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income.csv',na_values=[" ?"])
data
```
![Screenshot 2024-10-18 155328](https://github.com/user-attachments/assets/330e1a09-8244-4db2-89a9-306b15896604)

```
data.isnull().sum()
```
![Screenshot 2024-10-18 155347](https://github.com/user-attachments/assets/0dfa78c0-a154-466e-9493-f2ee4e66b473)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-18 155407](https://github.com/user-attachments/assets/613075e2-56af-49bd-8523-eb777e2b7d10)

```
data2 = data.dropna(axis=0)
data2
```
![Screenshot 2024-10-18 155542](https://github.com/user-attachments/assets/598e7650-3f6b-4229-8d3e-dc56485b1dfa)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-18 155600](https://github.com/user-attachments/assets/ec2c9279-6c1f-4358-8988-e8d8cff01e08)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-18 155607](https://github.com/user-attachments/assets/ef8ea314-0bc9-445c-b62c-081818705111)

```
data2
```
![Screenshot 2024-10-18 155622](https://github.com/user-attachments/assets/28da7870-604b-4463-b459-892711f078c4)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-18 155727](https://github.com/user-attachments/assets/d6d937a0-d137-46a1-bb5c-20a7f3b4206c)
![Screenshot 2024-10-18 155744](https://github.com/user-attachments/assets/54099444-0650-403d-8de4-a956c2ac11e5)

```
columns_list=list(new_data.columns)
print(columns_list)
```
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'SalStat', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
['gender_ Male', 'occupation_ Tech-support', 'nativecountry_ Laos', 'nativecountry_ Peru', 'nativecountry_ Ecuador', 'occupation_ Craft-repair', 'nativecountry_ England', 'nativecountry_ Holand-Netherlands', 'EdType_ Assoc-acdm', 'race_ White', 'nativecountry_ Thailand', 'nativecountry_ Iran', 'maritalstatus_ Married-spouse-absent', 'nativecountry_ Hong', 'maritalstatus_ Widowed', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'occupation_ Priv-house-serv', 'nativecountry_ Japan', 'nativecountry_ Italy', 'nativecountry_ France', 'nativecountry_ Nicaragua', 'relationship_ Unmarried', 'maritalstatus_ Married-AF-spouse', 'occupation_ Machine-op-inspct', 'JobType_ State-gov', 'nativecountry_ Puerto-Rico', 'nativecountry_ Cuba', 'EdType_ Bachelors', 'race_ Asian-Pac-Islander', 'nativecountry_ Hungary', 'nativecountry_ Poland', 'EdType_ 5th-6th', 'JobType_ Without-pay', 'nativecountry_ Portugal', 'capitalgain', 'JobType_ Private', 'nativecountry_ Honduras', 'EdType_ 12th', 'nativecountry_ Philippines', 'nativecountry_ Haiti', 'occupation_ Handlers-cleaners', 'EdType_ Assoc-voc', 'capitalloss', 'nativecountry_ Dominican-Republic', 'nativecountry_ Canada', 'occupation_ Transport-moving', 'EdType_ 11th', 'EdType_ Masters', 'occupation_ Prof-specialty', 'nativecountry_ Scotland', 'nativecountry_ Guatemala', 'hoursperweek', 'nativecountry_ Vietnam', 'nativecountry_ Ireland', 'EdType_ 1st-4th', 'EdType_ HS-grad', 'relationship_ Other-relative', 'occupation_ Other-service', 'nativecountry_ Jamaica', 'EdType_ Some-college', 'occupation_ Farming-fishing', 'nativecountry_ United-States', 'nativecountry_ Yugoslavia', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Never-married', 'EdType_ 7th-8th', 'occupation_ Protective-serv', 'occupation_ Armed-Forces', 'nativecountry_ Columbia', 'nativecountry_ Mexico', 'age', 'EdType_ 9th', 'nativecountry_ China', 'race_ Other', 'nativecountry_ Greece', 'nativecountry_ Taiwan', 'relationship_ Wife', 'relationship_ Own-child', 'nativecountry_ South', 'nativecountry_ Germany', 'maritalstatus_ Separated', 'EdType_ Doctorate', 'JobType_ Self-emp-inc', 'nativecountry_ India', 'nativecountry_ El-Salvador', 'EdType_ Preschool', 'JobType_ Local-gov', 'occupation_ Exec-managerial', 'race_ Black', 'JobType_ Self-emp-not-inc', 'relationship_ Not-in-family', 'nativecountry_ Trinadad&Tobago', 'EdType_ Prof-school', 'occupation_ Sales']

```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```
![Screenshot 2024-10-18 161217](https://github.com/user-attachments/assets/dde74804-280a-4a7e-81af-da7463aca340)

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-18 161223](https://github.com/user-attachments/assets/0448a031-50f8-422b-95f7-38150d235ba7)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![Screenshot 2024-10-18 161228](https://github.com/user-attachments/assets/191a4a62-94d2-451b-917a-478772ff70a9)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![Screenshot 2024-10-18 161232](https://github.com/user-attachments/assets/d255f5dc-691b-43fc-877f-26a6c45a4018)

```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![Screenshot 2024-10-18 161237](https://github.com/user-attachments/assets/e641aad2-5f32-48d8-9128-fe6cc3d244a7)

```
data.shape
```
![Screenshot 2024-10-18 161242](https://github.com/user-attachments/assets/acba2dd6-323f-4e33-b329-f0729fe652d3)


## FEATURE SELECTION TECHNIQUES

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-10-18 161247](https://github.com/user-attachments/assets/786eb15e-8a5d-49bb-b4a7-d6f64abd257b)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-18 161252](https://github.com/user-attachments/assets/f0d4a009-bde2-4dfb-a703-04d94fe36816)

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![Screenshot 2024-10-18 161257](https://github.com/user-attachments/assets/f1e8ca8c-27a2-4074-bb1b-b53b8cfab5bc)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-10-18 161302](https://github.com/user-attachments/assets/a5ea79f5-6e2b-4165-9754-9d519a46c6bd)

## RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.

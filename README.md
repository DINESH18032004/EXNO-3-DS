# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:

Read the given Data.

STEP 2:

Clean the Data Set using Data Cleaning Process.

STEP 3:

Apply Feature Encoding for the feature in the data set.

STEP 4:

Apply Feature Transformation for the feature in the data set.

STEP 5:

Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : DINESH KUMAR R
### Reg No : 212222110010

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/4e597843-b587-47ef-a0ef-8a5c148e1c43)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/b24d0ec2-ce88-433d-8a68-1b250eb90bc0)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/c7c17572-7e61-486c-8bc1-84ef7f7bf0a4)


```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/f644bd9e-faf1-4dc8-8f2e-017936d57c29)

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/e93ff887-364a-4648-bb3a-db57b22897b0)


```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/27de9791-f5cb-45ef-bda9-a01711a7cf62)


```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/6a3b23ee-e685-44fa-b5bd-376801b673f6)


```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/53c4b643-c66d-4e59-9a79-59b86659063a)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/de362b59-a607-4074-b936-51c2dd5961a9)


```py
df.skew()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/082d0996-a0e1-49f1-b3c5-3940b7d6979e)


```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/ffcc3fb5-28ee-4d92-9faa-c4bedc46b0b9)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/4f702e25-e31a-4331-bc26-b8a13ffb654b)


```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/e68f7497-d0b8-4117-b004-4703cfda95df)

```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/d4df145e-ad53-4d67-8f3d-530c1cfa57e2)


```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/e6c64b78-0d7c-4e13-a393-d368afd688b9)


```py
df.skew()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/fbca1acb-df31-42ab-8aa7-0c9da5f20599)


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/05958632-1c31-45b3-acdc-adb01e9dff4d)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/6febfb2f-6d9e-4d62-81a9-2d01903d33e6)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/32e38087-a50b-44f5-bf97-d3d2c328addb)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/848ca609-fe02-4f7f-98be-5abc2f625252)



```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/e87c63bb-5bf8-4391-8285-e72a92205eb9)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/a114a142-799d-4e82-8cb0-6930b5bf6f4d)


```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/95e18d7d-add7-45ad-906e-a8f6c37951cc)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/DINESH18032004/EXNO-3-DS/assets/119477784/658c6c99-d034-4393-b8bf-84e1bfefcb69)


## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       

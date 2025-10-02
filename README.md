## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
      ```
      import pandas as pd
      df=pd.read_csv("/content/Encoding Data.csv")
      df
     ```
     
   <img width="620" height="607" alt="Screenshot 2025-10-02 183435" src="https://github.com/user-attachments/assets/224ef09a-0016-4e7a-9591-525821d74bfd" />


  ```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
 pm=['Hot','Warm','Cold']
 e1=OrdinalEncoder(categories=[pm])
 e1.fit_transform(df[["ord_2"]])

```

<img width="793" height="372" alt="image" src="https://github.com/user-attachments/assets/408817aa-086a-40c0-b15f-4dd00dd2cd65" />

```

 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df

```

<img width="641" height="537" alt="image" src="https://github.com/user-attachments/assets/811220e1-abe1-4908-8002-cb1e204643e2" />

```

le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc

```


<img width="682" height="591" alt="image" src="https://github.com/user-attachments/assets/46f9b22b-941d-41da-86be-9a88f42b74c0" />

```

from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder(sparse_output=False)
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2

```

<img width="665" height="633" alt="image" src="https://github.com/user-attachments/assets/f6c02a30-6a4d-4adb-8db5-40fa704f3709" />


```

pd.get_dummies(df2,columns=["nom_0"])


```

<img width="941" height="527" alt="image" src="https://github.com/user-attachments/assets/e39f069d-0e08-407a-95d9-b830319c9f78" />

```

from category_encoders import BinaryEncoder
 df=pd.read_csv("/content/data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb


```

   <img width="1014" height="685" alt="image" src="https://github.com/user-attachments/assets/f692d75f-98d7-4114-8481-69cd32abab8d" />


```

from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC

```

<img width="829" height="635" alt="image" src="https://github.com/user-attachments/assets/09a27cad-cc26-4391-b47a-85b6aa5755a5" />


```

import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("/content/Data_to_Transform.csv")
 df

```


<img width="1085" height="657" alt="image" src="https://github.com/user-attachments/assets/5b83fbc1-6960-4c58-bdde-0e19e603b45d" />


```

df.skew()

```

<img width="540" height="313" alt="image" src="https://github.com/user-attachments/assets/0d931665-9c67-4834-acf0-1975a8b008f2" />


```

 np.log(df["Highly Positive Skew"])

```


<img width="536" height="644" alt="image" src="https://github.com/user-attachments/assets/70e34e00-6a60-4382-a21d-e0352997eafc" />


```

 np.reciprocal(df["Moderate Positive Skew"])

```

<img width="586" height="633" alt="image" src="https://github.com/user-attachments/assets/fa0ba3a9-1ec2-43e7-9731-f0b3714f6609" />


```

np.sqrt(df["Highly Positive Skew"])

```

<img width="529" height="629" alt="image" src="https://github.com/user-attachments/assets/c664bcd9-3a04-49ce-897a-5723a3902234" />


```

 np.square(df["Highly Positive Skew"])

```

<img width="533" height="636" alt="image" src="https://github.com/user-attachments/assets/ca314f59-fccf-4602-a0e7-28ee1d6042cd" />


```

import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```

<img width="1032" height="738" alt="image" src="https://github.com/user-attachments/assets/c82e202c-14e1-461e-9eb5-57ff1a7922f9" />

```

 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()

```

<img width="867" height="651" alt="image" src="https://github.com/user-attachments/assets/e830be5f-9d96-42fb-8b46-9bb55d461c44" />


```

from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```

<img width="885" height="711" alt="image" src="https://github.com/user-attachments/assets/57698c6b-461c-4f4b-9e69-ea02472f66d7" />


```

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()

```

<img width="870" height="680" alt="image" src="https://github.com/user-attachments/assets/c33d50b8-0c90-4bf7-be66-298a47773a65" />


```

dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt

```

```

from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
 plt.show()

```

<img width="853" height="712" alt="image" src="https://github.com/user-attachments/assets/1d77d26b-00f7-46c5-834f-facb1c115bff" />


```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()

```

<img width="847" height="654" alt="image" src="https://github.com/user-attachments/assets/83a137af-f5fd-40fa-ba33-af22698ffbbe" />

# RESULT:
       Thus the result is verified

       

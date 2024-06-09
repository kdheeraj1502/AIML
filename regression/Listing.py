import pandas as pd
import numpy as np

df=pd.read_csv("/Users/dheerajkumar/Documents/AIML-Classes/listings.csv", index_col=0)
print(df)
print(df.head())
print(df.describe())
df.drop_duplicates()
df.isna().sum()

#duplicates = df.duplicated()
df.drop(['license','neighbourhood_group'],axis=1)
df['reviews_per_month']=df['reviews_per_month'].fillna(0)
df['last_review']=df['last_review'].fillna(0000-00-00)
df['last_review']=df['last_review'].str[:-6].astype(float)
print(df.dtypes)
a =df['room_type'].value_counts()
print(a)
df['room_type']=df['room_type'].map({'Entire home/apt':0,'Private room':1,'Shared room':2,'Hotel room':3})
df=df.drop(['name','host_name'],axis=1)
y=df['price']
X=df.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split
X_train_data,X_test_data,y_train_data,y_test_data=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.model_selection import train_test_split
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train_data)

# Transform the test data
X_test_scaled = scaler.transform(X_test_data)


# Print the first 5 rows of the scaled training data
print(X_train_scaled[:5])
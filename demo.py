import numpy as np
import pandas as pd
import sklearn

#Preparation of data for a ML model

#Generating data
data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}
df_raw = pd.DataFrame(data=data)

#Making a copy of data and changing the data type
df = df_raw.copy()
print(df)
df.info()

for col in ['size', 'color', 'gender','bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')
df.info()

#Initial exploration
print("-------------------------")
print(df.describe(include=['category']).T)
print("-------------------------")
print(df.describe().T)
print("-------------------------")
print(df)

print("--------sklearn LabelEncoder----------")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df['bought'])
le.transform(df['bought'])

df['bought'] = le.fit_transform((df['bought']))
print(df)

print("--------pandas get_dummies--------")

df_dumm = pd.get_dummies(data=df, drop_first=True, columns=['size'])
print(df_dumm)

print("-------------------------")

#Standardization


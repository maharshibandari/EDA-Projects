############# DUMMY VARIABLES #############

import pandas as pd 

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")

df.columns #columns names
df.shape
df.dtypes
df.info()

#Drop columns
df.drop(["Employee_Name", "EmpID", "Zip"], axis = 1, inplace = True) #axis 1 = "column", 0 = "row" and "inplace" "True" will override & droop columns

#create dummy variables 
df.dtypes
df_new = pd.get_dummies(df)
df_new1 = pd.get_dummies(df, drop_first = True) #which comes first drop it
#Created dummies for all categorical columns


### One Hot Encoding ###

df.columns
df = df[["Salaries", "age", "Position", "State", "Sex", "MaritalDesc", "CitizenDesc", "EmploymentStatus", "Department", "Race"]] #Changing the order of the columns

from sklearn.preprocessing import OneHotEncoder

#Creating instance of OneHotEncoder
enc = OneHotEncoder() #Initializing method
enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray()) #[row:column,start_index:end_index], "toarray - Parse all the elements as it is".



### Label Encoder ###

from sklearn.preprocessing import LabelEncoder

#Creating instance of labelencoder 
labelencoder = LabelEncoder()

#Data split into Input & Output variables
X = df.iloc[:,:9] # 0 to 8 index
y = df.iloc[:,9:] # From 9 to rest of all

#Label Encoding
X["Sex"] = labelencoder.fit_transform(X["Sex"])
X["MaritalDesc"] = labelencoder.fit_transform(X["MaritalDesc"])
X["CitizenDesc"] = labelencoder.fit_transform(X["CitizenDesc"])

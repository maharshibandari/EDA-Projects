############# MISSING VALUES - IMPUTATION ##########
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\modified ethnic.csv")

# Check for count of NA's in each column
df.isna().sum()

# Creat an impute object that fills "Nan" values
# mean and median imputer are used for numeric data(Salaries)
# mode is used for discrete data (ex:- Position, Sex, MariatlDesc)
# For mean, median, mode imputation we can use "SimpleImputer" or "df.fillna()"
from sklearn.impute import SimpleImputer



### mean Imputer ###
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isna().sum() #All records replaced by mean



### median Imputer ###
median_imputer = SimpleImputer(missing_values = np.nan, strategy = "median")
df["age"] = pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum() #All records replaced by median

df.isna().sum()



### mode Imputer ###
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))
df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[["MaritalDesc"]]))
df.isnull().sum()



### Constant value Imputer ###
constant_imputer = SimpleImputer(missing_values = np.nan, strategy = "constant", fill_value = "F") #fill_value can be used for numeric or non_numeric values
df["Sex"] = pd.DataFrame(constant_imputer.fit_transform(df[["Sex"]]))

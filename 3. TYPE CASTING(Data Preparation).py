#######TYPE CASTING#######
import pandas as pd
df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")
df.dtypes
help(df.dtypes)

#Convert "int64" to "str" type
df.EmpID = df.EmpID.astype("str")
df.dtypes
df.Zip = df.Zip.astype("str")
df.dtypes

#For practice 
#Convert data types of columns from 
#"float64" to "int64" type
df.Salaries = df.Salaries.astype("int64")
df.dtypes

#"int64" to "float32"
df.age = df.age.astype("float32")
df.dtypes

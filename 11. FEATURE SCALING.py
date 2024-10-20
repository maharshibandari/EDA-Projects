############# STANDARDIZATION and NORMALIZATION #############

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\mtcars.csv")

a = df.describe() # Easy to  access it when we needed by storing it in variable



### STANDARDIZATION ###

from sklearn.preprocessing import StandardScaler

# Initialize the Scaler
scaler = StandardScaler()

# To scale data
df_1 = scaler.fit_transform(df)

# Convert the array back to dataframe 
dataset = pd.DataFrame(df_1) ### "df_1 = pd.DataFrame(scaler.fit_transform(df))" ###
res = dataset.describe()



### NORMALIZATION ###

ethnic1 = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")

ethnic1.columns
ethnic1.drop(["Employee_Name", "EmpID", "Zip"], axis = 1, inplace = True)
a1 = ethnic1.describe()

# Get dummies
ethnic = pd.get_dummies(ethnic1, drop_first = True, dtype = int)
a2 = ethnic.describe()

# Normalization Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max() - i.min())
    return(x)

df_norm = norm_func(ethnic)
b = df_norm.describe()

# Alternatively we can use the beelow function
from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()
ethnic_minmax = minmaxscaler.fit_transform(ethnic)
df_ethnic = pd.DataFrame(ethnic_minmax)
minmax_res = df_ethnic.describe()



### ROBUST SCALING ###
# Scale features using statistics that are robust to outliers
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(ethnic)
dataset_robust = pd.DataFrame(df_robust)
res_robust = dataset_robust.describe()

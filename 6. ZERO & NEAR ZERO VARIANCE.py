############# ZERO & NEAR ZERO VARIANCE #############
import pandas as pd

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")
df.dtypes

#If the variance is low or close to zero, then a feature is approxmately and will not improve the performance of the model.
#In that case, it should be removed.
help(df.var)
df.var(numeric_only = True)
df.var(numeric_only = True) == 0
df.var(numeric_only = True, axis = 0) == 0   #axis "0 = row", "1 = column"

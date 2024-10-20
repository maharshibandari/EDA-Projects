############# DISCRETIZATION #############

import pandas as pd
df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")

df.head() #Default only 5 rows
df.tail() #Default only 5 rows
df.info()
df.describe() #count, mean, std & etc...

### Binarization ###

df["Salaries_new"] = pd.cut(df.Salaries,
                            bins = [min(df.Salaries),
                                    df.Salaries.mean(),
                                    max(df.Salaries)],
                            labels = ["Low", "High"])

#Look out for the break up of categories
df.Salaries_new.value_counts() #Total no. of variables in each category

'''
We can observe that the total number of values are 309. This is because one of the value has beaome NA. 
This happen as the cut function by the default dose not consider the lower(min) while discretizing the values.
To overcome this issue we can use the parameter "include_lowest" set to "True"
'''

df["Salaries_new1"] = pd.cut(df.Salaries,
                             bins = [min(df.Salaries),
                                     df.Salaries.mean(),
                                     max(df.Salaries)],
                             include_lowest = True,
                             labels = ["Low", "High"])
df.Salaries_new1.value_counts()

### Discretization/Multiple Bins ###

df["Salaries_multi"] = pd.cut(df["Salaries"],
                              bins = [min(df.Salaries),
                                      df.Salaries.quantile(0.25),
                                      df.Salaries.mean(),
                                      df.Salaries.quantile(0.75),
                                      max(df.Salaries)],
                              include_lowest = True,
                              labels = ["P1","P2", "P3", "P4"])
df.Salaries_multi.value_counts()
df.MaritalDesc.value_counts()

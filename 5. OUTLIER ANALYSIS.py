#######Outlier Analysis#######
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\ethnic diversity.csv")
df.dtypes

#Let's findout Outliers in Salaries
sns.boxplot(df.Salaries)

sns.boxplot(df.age)
#No outliers in age column

#Detection of outliers(find limits for salary based on IQR)
IQR = df["Salaries"].quantile(0.75) - df.Salaries.quantile(0.25) #we can declare in both ways
lower_limit = df.Salaries.quantile(0.25) - (IQR * 1.5)
upper_limit = df.Salaries.quantile(0.75) + (IQR * 1.5)



############# 1. REMOVE (Let's trim the dataset) #############

#Trimming technique
#Let's flag the outliers in the dataset
outliers_df = np.where(df.Salaries > upper_limit, True, np.where(df.Salaries < lower_limit, True, False)) #If both conditions failed it gives false
df_trimmed = df.loc[~(outliers_df),] #If answer is True then only it prints the value
df.shape
df_trimmed.shape

#Let's explore outliiers in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)



############# 2. REPLACE #############

#Replace the outliers by the maximum & minimum limit
df["df_replaced"] = pd.DataFrame(np.where(df.Salaries > upper_limit, upper_limit, np.where(df.Salaries < lower_limit, lower_limit, df.Salaries)))
sns.boxplot(df.df_replaced)



############# 3. WINSORIZATION #############

#pip install feature_engine  (Install the package)
from feature_engine.outliers import Winsorizer

####### Define the modl with IQR method
winsor_iqr = Winsorizer(capping_method = "iqr",   #Choose IQR rule or GAUSSIAN method for mean & std
                       tail = "both",   #cap left, right or both
                       fold = 1.5,   #start & end point
                       variables = ["Salaries"])
df_s = winsor_iqr.fit_transform(df[["Salaries"]])   #Always feature_engine function execute when we run fit_transform

#Inspect the minimum caos and maximum caps
#winsor_iqr.left_tail_caps_, winsor_iqr.right_tail_caps_

#Box plot
sns.boxplot(df_s)

####### Define the model wtih GAUSSIAN method
winsor_gaussian = Winsorizer(capping_method = "gaussian", ########### It dose not give accurate result
                             tail = "both",
                             fold = 3,
                             variables = ["Salaries"])
df_a = winsor_gaussian.fit_transform(df[["Salaries"]])
sns.boxplot(df_a)

####### Define the model with percentail
#Default values
#Right tail: 95th percentail
#Left tail: 5th percentail
winsor_percentile = Winsorizer(capping_method = "quantiles",
                               tail = "both",
                               fold = 0.05,
                               variables = ["Salaries"])
df_p = winsor_percentile.fit_transform(df[["Salaries"]])
sns.boxplot(df_p.Salaries)

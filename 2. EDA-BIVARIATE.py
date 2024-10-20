#######SCATTER PLOT#######
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\Cars.csv")

#Scatter plot   [It describes DIRECTION, STRENGTH = SUBJECTIVE, LINEAR]
import matplotlib.pyplot as plt
plt.scatter(x = df["HP"], y = df["MPG"], color = "blue")
plt.scatter(x = df.HP, y = df.SP)

#Correlation coefficient   [DIRECTION, STRENGTH = OBJECTIVE]
np.corrcoef(df.HP, df.SP)

#Covariance  [DIRECTION]
cov_output = np.cov(df.HP, df.SP)[0, 1]
cov_output

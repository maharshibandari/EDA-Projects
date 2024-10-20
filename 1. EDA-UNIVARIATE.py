import pandas as pd

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\education.csv")

#Measure of central tendency (or) First moment business decision
df.workex.mean()
df.workex.median()
df.workex.mode()

#Meaasure of dispersion (or) Second moment business decision
df.workex.var()
df.workex.std()
range = max(df.workex) - min(df.workex)
range

df.info()

#Skewness (or) Third moment business decision
df.workex.skew()
df.gmat.skew()

#Kurtosis (or) Fourth moment business decision
df.workex.kurt()
df.gmat.kurt()

#######Graphical representation (or) Fifth moment business decision
import matplotlib.pyplot as plt
import numpy as np

df.shape

#Bar plot
plt.bar(height = df.gmat, x = np.arange(1,774,1))

#Histogram
plt.hist(df.gmat)
plt.hist(df.gmat, bins = [600, 640, 680, 710, 740, 780], edgecolor = "black")
plt.hist(df.workex)
plt.hist(df.workex, color = "blue", edgecolor = "black", bins = 7)
help(plt.hist)

#Histogram using seaborn
import seaborn as sns
sns.distplot(df.gmat) #Distribution plot (HISTOGRAM + DENSITY PLOT)
sns.displot(df.gmat) #Histogram using seaborn

#Density plot
sns.kdeplot(df.gmat)
sns.kdeplot(df.gmat, bw = 0.5, fill = True) # bw IS SMOOTHNESS OF PLOT
help(sns.kdeplot)

#Box plot
plt.figure() #plane surface & plot the data we needed
plt.boxplot(df.gmat)
help(plt.boxplot)

#Q-Q plot
from scipy import stats
import pylab
stats.probplot(df.gmat, dist = "norm", plot = pylab) #norm MEANS NORMALLY DISTRIBUTED
plt.show()


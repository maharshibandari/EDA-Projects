############# DATA TRANSFORMATION #############

### Log Transformation ###

'''
Used for exponential distribution
'''
import pandas as pd

# Read data into Python
education = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\education.csv")

import scipy.stats as stats
import pylab

# Checking whether data is normally distributed
stats.probplot(education.gmat, dist = "norm", plot = pylab)

stats.probplot(education.workex, dist = "norm", plot = pylab)

import numpy as np

# Transformation to make workex variable normal
stats.probplot(np.log(education.workex), dist = "norm", plot = pylab)


### Box-Cox Transformation ###

'''
It does not support zero & negative values (ex:- Profits)
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\education.csv")

# Original data
prob = stats.probplot(df.workex, dist = stats.norm, plot = pylab)

# Transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(df.workex)

# Creating axes to draw pots (Two Sub-Plots)
fig, ax = plt.subplots(1, 2) # row & column

# Plotting the original data (non_normal) and fitted data (normal)
sns.distplot(df.workex, hist = False, kde = True,
             kde_kws = {"shade" : True, "linewidth" : 2},
             label = "Non-Normal", color = "green", ax = ax[0]) # kde - density plot; ax[0] = 1st axis(1st plot); shade = True (the outer line is thicker)

#plotting the fitted data
sns.distplot(fitted_data, hist = False, kde = True,
             kde_kws = {"shade" : True, "linewidth" : 2},
             label = "Normal", color = "green", ax = ax[1])

# Adding legends to the subplots
plt.legend(loc = "upper right") # Normal distribution tag

# Rescalinig the sub-plots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data 
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)



### Yeo-Jahnson Transformation ###

'''
We can apply it to our dataset without scaling the data.
It supports zero and negative values.
It does not require the values for each input variable to be strictly positive.
In box-cox transform the input variable has to be postive.
'''
import pandas as pd
from scipy import stats

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\education.csv")

# Original data
probs = stats.probplot(df.workex, dist = stats.norm, plot = pylab)

from feature_engine import transformation

# Setup the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = "workex")
df_tf = tf.fit_transform(df)

# Transformed data
prob = stats.probplot(df_tf.workex, dist = stats.norm, plot = pylab)

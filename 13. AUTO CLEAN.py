############# AUTO CLEAN #############

#pip install py-AutoClean

from AutoClean import AutoClean

df = pd.read_csv(r"FILE PATH")

help(AutoClean)

pipeline = AutoClean(df)
df_clean = pipeline.Output
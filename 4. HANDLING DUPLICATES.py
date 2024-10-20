#######HANDLING DUPLICATES#######
import pandas as pd
df = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\mtcars_dup.csv")
help(df.duplicated)

#Duplicates in row
duplicate = df.duplicated() #Retuurn boolean 
duplicate
sum(duplicate) #No. of duplicate rows
#Parameters
duplicate = df.duplicated(keep = "last")
duplicate 
duplicate = df.duplicated(keep = False)
duplicate

#Removnig duplicates
df1 = df.drop_duplicates() #Return the df with duplicate rows removed
#Parameters
df1 = df.drop_duplicates(keep = "last")
df1 = df.drop_duplicates(keep = False)

#Duplicates in columns
#We can use correlation coefficient values to identify columns with duplicates
cars = pd.read_csv(r"C:\Users\mahar\LOVE\DATA SCIENCE\CODING\DATA SETS\Cars.csv")

#Correlatons Coefficient
cars.corr() ###Which 2 variables have high correlation cooefficient, then remove one of them

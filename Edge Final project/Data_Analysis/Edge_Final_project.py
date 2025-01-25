import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student-scores.csv")
print(df.head)

print(df.describe())

print(df.info())

print(df.isnull().sum())  #To see how many null values there in the data frame

#Gender distribution
plt.figure(figsize=(4,4)) #4,4 states height and width
ax = sns.countplot(data = df, x="gender")
ax.bar_label(ax.containers[0]) #To see total numbers
plt.title("Gender Distribution")
plt.show()

#Gropuby data with a specific columns

gb = df.groupby("part_time_job").agg({"math_score":'mean',"chemistry_score":'mean', "biology_score":'mean'})
print(gb)
#showing that Groupby by heatmap
plt.figure(figsize=(6,4))
sns.heatmap(gb, annot = True) #annot = True To display values inside heatmap
plt.title("Relationship between part time jobs with scores")
plt.show()




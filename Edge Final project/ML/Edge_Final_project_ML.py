import pandas as pd
data = pd.read_csv('marketing_campaign.csv')
print(data)
x = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values
#print(x)
#print(y)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x[:,0] = lb.fit_transform(x[:,0])
print(x)
lb2 = LabelEncoder()
y = lb2.fit_transform(y)
print("\n The respected dependent variable are =", y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(x_test)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scaled_data = scale.fit_transform(x_train)
print(scaled_data)

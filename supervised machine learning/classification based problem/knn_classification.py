#===== DATASET
import pandas as pd
df = pd.read_csv('wheather dataset.csv')


#===== PRE-PROCESSING
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df.loc[:, 'weather'] = le.fit_transform(df.loc[:, 'weather'].values)
df.loc[:, 'temp'] = le.fit_transform(df.loc[:, 'temp'].values)
df.loc[:, 'play'] = le.fit_transform(df.loc[:, 'play'].values)


# ===== DATA SPLIT
x = df.loc[:, ['weather', 'temp']]        # feature   
y = df.loc[:, ['play']]                   # label

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=.3, random_state=42)
print('*********')
import numpy as np
print(x_train)
print('# \n', list(zip(x.loc[:, 'weather'], x.loc[:, 'temp'])))

#===== ML MODEL
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train , y_train)
# y_pred = model.predict(x_test)
# model.fit(list(zip(x.loc[:, 'weather'], x.loc[:, 'temp'])), y.loc[:, 'play'])
y_pred = model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(y_pred)

#===== ANALYSIS 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# print("MSE: ", mean_absolute_error(y_true=y_test, y_pred=y_pred))
# print("MAE: ", mean_squared_error(y_true=y_test, y_pred=y_pred))
# print('r2_score = ', r2_score(y_test,y_pred))
# accuracy = model.score(y_test, y_pred)
# print('accuracy: ',accuracy*100,'%')







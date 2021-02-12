
#===== DATASET
import pandas as pd
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
x = df.loc[:, ['km_driven', 'fuel']]                   
y = df.loc[:, ['selling_price']]        

#===== PRE-PROCESSING
print('before\n', x.head(5))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
le = LabelEncoder() 
x['fuel'] = le.fit_transform(x['fuel']) 
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(x[['fuel']]).toarray())
x = x.join(enc_df)
x = x.drop(['fuel'], axis=1)
print('after\n', x.head(5))


# ===== DATA SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=.3, random_state=42)

# #===== ML MODEL
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model = model.fit(x_train, y_train) 
y_pred = model.predict([0,2])
print(y_pred)

#===== ANALYSIS 
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# print("MSE: ", mean_absolute_error(y_true=y_test, y_pred=y_pred))
# print("MAE: ", mean_squared_error(y_true=y_test, y_pred=y_pred))
# print('r2_score = ', r2_score(y_test,y_pred))
# accuracy = model.score(y_test, y_pred)
# print('accuracy: ',accuracy*100,'%')

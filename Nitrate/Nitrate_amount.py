import pandas as pd
df = pd.read_csv("/content/Nitrate/nitrate 2.csv")
df.columns
df = df.drop_duplicates(subset="SampleDate")
df.shape
df.sort_values(by=['SampleDate'])

df

df['Date'] = pd.to_datetime(df['SampleDate'])

r= pd.date_range(start=df.Date.min(),end=df.Date.max())

import matplotlib.pyplot as plt
plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df.index, df.Result,color='blue')

df2 = df.set_index('Date').reindex(r).interpolate().rename_axis('Date').reset_index()

plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df2.index, df2.Result,color='blue')

df2_mean = df2.resample('M',on='Date').mean()

df2.shape

df2_mean.shape
#Adding more colums makeing it so that it uses the last4 values to predict the next value
df2['LR'] = df2.Result.shift(1)
df2_mean['LR'] = df2_mean.Result.shift(1)

df2['LRR'] = df2.LR.shift(1)
df2_mean['LRR'] = df2_mean.LR.shift(1)
df2_mean['LRRR'] = df2_mean.LRR.shift(1)
df2_mean['LRRRR'] = df2_mean.LRRR.shift(1)

df2 = df2.dropna()
df2_mean = df2_mean.dropna()

df_x_m = df2_mean[['LR','LRR','LRRR','LRRRR']]
df_y_m = df2_mean[['Result']]

df_x_m.values

import matplotlib.pyplot as plt
plt.plot(df2_mean.index, df2_mean.Result)

plt.plot(df2_mean.index, df2_mean.Result)

plt.scatter(df2_mean.index, df2_mean.Result)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_x_m, df_y_m, test_size=0.1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

from sklearn.svm import SVR
regressor = SVR(kernel = 'poly', gamma = "auto",coef0 = .2, C = 100, degree= 2)
regressor.fit(x_train,y_train)

predictions = regressor.predict(x_test)

df_x_m.shape[1]

df_y_m.shape

plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red')
plt.plot(df_x_m.index, model.predict(df_x_m),color='blue')

plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, model.predict(df_x_m),color='blue')

import pickle
file_name = 'model.pkl'
with open(file_name, 'wb') as pkl_file:
  pickle.dump(model, pkl_file)

print("Model Saved!")

#Finding the best Graphs

plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.scatter(df_x_m.index, df_y_m,color='red')
plt.scatter(df_x_m.index, regressor.predict(df_x_m),color='blue')
plt.scatter(df_x_m.index, model.predict(df_x_m),color='green')

plt.title('Nitrate')
plt.ylabel('Nitrate Averages (mg/l)')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, regressor.predict(df_x_m),color='green', label='SVR Model Prediction')
plt.legend(loc='upper right')

plt.title('Nitrate Rates Over Time In California')
plt.ylabel('Average Amount of Nitrate in water (mg/l)')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.scatter(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, regressor.predict(df_x_m),color='green', label='SVR Model Prediction')
plt.legend(loc='upper right')

plt.title('Nitrate Rates Over Time In California')
plt.ylabel('Average Amount of Nitrate in water (mg/l)')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, model.predict(df_x_m),color='green', label='Linear Model Prediction')
plt.legend(loc='upper right')

df2.values

df_x_m

#Code to predict the values of the next 5 years
import numpy as np
years = 2025 - 2020
Last = df_x_m.loc['2018-12-31']
Last_Last = df_x_m.loc['2018-11-30']
Last_Last_Last = df_x_m.loc['2018-10-31']
Last_Last_Last_Last = df_x_m.loc['2018-9-30']
Last = Last[0]
Last_Last = Last_Last[0]
Last_Last_Last = Last_Last_Last[0]
Last_Last_Last_Last = Last_Last_Last_Last[0]
Predicted = []
Prediction = 0
pred = []
Predicted_data=[]
for year in range(0,years):
  for month in range(0,12):
    x_val = [[Last, Last_Last , Last_Last_Last , Last_Last_Last_Last]]
    prediction = regressor.predict(x_val)
    Predicted_data += [[Last , Last_Last, Last_Last_Last , Last_Last_Last_Last],[prediction]]
    pred += [prediction[0]]
    Predicted += [[Last , Last_Last, Last_Last_Last , Last_Last_Last_Last]]
    Last_Last_Last_Last = Last_Last_Last
    Last_Last_Last = Last_Last
    Last_Last = Last
    Last = prediction[0]

Predicted_data

x_vals = []
y_vals = []
flip = 1
for x in Predicted_data:
  if flip == 1:
    x_vals += [x]
    flip = 2
  else:
    y_vals += x
    flip = 1

pred

y_vals

Last_Last = df_x_m.loc['2013-12-31']
Last_Last[0]

plt.title('Nitrate')
plt.ylabel('Nitrate Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(Predicted_data.index,y_vals,color='red' , label='Original Data')
plt.legend(loc='upper right')

df_x_m.loc['2013-11-30'].values
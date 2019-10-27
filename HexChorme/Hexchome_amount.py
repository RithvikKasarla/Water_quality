import pandas as pd

df = pd.read_csv("/content/HexChrom.csv")

df.columns

df = df.drop_duplicates(subset="SampleDate")

df.shape

df.sort_values(by=['SampleDate'])

df

df['Date'] = pd.to_datetime(df['SampleDate'])

r= pd.date_range(start=df.Date.min(),end=df.Date.max())

import matplotlib.pyplot as plt
plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df.index, df.Result,color='blue')

df2 = df.set_index('Date').reindex(r).interpolate().rename_axis('Date').reset_index()

plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df2.index, df2.Result,color='blue')

df2_mean = df2.resample('M',on='Date').mean()

df2.shape

df2_mean.shape

df2['LR'] = df2.Result.shift(1)
df2_mean['LR'] = df2_mean.Result.shift(1)

df2['LRR'] = df2.LR.shift(1)
df2_mean['LRR'] = df2_mean.LR.shift(1)

df2 = df2.dropna()
df2_mean = df2_mean.dropna()

df_x_m = df2_mean[['LR','LRR']]
df_y_m = df2_mean[['Result']]

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

plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red')
plt.plot(df_x_m.index, model.predict(df_x_m),color='blue')


plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, model.predict(df_x_m),color='blue')

import pickle
file_name = 'HexChrom.pkl'
with open(file_name, 'wb') as pkl_file:
  pickle.dump(model, pkl_file)

print("Model Saved!")

plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.scatter(df_x_m.index, df_y_m,color='red')
plt.scatter(df_x_m.index, regressor.predict(df_x_m),color='blue')
plt.scatter(df_x_m.index, model.predict(df_x_m),color='green')

plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages (ug/l)')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, regressor.predict(df_x_m),color='green', label='SVR Model Prediction')
plt.legend(loc='upper right')


plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages (ug/l)')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.scatter(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, regressor.predict(df_x_m),color='green', label='SVR Model Prediction')
plt.legend(loc='upper right')

plt.title('Hexavalent Chromium')
plt.ylabel('Hexavalent Chromium Averages')
plt.xlabel('Year')
plt.rcParams['figure.figsize'] = [10, 5]
plt.plot(df_x_m.index, df_y_m,color='red' , label='Original Data')
plt.plot(df_x_m.index, model.predict(df_x_m),color='green', label='Linear Model Prediction')
plt.legend(loc='upper right')

df_x_m.shape
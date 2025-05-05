import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

st.title('Prediksi Harga Mobil Bekas UK')

st.header('**Dataset yang digunakan**')
df = pd.read_csv('https://raw.githubusercontent.com/M-Fathul/startingML/refs/heads/master/cars_dataset.csv', sep=',')

df.dropna(inplace=True)
df = df[df['engineSize'] != 0]
df = df[df['tax'] != 0]
df.drop_duplicates(inplace=True)

df = df[df['transmission'] != 'Other']
df = df[(df['fuelType'] != 'Other') & (df['fuelType'] != 'Electric')]
df = df[df['year'] > 2000]
df = df[df['mileage'] < 200000]
df = df[df['tax'] < 500]
df = df[(df['mpg'] < 85) & (df['mpg'] > 20)]
df = df[df['engineSize'] < 6]
df['model'] = df['model'].str.lstrip()

st.dataframe(df)
dfprep = df.copy()

labeling = LabelEncoder()
scaler = MinMaxScaler(copy = True, feature_range = (0,1))

numerical_features = dfprep.select_dtypes(exclude=['object']).columns
dfprep[numerical_features] = scaler.fit_transform(dfprep[numerical_features])

kolomkategori = df.select_dtypes(include=['object']).columns.tolist()
labeling.fit(pd.concat([df[col] for col in kolomkategori]))

x = dfprep.drop('price', axis=1)
y = dfprep['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


with st.sidebar:
  Make = st.selectbox('Make', df['Make'].unique())
  model = st.selectbox('Model', df['model'].unique())
  year = st.slider('Tahun Beli', df['year'].min(), df['year'].max())
  transmission = st.selectbox('Transmisi', df['transmission'].unique())
  fuelType = st.selectbox('Bahan Bakar', df['fuelType'].unique())
  engineSize = st.slider('Ukuran Mesin', df['engineSize'].min(), df['engineSize'].max())
  mileage = st.slider('Jarak Tempuh', df['mileage'].min(), df['mileage'].max())
  mpg = st.slider('Kapasitas Bahan Bakar', df['mpg'].min(), df['mpg'].max())
  tax = st.slider('Pajak', df['tax'].min(), df['tax'].max())
  price = 0
  prediksi = 0
  
  data = {'model': model, 'year': year, 'price': price, 'transmission': transmission, 'mileage': mileage, 'fuelType': fuelType, 'tax': tax, 'mpg': mpg, 'engineSize': engineSize, 'Make': Make,}
new_data = pd.DataFrame(data, index=[0])
new_data_prep = new_data.copy()
numerical_features = new_data_prep.select_dtypes(exclude=['object']).columns
new_data_prep[numerical_features] = scaler.transform(new_data_prep[numerical_features])
for col in new_data_prep.select_dtypes(include=['object']):
  new_data_prep[col] = labeling.transform(new_data_prep[col])
new_data_prep = new_data_prep.drop('price', axis=1)


if st.button('Prediksi'):
  st.dataframe(new_data_prep)
  st.dataframe(new_data)
  st.write('Prediksi Harga Mobil Bekas: ', prediksi)
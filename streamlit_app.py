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
st.header('**Tentang Aplikasi**')
st.markdown('Aplikasi ini bertujuan untuk memprediksi harga mobil bekas di UK berdasarkan beberapa fitur yang ada di dataset menerapkan Random Forest dalam prediksi harga mobil bekas di pasar Inggris menggunakan dataset yang mencakup berbagai fitur kendaraan seperti merek, model, tahun pembuatan, mileage, ukuran mesin, dan jenis bahan bakar. Melalui penelitian ini, diharapkan dapat ditemukan model yang dapat memberikan prediksi harga yang lebih akurat dan efisien, serta lebih mudah diakses oleh konsumen dan penjual.Selain itu, penelitian ini juga berfokus pada analisis faktor-faktor yang paling berpengaruh dalam menentukan harga mobil bekas. Dengan mengetahui faktor-faktor utama yang mempengaruhi harga, penelitian ini dapat memberikan wawasan yang lebih dalam mengenai dinamika pasar mobil bekas di Inggris. Hal ini akan bermanfaat tidak hanya bagi pembeli dan penjual, tetapi juga bagi perusahaan otomotif, dealer mobil bekas, serta platform jual beli mobil yang semakin berkembang.')

st.header('**Dataset yang digunakan**')
df = pd.read_csv('https://raw.githubusercontent.com/M-Fathul/startingML/refs/heads/master/cars_dataset.csv', sep=',')

st.write("")

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
for col in kolomkategori:
  dfprep[col] = labeling.transform(dfprep[col])

x = dfprep.drop('price', axis=1)
y = dfprep['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
modelRandomForest = RandomForestRegressor()
modelRandomForest.fit(x_train, y_train)

with st.sidebar:
  if 'Make' not in st.session_state:
    st.session_state.Make = df['Make'].unique()[0]
  if 'model' not in st.session_state:
    st.session_state.model = df[df['Make'] == st.session_state.Make]['model'].unique()[0]
  st.session_state.Make = st.selectbox('Make', df['Make'].unique(), key='Make_select')
  filtered_models = df[df['Make'] == st.session_state.Make]['model'].unique()
  st.session_state.model = st.selectbox('Model', filtered_models, key='model_select')
  year = st.number_input('Tahun Beli', df['year'].min(), df['year'].max(), 2017)
  transmission = st.selectbox('Transmisi', df['transmission'].unique())
  fuelType = st.selectbox('Bahan Bakar', df['fuelType'].unique())
  engineSize = st.number_input('Ukuran Mesin', df['engineSize'].min(), df['engineSize'].max(), 1.4)
  mileage = st.number_input('Jarak Tempuh', df['mileage'].min(), df['mileage'].max(), 15735)
  mpg = st.number_input('Kapasitas Bahan Bakar', df['mpg'].min(), df['mpg'].max(), 55.4)
  tax = st.number_input('Pajak', df['tax'].min(), df['tax'].max(), 150.0)
  price = 0
  prediksi = 0
  if st.button('prediksi harga'):
    new_data = pd.DataFrame({
      'model': [st.session_state.model],
      'year': [year],
      'price': [price],
      'transmission': [transmission],
      'mileage': [mileage],
      'fuelType': [fuelType],
      'tax': [tax],
      'mpg': [mpg],
      'engineSize': [engineSize],
      'Make': [st.session_state.Make],
    })
    new_data_prep = new_data.copy()
    numerical_features = new_data_prep.select_dtypes(exclude=['object']).columns
    new_data_prep[numerical_features] = scaler.transform(new_data_prep[numerical_features])
    for col in new_data_prep.select_dtypes(include=['object']):
      new_data_prep[col] = labeling.transform(new_data_prep[col])
    new_data_prep = new_data_prep.drop('price', axis=1)
    y_pred_scaled = modelRandomForest.predict(new_data_prep)
    new_data_prep.insert(2, 'price', y_pred_scaled)
    numerical_features = new_data.select_dtypes(exclude=['object']).columns
    new_data[numerical_features] = scaler.inverse_transform(new_data_prep[numerical_features])
    prediksi = int(new_data['price'])
    st.write('Prediksi Harga Mobil Bekas: ', + prediksi)

import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

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
st.dataframe(df)

with st.sidebar:
  Status = st.selectbox('Status', (0, 1, 2, 3))
  Kelamin = st.selectbox('Jenis kelamin', (0, 1))
  Usia = st.number_input('Usia', value=None, placeholder="...")
  Memiliki_Mobil = st.number_input('Jumlah mobil yang dimiliki', value=None, placeholder="...")
  Penghasilan = st.number_input('penghasilan juta perthaun', value=None, placeholder="...")
  data = {'Status': Status, 'Kelamin': Kelamin, 'Usia': Usia, 'Memiliki_Mobil': Memiliki_Mobil, 'Penghasilan': Penghasilan}
  input_df = pd.DataFrame(data, index=[0])
input_df

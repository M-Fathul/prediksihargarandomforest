import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas UK",
    page_icon="🚘",
    layout="wide"
)

st.title('Prediksi Harga Mobil Bekas UK')
st.header('**Tentang Aplikasi**')
st.write("""
    Pasar mobil bekas di Inggris merupakan salah satu pasar yang sangat besar dan dinamis. Setiap tahun, ribuan mobil bekas diperdagangkan, baik melalui dealer mobil, lelang, maupun secara langsung antara penjual dan pembeli. Namun, salah satu tantangan terbesar yang dihadapi oleh konsumen dan penjual dalam pasar ini adalah penentuan harga yang adil dan realistis. seringkali penentuan harga mobil bekas dilakukan secara subjektif dan berdasarkan pengalaman atau intuisi, yang tentu saja dapat menghasilkan harga yang tidak akurat. Kondisi ini sering kali merugikan pembeli yang ingin mendapatkan harga yang wajar, atau penjual yang ingin menjual mobil mereka dengan harga yang optimal. Oleh karena itu, diperlukan sebuah metode yang lebih objektif dan sistematis untuk memprediksi harga mobil bekas berdasarkan data yang ada
    
    Aplikasi ini bertujuan untuk memprediksi harga mobil bekas di UK berdasarkan beberapa fitur yang ada di dataset menerapkan algoritma Random Forest dalam prediksi harga mobil bekas di pasar Inggris menggunakan dataset yang mencakup berbagai fitur kendaraan seperti merek, model, tahun pembuatan, mileage, ukuran mesin, dan jenis bahan bakar. Melalui aplikasi ini, diharapkan dapat memanfaatkan model yang dapat memberikan prediksi harga yang akurat dan efisien, serta lebih mudah diakses oleh konsumen dan penjual. Hal ini akan bermanfaat tidak hanya bagi pembeli dan penjual, tetapi juga bagi perusahaan otomotif, dealer mobil bekas, serta platform jual beli mobil yang semakin berkembang.
    """)
st.divider()
st.header('**Pengembangan Aplikasi**')
st.write("""
     Pada aplkasi ini terdapat model dengan metode regresi dengan algoritma Random Forest, yang dimana model tersebut dilatih menggunakan dataset yang berisi beberapa kolom yang memberikan informasi detail tentang mobil bekas:
    """)
df = pd.read_csv('https://raw.githubusercontent.com/M-Fathul/startingML/refs/heads/master/cars_dataset.csv', sep=',')
with st.expander("Dataset"):
  st.dataframe(df)
st.text("""
    Berikut penjelasan fitur yanga ada pada Dataset:\nMake\t\t\t: Merek mobil \nmodel\t\t\t: Model spesifik dari merek tersebut \nyear\t\t\t: Tahun produksi mobil.\nprice\t\t\t: Harga jual mobil, dalam Poundsterling.\ntransmission\t: Tipe transmisi\nmileage\t\t: Jarak tempuh mobil, kemungkinan dalam satuan miles.\nfuelType\t\t: Jenis bahan bakar.\ntax\t\t\t\t: Pajak jalan tahunan dalam Poundsterling.\nmpg\t\t\t: Miles per gallon (efisiensi konsumsi bahan bakar per satuan jarak miles).\nengineSize\t\t: Ukuran mesin mobil dalam liter.
    """)
st.subheader("Informasi Dataset")
col1, col2, col3 = st.columns(3)
with col1:
  st.markdown("""
  Tipe Data Objek:
  :green-badge[Make] :orange-badge[model] :gray-badge[transmission] :blue-badge[fuelType]
  """)
with col2:
  st.markdown("""
  Tipe Data Numerik:
  :green-badge[year] :orange-badge[price] :gray-badge[mileage] :red-badge[tax] :violet-badge[mpg] :blue-badge[engineSize]
  """)
with col3:
  st.metric("Jumlah Data", df.shape[0])

st.subheader("Eksplorasi Data")
colheatmap, penjelasanheatmap = st.columns(2)
with colheatmap:
  kor = df.select_dtypes(exclude=['object']).corr()
  fig, ax = plt.subplots()
  sns.heatmap(kor, annot=True, cmap="coolwarm", ax=ax)
  plt.title("Heatmap Korelasi")
  st.pyplot(fig)
with penjelasanheatmap:
  st.text("""
  Heatmap ini menunjukkan hubungan linier antara berbagai fitur (variabel) dalam dataset mobil bekas. Nilai korelasi berkisar dari -1 hingga 1:
  1 (merah gelap)\t: Jika satu variabel meningkat, yang lain juga cenderung meningkat.
  0 (abu-abu/putih)\t: tidak berkorelasi
  -1 (biru gelap)\t\t: Jika satu variabel meningkat, yang lain cenderung menurun. sangat positif
  
  Dari heatmap ini, faktor-faktor seperti usia mobil (year) dan jarak tempuh (mileage) adalah prediktor kuat untuk harga (price) mobil bekas. Semakin tua dan semakin tinggi jarak tempuh, semakin rendah harganya. Selain itu, ukuran mesin (engineSize) juga memiliki pengaruh positif pada harga dan cenderung berkorelasi negatif dengan efisiensi bahan bakar (mpg) serta sedikit positif dengan pajak (tax).
  """)
st.write("Pada dataset terdapat fitur Make yang merupakan merek mobil, dan model yang merupakan model spesifik dari merek tersebut. Dengan begitu aplikasi ini bisa memprediksi harga mobil dari merek-merek berserta model yang ada pada dataset berikut distribusinya:")
dftes = pd.DataFrame(
    {
        "Make": ["Roadmap", "Roadmap", "Roadmap", "Extras", "Extras", "Extras", "Issues"],
        "model": ["A", "B", "C", "D", "E", "F", "G"],
    }
)
coldistribusi, penjelasandistribusi = st.columns(2)
with coldistribusi:
  st.bar_chart(dftes, x="Make", color="model")

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

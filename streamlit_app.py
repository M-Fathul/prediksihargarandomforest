import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Page Configuration
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas UK",
    page_icon="🚘",
    layout="wide"
)

# 2. Data Loading Function
@st.cache_data
def load_dataset():
    local_csv = os.path.join(os.path.dirname(__file__), 'cars_dataset.csv')
    if os.path.exists(local_csv):
        df_data = pd.read_csv(local_csv, sep=',')
    else:
        df_data = pd.read_csv('https://raw.githubusercontent.com/M-Fathul/startingML/refs/heads/master/cars_dataset.csv', sep=',')
    
    # Strip leading/trailing whitespaces from string columns
    for col in df_data.select_dtypes(include=['object']).columns:
        df_data[col] = df_data[col].astype(str).str.strip()
        
    return df_data

df = load_dataset()

# 3. Model Training & Pipeline Setup Function
@st.cache_resource
def train_car_model(data_frame):
    # Data Cleaning Pipeline
    df_clean = data_frame.dropna().copy()
    df_clean = df_clean[(df_clean['engineSize'] != 0) & (df_clean['tax'] != 0)]
    df_clean = df_clean.drop_duplicates()
    
    # Outlier Removal on numeric features
    for col in df_clean.select_dtypes(exclude=['object']).columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    dfprep = df_clean.copy()
    
    # Label Encoding per categorical column
    label_encoders = {}
    cat_cols = dfprep.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        dfprep[col] = le.fit_transform(dfprep[col])
        label_encoders[col] = le

    # Numerical feature scaling and target scaling
    feature_num_cols = [c for c in dfprep.select_dtypes(exclude=['object']).columns if c != 'price']
    scaler_X = MinMaxScaler()
    dfprep[feature_num_cols] = scaler_X.fit_transform(dfprep[feature_num_cols])

    scaler_y = MinMaxScaler()
    dfprep[['price']] = scaler_y.fit_transform(dfprep[['price']])

    # Feature and Target Split
    X = dfprep.drop('price', axis=1)
    y = dfprep['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    modelRandomForest = RandomForestRegressor(n_estimators=100, random_state=42)
    modelRandomForest.fit(X_train, y_train)

    # Model Evaluation
    y_pred_scaled = modelRandomForest.predict(X_test)
    r2_val = r2_score(y_test, y_pred_scaled)
    mse_val = mean_squared_error(y_test, y_pred_scaled)

    return {
        'df_clean': df_clean,
        'modelRandomForest': modelRandomForest,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'label_encoders': label_encoders,
        'feature_num_cols': feature_num_cols,
        'feature_order': X.columns.tolist(),
        'X_test': X_test,
        'y_test': y_test,
        'r2': r2_val,
        'mse': mse_val
    }

pipeline = train_car_model(df)
modelRandomForest = pipeline['modelRandomForest']
scaler_X = pipeline['scaler_X']
scaler_y = pipeline['scaler_y']
label_encoders = pipeline['label_encoders']

# 4. Header & Overview Section
st.title('Prediksi Harga Mobil Bekas UK')
st.header('**Tentang Aplikasi**')
st.write("""
    Pasar mobil bekas di Inggris merupakan salah satu pasar yang sangat besar dan dinamis. Setiap tahun, ribuan mobil bekas diperdagangkan, baik melalui dealer mobil, lelang, maupun secara langsung antara penjual dan pembeli. Namun, salah satu tantangan terbesar yang dihadapi oleh konsumen dan penjual dalam pasar ini adalah penentuan harga yang adil dan realistis. seringkali penentuan harga mobil bekas dilakukan secara subjektif dan berdasarkan pengalaman atau intuisi, yang tentu saja dapat menghasilkan harga yang tidak akurat. Kondisi ini sering kali merugikan pembeli yang ingin mendapatkan harga yang wajar, atau penjual yang ingin menjual mobil mereka dengan harga yang optimal. Oleh karena itu, diperlukan sebuah metode yang lebih objektif dan sistematis untuk memprediksi harga mobil bekas berdasarkan data yang ada
    
    Aplikasi ini bertujuan untuk memprediksi harga mobil bekas di UK berdasarkan beberapa fitur yang ada di dataset menerapkan algoritma Random Forest dalam prediksi harga mobil bekas di pasar Inggris menggunakan dataset yang mencakup berbagai fitur kendaraan seperti merek, model, tahun pembuatan, mileage, ukuran mesin, dan jenis bahan bakar. Melalui aplikasi ini, diharapkan dapat memanfaatkan model yang dapat memberikan prediksi harga yang akurat dan efisien, serta lebih mudah diakses oleh konsumen dan penjual. Hal ini akan bermanfaat tidak hanya bagi pembeli dan penjual, tetapi juga bagi perusahaan otomotif, dealer mobil bekas, serta platform jual beli mobil yang semakin berkembang.
    """)

# 5. Prediction Dialog Modal
@st.dialog("Masukan Spesifikasi Mobil")
def prediksi():
  predik1, predik2 = st.columns(2)
  makes = sorted(df['Make'].unique())
  with predik1:
    if 'Make' not in st.session_state or st.session_state.Make not in makes:
      st.session_state.Make = makes[0]
    
    selected_make = st.selectbox('Make', makes, index=makes.index(st.session_state.Make), key='Make_select')
    st.session_state.Make = selected_make
    
    filtered_models = sorted(df[df['Make'] == selected_make]['model'].unique())
    if 'model' not in st.session_state or st.session_state.model not in filtered_models:
      st.session_state.model = filtered_models[0]
      
    selected_model = st.selectbox('Model', filtered_models, index=filtered_models.index(st.session_state.model), key='model_select')
    st.session_state.model = selected_model

    year = st.number_input('Tahun Beli', int(df['year'].min()), int(df['year'].max()), 2017)
    transmissions = sorted(df['transmission'].unique())
    transmission = st.selectbox('Transmisi', transmissions)
    fuel_types = sorted(df['fuelType'].unique())
    fuelType = st.selectbox('Bahan Bakar', fuel_types)
  with predik2:
    engineSize = st.number_input('Ukuran Mesin', float(df['engineSize'].min()), float(df['engineSize'].max()), 1.4, step=0.1)
    mileage = st.number_input('Jarak Tempuh', int(df['mileage'].min()), int(df['mileage'].max()), 15735)
    mpg = st.number_input('Kapasitas Bahan Bakar', float(df['mpg'].min()), float(df['mpg'].max()), 55.4, step=0.1)
    tax = st.number_input('Pajak', float(df['tax'].min()), float(df['tax'].max()), 150.0, step=10.0)

    if st.button("prediksi harga"):
      # Create input DataFrame with exact column names expected
      new_data = pd.DataFrame({
          'Make': [selected_make],
          'model': [selected_model],
          'year': [year],
          'transmission': [transmission],
          'mileage': [mileage],
          'fuelType': [fuelType],
          'tax': [tax],
          'mpg': [mpg],
          'engineSize': [engineSize]
      })

      new_data_prep = new_data.copy()

      # Apply LabelEncoding
      for col in label_encoders:
          le = label_encoders[col]
          if new_data_prep[col].iloc[0] in le.classes_:
              new_data_prep[col] = le.transform(new_data_prep[col])
          else:
              new_data_prep[col] = 0

      # Apply MinMaxScaler on numerical features
      num_cols = pipeline['feature_num_cols']
      new_data_prep[num_cols] = scaler_X.transform(new_data_prep[num_cols])

      # Reorder columns to match model training feature order
      new_data_prep = new_data_prep[pipeline['feature_order']]

      # Predict scaled price and inverse transform
      y_pred_scaled = modelRandomForest.predict(new_data_prep)
      y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
      prediksi_val = float(y_pred_unscaled[0][0])

      st.success(f"Prediksi Harga Mobil Bekas: £{prediksi_val:,.2f}")
      st.metric("Prediksi Harga (Nominal GBP)", f"£{int(round(prediksi_val)):,}")

if st.button("Prediksi Harga Mobil Bekas Anda"):
  prediksi()

st.divider()
st.header('**Pengembangan Aplikasi**')
st.write("""
     Pada aplkasi ini terdapat model dengan metode regresi dengan algoritma Random Forest, yang dimana model tersebut dilatih menggunakan dataset yang berisi beberapa kolom yang memberikan informasi detail tentang mobil bekas:
    """)

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
st.write("Pada dataset terdapat fitur Make yang merupakan merek mobil, dan model yang merupakan model spesifik dari merek tersebut. Dengan begitu aplikasi ini bisa mempredict harga mobil dari merek-merek berserta model yang ada pada dataset berikut distribusinya:")
st.html("""
    <h3>\nMerek mobil yang ada dalam dataset dan distribusinya:</h3>
    """)
jumlahmerek = df['Make'].value_counts()
st.bar_chart(jumlahmerek)

def show_brand_chart(brand_name, title_html):
    st.html(title_html)
    brand_df = df[df['Make'].str.lower() == brand_name.lower()]
    counts = brand_df['model'].value_counts()
    st.bar_chart(counts, horizontal=True)

show_brand_chart('audi', "<h3>\nModel pada merek Audi yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('BMW', "<h3>\nModel pada merek BMW yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('Ford', "<h3>\nModel pada merek Ford yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('vw', "<h3>\nModel pada merek VW yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('toyota', "<h3>\nModel pada merek Toyota yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('skoda', "<h3>\nModel pada merek Skoda yang ada dalam dataset dan distribusinya:</h3>")
show_brand_chart('Hyundai', "<h3>\nModel pada merek Hyundai yang ada dalam dataset dan distribusinya:</h3>")

spek1, spek2 = st.columns(2)
with spek1:
  st.html("<h3>\nDistribusi spesifikasi tahun mobil:</h3>")
  jumlahyear = df['year'].value_counts()
  st.bar_chart(jumlahyear)
  st.html("<h3>\nDistribusi spesifikasi mileage mobil:</h3>")
  jumlahmileage = df['mileage'].value_counts()
  st.line_chart(jumlahmileage)
  st.html("<h3>\nDistribusi spesifikasi mpg mobil:</h3>")
  jumlahmpg = df['mpg'].value_counts()
  st.line_chart(jumlahmpg)
  st.html("<h3>\nDistribusi spesifikasi transmisi mobil:</h3>")
  jumlahtransmission = df['transmission'].value_counts()
  st.bar_chart(jumlahtransmission)
with spek2:
  st.html("<h3>\nDistribusi spesifikasi harga mobil:</h3>")
  jumlahprice = df['price'].value_counts()
  st.line_chart(jumlahprice)
  st.html("<h3>\nDistribusi spesifikasi pajak mobil:</h3>")
  jumlahtax = df['tax'].value_counts()
  st.line_chart(jumlahtax)
  st.html("<h3>\nDistribusi spesifikasi ukuran mesin mobil:</h3>")
  jumlahenginesize = df['engineSize'].value_counts()
  st.bar_chart(jumlahenginesize)
  st.html("<h3>\nDistribusi spesifikasi jenis bahan bakar mobil:</h3>")
  jumlahfueltype = df['fuelType'].value_counts()
  st.bar_chart(jumlahfueltype)

st.subheader("Pembersihan Data")
st.write("""
    Untuk memastikan akurasi dari aplikasi ini diperlukan pembersihan data sebelum digunakan untuk melatih model yang digunakan pada aplikasi ini, tahapan pembersihan antara lain:
    """)

# Session state flags for interactive cleaning buttons
if 'clean_null_zero' not in st.session_state:
    st.session_state.clean_null_zero = False
if 'clean_duplicates' not in st.session_state:
    st.session_state.clean_duplicates = False
if 'clean_outliers' not in st.session_state:
    st.session_state.clean_outliers = False

st.html("<h3>\n1. Menghapus data yang memiliki nilai null dan 0:</h3>")
hapus1, hapus2 = st.columns(2)
with hapus1:
  st.write("""Pada dataset terdapat nilai 0 yang merupakan anomali pada variabel engineSize dan tax, karena engineSize merupakan sebuah ukuran yang tidak dapat bernilai 0 dan tax merupakan sebuah nilai yang bisa melambangkan kondisi bekas jika bernilai 0 maka bukanlah mmobil bekas. Oleh karena itu, nilai 0 pada engineSize dan tax dihapus pada dataset.""")
  st.code("""
  df = df.dropna()
  df = df[df['engineSize'] != 0]
  df = df[df['tax'] != 0]
  """, language="python")
  if st.button("Hapus Data null dan 0"):
    st.session_state.clean_null_zero = True

with hapus2:
  if st.session_state.clean_null_zero:
    df_temp1 = df.dropna()
    df_temp1 = df_temp1[(df_temp1['engineSize'] != 0) & (df_temp1['tax'] != 0)]
    st.metric("Jumlah Data setelah null dan 0 berhasil dihapus", df_temp1.shape[0])

st.html("<h3>\n2. Menghapus data duplikat:</h3>")
duplikat1, duplikat2 = st.columns(2)
with duplikat1:
  st.write("""Pada dataset terdapat data duplikat, sehingga data duplikat dihapus pada dataset.""")
  st.code("""
  df = df.drop_duplicates()
  """, language="python")
  if st.button("Hapus Data Duplikat"):
    st.session_state.clean_duplicates = True

with duplikat2:
  if st.session_state.clean_duplicates:
    df_temp2 = df.dropna()
    df_temp2 = df_temp2[(df_temp2['engineSize'] != 0) & (df_temp2['tax'] != 0)].drop_duplicates()
    st.metric("Jumlah Data setelah duplikat berhasil dihapus", df_temp2.shape[0])

st.html("<h3>\n3. Menghapus data outlier:</h3>")
outlier1, outlier2 = st.columns(2)
with outlier1:
  st.write("""Pada dataset terdapat data outlier, sehingga data outlier dihapus pada dataset.""")
  st.code("""
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
  """, language="python")
  if st.button("Hapus Data Outlier"):
    st.session_state.clean_outliers = True

with outlier2:
  if st.session_state.clean_outliers:
    st.metric("Jumlah Data setelah outlier berhasil dihapus", pipeline['df_clean'].shape[0])

with st.expander("Dataset yang digunakan untuk melatih model"):
  st.dataframe(pipeline['df_clean'])

st.subheader("Latihan Model")
model1, model2 = st.columns(2)
with model1:
  st.write("""
  Sebelum dilakukan pelatihan perlu dilakukan transformasi data agar data dapat digunakan untuk melatih mesin lebih akurat.
  transformasi data dilakukan dengan menggunakan
  - MinMaxScaler pada data yang bertipe numerik. MinMaxScaler merubah data menjadi nilai antara 0 dan 1.
  - LabelEncoder pada data yang bertipe kategorikal. LabelEncoder merubah data menjadi angka.
  
  Pada Dataset juga dilakukan pemisahan kolom yang akan digunakan sebagai fitur dan target, yaitu kolom price sebagai target dan seluruh kolom lainnya sebagai fitur.
  """)
with model2:
  st.write("""Kemudian data akan di split menjadi 2 bagian yaitu training dan testing data. Training data digunakan untuk melatih model dan testing data digunakan untuk mengevaluasi akurasi model yang telah dilatih. pembagian antara training dan testing akan berukuran 80 sebagai training dan 20 sebagai testing. Pada aplikasi ini prediksi harga dihasilkan dari model Random Forest Regressor.""")
  st.code("""
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor

  X = dfprep.drop('price', axis=1)
  y = dfprep['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  """, language="python")

st.subheader("Evaluasi Model")
st.write("""
    Untuk memastikan akurasi dari aplikasi ini diperlukan evaluasi model, dengan menggunakan metrik R2 (R-Squared) dan Mean Squared Error (MSE) sebagai pengukur akurasi algoritma Random Forest Regressor.
    """)
eval1, eval2 = st.columns(2)
with eval1:
  st.write("""
  R2 (R-Squared) adalah metrik yang digunakan untuk mengukur seberapa baik model dapat menjawab pertanyaan prediksi. Nilai R2 berbentuk sekala dari 0 hingga 1. Nilai 1 menunjukkan bahwa model dapat menjawab pertanyaan prediksi dengan sempurna, sedangkan nilai 0 menunjukkan bahwa model tidak dapat menjawab pertanyaan prediksi dengan sempurna. Berikut akurasi model pada aplikasi ini:""")
  st.metric("Nilai R2", round(pipeline['r2'], 4))
with eval2:
  st.write("""
  Mean Squared Error (MSE) adalah metrik yang digunakan untuk mengukur seperbaikan model dapat menjawab pertanyaan prediksi. Nilai MSE merupakan nilai yang dihitung dengan rumus berikut: MSE = (1/n) * Σ(y_pred - y_true)^2. Nilai MSE yang mendekati 0 menunjukkan bahwa model dapat menjawab pertanyaan prediksi dengan lebih baik. Berikut akurasi model pada aplikasi ini:""")
  st.metric("Nilai MSE", round(pipeline['mse'], 6))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Analisis Data Sepeda",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    main_data = pd.read_csv("../Dashboard/main_data.csv")
    main_data["dteday"] = main_data["dteday"].astype(str)
    return main_data

main_data = load_data()

# Navigasi dengan Button Sidebar
st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Analisis RFM", "Distribusi Data", "Korelasi Antar Variabel", "Tren Peminjaman", "Pengaruh Cuaca", "Lonjakan Penggunaan Sepeda", "Kesimpulan"])

if page == "Beranda":
    st.title("📊 Dashboard Analisis Data Sepeda 🚴")
    st.write("Selamat datang di dashboard ini. Silakan pilih menu di sidebar untuk eksplorasi data lebih lanjut.")
    
    st.subheader("🔍 Sekilas Data")
    st.dataframe(main_data.head())

    st.subheader("📌 Statistik Data")
    st.write(main_data.describe())

elif page == "Analisis RFM":
    st.title("📈 Analisis RFM")
    last_date = main_data["dteday"].max()
    rfm = main_data.groupby("instant").agg({
        "dteday": lambda x: (last_date - x.max()).days,
        "instant": "count", 
        "cnt": "sum" 
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    st.dataframe(rfm)

    st.subheader("📊 Distribusi RFM")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(rfm["Recency"], bins=20, color="skyblue", edgecolor="black")
    ax[0].set_title("Distribusi Recency")
    ax[1].hist(rfm["Frequency"], bins=20, color="orange", edgecolor="black")
    ax[1].set_title("Distribusi Frequency")
    ax[2].hist(rfm["Monetary"], bins=20, color="green", edgecolor="black")
    ax[2].set_title("Distribusi Monetary")
    st.pyplot(fig)

elif page == "Distribusi Data":
    st.title("📊 Distribusi Data")
    num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for i, col in enumerate(num_cols):
        sns.histplot(main_data[col], bins=30, kde=True, ax=ax[i//3, i%3])
        ax[i//3, i%3].set_title(f'Distribusi {col}')
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Korelasi Antar Variabel":
    st.title("📊 Korelasi Antar Variabel")
    plt.figure(figsize=(10, 6))
    sns.heatmap(main_data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

elif page == "Tren Peminjaman":
    st.title("📈 Tren Peminjaman Sepeda")
    fig = px.line(main_data, x="mnth", y="cnt", title="Tren Peminjaman Sepeda per Bulan", labels={"mnth": "Bulan", "cnt": "Jumlah Peminjaman"})
    st.plotly_chart(fig)

elif page == "Pengaruh Cuaca":
    st.title("🌦️ Pengaruh Cuaca terhadap Peminjaman")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="weathersit", y="cnt", data=main_data)
    plt.xticks([0, 1, 2, 3], ['Cerah', 'Mendung', 'Hujan Ringan', 'Hujan Lebat'])
    plt.title("Pengaruh Cuaca terhadap Peminjaman Sepeda")
    st.pyplot(plt)

elif page == "Lonjakan Penggunaan Sepeda":
    st.title("📈 Lonjakan Penggunaan Sepeda")
    fig = px.line(main_data, x="dteday", y="cnt", title="Tren Penggunaan Sepeda Harian", labels={"dteday": "Tanggal", "cnt": "Jumlah Peminjaman"})
    st.plotly_chart(fig)

elif page == "Kesimpulan":
    st.title("📌 Kesimpulan")
    st.markdown("**1. Kondisi cuaca sangat mempengaruhi jumlah peminjaman sepeda.** Cuaca cerah memiliki jumlah peminjaman tertinggi, sedangkan hujan dan berkabut menunjukkan penurunan signifikan.")
    st.markdown("**2. Bulan Juni hingga November memiliki peminjaman yang tinggi.** Kemungkinan karena musim panas, sementara Desember dan Januari mengalami penurunan karena musim dingin atau liburan.")

if __name__ == "__main__":
    st.sidebar.success("Pilih halaman di atas")

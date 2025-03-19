import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Data Sepeda",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    merge_cleaned = pd.read_csv("merge_cleaned.csv", parse_dates=["dteday"])
    return merge_cleaned
merge_cleaned = load_data()

# Sidebar Navigasi
st.sidebar.header("Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Analisis RFM", "Analisis Geospasial", "Clustering"])

if page == "Beranda":
    st.title("📊 Dashboard Analisis Data Sepeda 🚴")
    st.write("Selamat datang di dashboard ini. Silakan pilih menu di sidebar untuk eksplorasi data lebih lanjut.")
    
    # Menampilkan preview data
    st.subheader("🔍 Sekilas Data")
    st.dataframe(merge_cleaned.head())

    # Statistik umum
    st.subheader("📌 Statistik Data")
    st.write(merge_cleaned.describe())

elif page == "Analisis RFM":
    st.title("📈 Analisis RFM")
    
    # Tentukan tanggal terakhir dalam dataset
    last_date = merge_cleaned["dteday"].max()

    # Hitung RFM Metrics
    rfm = merge_cleaned.groupby("instant").agg({
        "dteday": lambda x: (last_date - x.max()).days,
        "instant": "count", 
        "cnt": "sum" 
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    st.dataframe(rfm)

    # Visualisasi Distribusi RFM
    st.subheader("📊 Distribusi RFM")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].hist(rfm["Recency"], bins=20, color="skyblue", edgecolor="black")
    ax[0].set_title("Distribusi Recency")
    
    ax[1].hist(rfm["Frequency"], bins=20, color="orange", edgecolor="black")
    ax[1].set_title("Distribusi Frequency")
    
    ax[2].hist(rfm["Monetary"], bins=20, color="green", edgecolor="black")
    ax[2].set_title("Distribusi Monetary")
    
    st.pyplot(fig)

elif page == "Analisis Geospasial":
    st.title("🌍 Analisis Geospasial Peminjaman Sepeda")
    
    # Pastikan dataset memiliki kolom lokasi (misalnya lat, lon)
    if "latitude" in merge_cleaned.columns and "longitude" in merge_cleaned.columns:
        fig = px.scatter_mapbox(
            merge_cleaned,
            lat="latitude",
            lon="longitude",
            color="cnt",
            size="cnt",
            mapbox_style="carto-positron",
            zoom=10
        )
        st.plotly_chart(fig)
    else:
        st.warning("Dataset tidak memiliki informasi geospasial.")

elif page == "Clustering":
    st.title("🔬 Clustering Data Sepeda (Manual Grouping)")

    def categorize_usage(cnt):
        if cnt < 1000:
            return "Rendah"
        elif 1000 <= cnt < 4000:
            return "Sedang"
        else:
            return "Tinggi"

    merge_cleaned["usage_category"] = merge_cleaned["cnt"].apply(categorize_usage)
    
    # Menampilkan jumlah data per kategori
    cluster_counts = merge_cleaned["usage_category"].value_counts()
    st.bar_chart(cluster_counts)

    # Scatter plot jumlah peminjaman berdasarkan temperatur dan kelembaban
    st.subheader("📊 Peminjaman Sepeda berdasarkan Temperatur & Kelembaban")
    fig = px.scatter(
        merge_cleaned,
        x="temp",
        y="hum",
        color="usage_category",
        title="Clustering Peminjaman Sepeda",
        labels={"temp": "Temperatur", "hum": "Kelembaban"}
    )
    st.plotly_chart(fig)

# Menjalankan dashboard
if __name__ == "__main__":
    st.sidebar.success("pilih halaman di atas")
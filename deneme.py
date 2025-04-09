import pandas as pd
import joblib
import streamlit as st
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔧 Kalıp Üretim Termin Süresi Tahmini")
st.markdown("Bu uygulama, kalıp üretim termin süresini tahmin etmek için veri alarak model eğitir ve tahmin yapar.")

# 📂 **Excel Yükleme**
uploaded_file = st.file_uploader("📁 Kalıp üretim verilerini içeren bir Excel dosyası yükleyin.", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # 📊 **Veriyi Oku**
        data = pd.read_excel(uploaded_file)
        st.write("📊 **Veri Önizleme:**", data.head())
        data.replace("-", np.nan, inplace=True)

        # 📌 **Gerekli Sütunlar**
        required_columns = [
            "kalip_tipi", "kalip_adedi", "sac_cinsi", "sac_kalinligi", "operasyon_sayisi","operasyon_tipi",
            "goz_adedi", "form", "yolluk_tipi", "vardiya_sayisi", "operator_deneyimi",
            "makina_kapasitesi", "malzeme_sertliği", "kalip_kompleksitesi", "parca_toleransi", "termin_suresi"
        ]

        # 📌 **Eksik Sütun Kontrolü**
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"❌ Eksik sütunlar: {missing_columns}")
        else:
            # 📌 **Kategorik Verileri Sayısallaştır**
            label_encoders = {}
            categorical_cols = ["kalip_tipi", "sac_cinsi", "form", "yolluk_tipi",
                                "malzeme_sertliği", "kalip_kompleksitesi", "parca_toleransi","operasyon_tipi"]

            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le

            # 📌 **Operatör Deneyimi Dönüştürme**
           # operator_map = {"Yeni": 1, "Orta": 2, "Deneyimli": 3}
            #data["operator_deneyimi"] = data["operator_deneyimi"].astype(str).map(operator_map)


            # 📌 **Eksik Verileri Medyan ile Doldur**
            for col in data.columns:
                if data[col].dtype != "object":
                    data[col] = data[col].fillna(data[col].median())

            # 📌 **Bağımsız (X) ve Bağımlı (y) Değişkenleri Tanımla**
            columns_to_drop = [col for col in ["termin_suresi", "proje_adi"] if col in data.columns]
            X = data.drop(columns=columns_to_drop)
            y = data["termin_suresi"]

            # 📌 **Veriyi %80 Eğitim - %20 Test Olarak Ayır**
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # 📌 **Modeli Eğit**
            model= RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            #model = RandomForestRegressor(n_estimators=100, random_state=45)
            model.fit(X_train, y_train)

            # 📈 **Modelin Doğruluğunu Ölç**
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # ✅ **Sonuçları Yazdır**
            st.success("✅ Model başarıyla eğitildi!")
            st.write(f"📈 **R² Skoru:** {r2:.2f}")
            st.write(f"📉 **Ortalama Mutlak Hata (MAE):** {mae:.2f} saat")
            st.write(f"📊 **Kök Ortalama Kare Hata (RMSE):** {rmse:.2f} saat")

            # 📌 **Modeli Kaydet**
            joblib.dump((model, label_encoders, X_train.columns.tolist()), "termin_tahmini_model.pkl")
    except Exception as e:
        st.error(f"❌ Bir hata oluştu: {e}")

# 🎯 **Tahmin İçin Kullanıcıdan Veri Alma**
st.sidebar.header("Yeni Proje Verilerini Girin")

kalip_tipi = st.sidebar.selectbox("Kalıp Tipi", ["Sac", "Plastik", "Kauçuk"])

yeni_veri = {
    "vardiya_sayisi": st.sidebar.number_input("Vardiya Sayısı", min_value=1, value=1, step=1),
   # "operator_deneyimi": st.sidebar.selectbox("Operatör Deneyimi", ["Yeni", "Orta", "Deneyimli"]),
    "makina_kapasitesi": st.sidebar.slider("Makina Kapasitesi", 1.0, 50.0, 25.0),
    "kalip_tipi": kalip_tipi
}

if kalip_tipi == "Sac":
    yeni_veri.update({
        "kalip_adedi": st.sidebar.number_input("Kalıp Adedi", min_value=1, value=1, step=1),
        "sac_cinsi": st.sidebar.text_input("Sac Cinsi"),
        "operasyon_sayisi": st.sidebar.number_input("Operasyon Sayısı", min_value=1, value=1, step=1)
    })
elif kalip_tipi == "Plastik":
    yeni_veri.update({
        "goz_adedi": st.sidebar.number_input("Göz Adedi", min_value=1, value=1, step=1),
        "form": st.sidebar.text_input("Form"),
        "yolluk_tipi": st.sidebar.text_input("Yolluk Tipi")
    })
elif kalip_tipi == "Kauçuk":
    yeni_veri.update({
        "karisim_tipi": st.sidebar.text_input("Karışım Tipi"),
        "sertlik_degeri": st.sidebar.number_input("Sertlik Değeri", min_value=20, max_value=100, value=50, step=1)
    })

# 🔮 **Tahmin Yap**
if st.sidebar.button("Termin Süresini Tahmin Et"):
    if os.path.exists("termin_tahmini_model.pkl"):
        try:
            model, label_encoders, train_columns = joblib.load("termin_tahmini_model.pkl")

            #yeni_veri["operator_deneyimi"] = operator_map[yeni_veri["operator_deneyimi"]]
            yeni_veri_df = pd.DataFrame([yeni_veri])

            for col in ["sac_cinsi", "form", "yolluk_tipi", "kalip_tipi"]:
                if col in yeni_veri_df.columns and col in label_encoders:
                    if yeni_veri_df[col][0] in label_encoders[col].classes_:
                        yeni_veri_df[col] = label_encoders[col].transform(yeni_veri_df[col].astype(str))
                    else:
                        yeni_veri_df[col] = -1

            for col in train_columns:
                if col not in yeni_veri_df.columns:
                    yeni_veri_df[col] = 0

            yeni_veri_df = yeni_veri_df[train_columns]
            tahmin = model.predict(yeni_veri_df)[0]
            st.sidebar.success(f"🕒 Tahmini Termin Süresi: {tahmin:.2f} saat")
        except Exception as e:
            st.sidebar.error(f"❌ Tahmin hatası: {e}")
    else:
        st.sidebar.error("❌ Model bulunamadı. Lütfen önce Excel yükleyin!")




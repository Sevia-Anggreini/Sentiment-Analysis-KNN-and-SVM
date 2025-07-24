import streamlit as st
import pandas as pd
import time
import datetime
import joblib
import io
from preprocessing import preprocess_dataframe

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="🔍",
    layout="wide",
)

# Load kamus dan stopwords
@st.cache_data
def load_resources():
    kamus_df = pd.read_excel('data/kamuskatabaku.xlsx')
    stopword_df = pd.read_excel('data/stopwordbahasa.xlsx')
    normalization_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))
    stopwords = set(stopword_df.iloc[:, 0].dropna().str.lower())
    positive_words = set(pd.read_excel('data/positive_word.xlsx').iloc[:, 0].astype(str).str.lower())
    negative_words = set(pd.read_excel('data/negative_word.xlsx').iloc[:, 0].astype(str).str.lower())
    return normalization_dict, stopwords, positive_words, negative_words

normalization_dict, stopwords, positive_words, negative_words = load_resources()

st.sidebar.markdown("<h1 style='text-align: center;'>🧑‍🏫 SENTIMEN ANALISIS 🧑‍🏫</h1>", unsafe_allow_html=True)
st.sidebar.caption("Made by [Sevia Anggreini S](https://www.linkedin.com/in/sevia-anggreini-simanjuntak-59b428217/)")
st.sidebar.markdown("<h6 style='height:5px;color:#fff;border:none;' h6>", unsafe_allow_html=True)

with st.sidebar.expander("❔Cara Menggunakan Sistem"):
    st.caption("1. Memilih model klasifikasi: K-Nearest Neighbor (K-NN) atau Support Vector Machine (SVM) 🛠️")
    st.caption("2. Memasukkan input data tunggal atau multi-data (Upload file CSV) 📝")
    st.caption("3. Melakukan klasifikasi sentimen berdasarkan data yang dimasukkan ⚙️")
    st.caption("4. Mengunduh hasil klasifikasi dalam format Excel 📥")

def preprocess_ulasan(df):
    # df: DataFrame dengan kolom 'ulasan'
    df = df.copy()
    df['Context'] = df['ulasan']
    df = preprocess_dataframe(df, normalization_dict, stopwords)
    # result_preprocessing sudah bersih, siap untuk vektorisasi
    df['ulasan_clean'] = df['result_preprocessing']
    return df

def klasifikasi_knn(data_input):
    try:
        df = preprocess_ulasan(data_input)
        vectorizer = joblib.load(r'C:\streamlit\sentimen\model\vectorizer.pkl')
        knn_model = joblib.load(r'C:\streamlit\sentimen\model\knn_model.pkl')
        ulasan_vector = vectorizer.transform(df['ulasan_clean'])
        prediksi = knn_model.predict(ulasan_vector)
        # Mapping label jika model.classes_ ada
        if hasattr(knn_model, 'classes_'):
            label_mapping = {i: label for i, label in enumerate(knn_model.classes_)}
            if isinstance(prediksi[0], str):
                df['Hasil Klasifikasi KNN'] = prediksi
            else:
                df['Hasil Klasifikasi KNN'] = [label_mapping.get(x, x) for x in prediksi]
        else:
            label_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
            df['Hasil Klasifikasi KNN'] = [label_mapping.get(x, x) for x in prediksi]
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
        return None

def klasifikasi_svm(data_input):
    try:
        # Preprocessing ulasan
        df = preprocess_ulasan(data_input)

        # Load vectorizer dan model (hasil training yang sama)
        vectorizer = joblib.load(r'C:\streamlit\sentimen\model\vectorizer.pkl')
        svm_model = joblib.load(r'C:\streamlit\sentimen\model\svm_model.pkl')

        # Transform menggunakan vectorizer hasil training
        ulasan_vector = vectorizer.transform(df['ulasan_clean'])

        # Prediksi
        prediksi = svm_model.predict(ulasan_vector)

        # Cek format prediksi dan mapping label
        if hasattr(svm_model, 'classes_'):
            label_mapping = {i: label for i, label in enumerate(svm_model.classes_)}
            if isinstance(prediksi[0], str):
                df['Hasil Klasifikasi SVM'] = prediksi
            else:
                df['Hasil Klasifikasi SVM'] = [label_mapping.get(x, x) for x in prediksi]
        else:
            label_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
            df['Hasil Klasifikasi SVM'] = [label_mapping.get(x, x) for x in prediksi]

        return df

    except Exception as e:
        st.error(f"Terjadi kesalahan saat klasifikasi SVM: {e}")
        return None

def intro():
    st.markdown(
        """
        <h1 style='text-align: center;'>KLASIFIKASI SENTIMEN</h1>
        <h3 style='text-align: center;'>🔍 Sistem Analisis Sentimen Ulasan JMO 💭</h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='height:2px;border:none;color:#000;background-color:#f00;' />", unsafe_allow_html=True)
    
    st.markdown(
        """
        <p style='text-align: center;'>
            Sistem ini digunakan untuk melakukan <b>analisis sentimen</b> pada ulasan aplikasi JMO (Jamsostek Mobile). 
            Analisis sentimen bertujuan untuk mengklasifikasikan opini pengguna menjadi kategori 
            <code>positif</code>, <code>negatif</code>, atau <code>netral</code> berdasarkan isi ulasan yang diberikan. 
            Dengan analisis ini, pengelola aplikasi dapat memahami persepsi dan kepuasan pengguna secara otomatis dan efisien.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📌 Tentang Algoritma K-Nearest Neighbor (K-NN)"):
        st.markdown(
            """
            **K-Nearest Neighbor (K-NN)** adalah algoritma klasifikasi yang bekerja berdasarkan kemiripan antar data.

            Algoritma ini:
            - Mengklasifikasikan ulasan berdasarkan tetangga terdekat yang kategorinya sudah diketahui.
            - Mencari sejumlah tetangga terdekat (**K**) dari ulasan baru.
            - Menentukan sentimen berdasarkan mayoritas kategori dari tetangga tersebut.

            🔧 Pada sistem ini, digunakan nilai parameter <code>K = 455</code> yang memberikan hasil akurasi terbaik sebesar <code>78.25%</code> setelah dilakukan pelatihan dan pengujian terhadap data ulasan.
            """,
            unsafe_allow_html=True
        )

    with st.expander("📌 Tentang Algoritma Support Vector Machine (SVM)"):
        st.markdown(
            """
            **Support Vector Machine (SVM)** adalah algoritma klasifikasi yang bekerja dengan mencari garis pemisah terbaik antar kelas.

            Kelebihannya:
            - Mampu memisahkan data ulasan ke dalam kategori sentimen secara optimal.
            - Efektif untuk menangani data teks dalam tugas analisis sentimen.
            - Menentukan margin pemisah antar kelas dengan presisi tinggi.

            🔧 Pada sistem ini, digunakan parameter <code>C = 1</code> dan <code>gamma = 1</code> yang memberikan hasil akurasi terbaik sebesar <code>90%</code> setelah dilakukan pelatihan dan pengujian terhadap data ulasan.
            """,
            unsafe_allow_html=True
        )

def model_knn():
    st.markdown(
        """
        <h1 style='text-align: center;'>🎭 KLASIFIKASI SENTIMEN 🎭</h1>
        <h3 style='text-align: center;'>Model K-Nearest Neighbor (KNN)</h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='height:2px;border:none;color:#000;background-color:#f00;' />", unsafe_allow_html=True)

    input_method = st.radio("Pilih Mode Input Ulasan:", ("Input Manual", "Upload File CSV"))

    if input_method == "Input Manual":
        manual_review = st.text_area("Input ulasan pada kolom berikut:")
        if st.button("Lanjutkan Proses Klasifikasi"):
            if manual_review.strip() == "":
                st.warning("Mohon masukkan ulasan terlebih dahulu.")
            else:
                with st.spinner("Memproses ulasan..."):
                    data_input = pd.DataFrame({'ulasan': [manual_review]})
                    start_time = time.time()
                    hasil_klasifikasi = klasifikasi_knn(data_input)
                    end_time = time.time()
                    if hasil_klasifikasi is not None:
                        durasi = round(end_time - start_time, 2)
                        st.markdown("<hr style='height:1px;border:none;color:#000;background-color:#fff;' />", unsafe_allow_html=True)
                        st.success("Ulasan berhasil diproses!")
                        st.write(f"🕒 Waktu klasifikasi: {durasi} detik")
                        st.write("Hasil Klasifikasi:")
                        st.write(hasil_klasifikasi[['ulasan', 'Hasil Klasifikasi KNN']])
                    else:
                        st.error("Terjadi kesalahan saat memproses ulasan.")

    elif input_method == "Upload File CSV":
        uploaded_file = st.file_uploader(
            "Upload file CSV yang berisi ulasan",
            type=["csv"],
            help="Pastikan file berisi kolom 'ulasan'."
        )
        if uploaded_file is not None:
            progress_text = "Memproses file..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1, text=progress_text)
            try:
                data = pd.read_csv(uploaded_file)
                if 'ulasan' not in data.columns:
                    st.error("File harus memiliki kolom 'ulasan'.")
                else:
                    st.info(f"⛓️ Jumlah data: {len(data)} ulasan")

                    with st.spinner("Sedang memproses..."):
                        start_time = time.time()
                        hasil_klasifikasi = klasifikasi_knn(data)
                        end_time = time.time()

                    if hasil_klasifikasi is not None:
                        durasi = round(end_time - start_time, 2)
                        st.markdown("<hr style='height:1px;border:none;color:#000;background-color:#fff;' />", unsafe_allow_html=True)
                        st.success("✅ Klasifikasi berhasil!")
                        st.write(f"🕒 Waktu klasifikasi: {durasi} detik")
                        st.write("Hasil Klasifikasi:")
                        st.write(hasil_klasifikasi[['ulasan', 'Hasil Klasifikasi KNN']])
                        output = io.BytesIO()
                        hasil_klasifikasi.to_excel(output, index=False)
                        st.download_button(
                            label="Download Hasil Klasifikasi",
                            data=output.getvalue(),
                            file_name='hasil_klasifikasi_knn.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    else:
                        st.error("Gagal melakukan klasifikasi.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

def model_svm():
    st.markdown(
        """
        <h1 style='text-align: center;'>🎭 KLASIFIKASI SENTIMEN 🎭</h1>
        <h3 style='text-align: center;'>Model Support Vector Machine (SVM)</h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='height:2px;border:none;color:#000;background-color:#f00;' />", unsafe_allow_html=True)

    input_method = st.radio("Pilih Mode Input Ulasan:", ("Input Manual", "Upload File CSV"))

    if input_method == "Input Manual":
        manual_review = st.text_area("Input ulasan pada kolom berikut:")
        if st.button("Lanjutkan Proses Klasifikasi"):
            if manual_review.strip() == "":
                st.warning("Mohon masukkan ulasan terlebih dahulu.")
            else:
                with st.spinner("Memproses ulasan..."):
                    data_input = pd.DataFrame({'ulasan': [manual_review]})
                    start_time = time.time()
                    hasil_klasifikasi = klasifikasi_svm(data_input)
                    end_time = time.time()
                    if hasil_klasifikasi is not None:
                        durasi = round(end_time - start_time, 2)
                        st.markdown("<hr style='height:1px;border:none;color:#000;background-color:#fff;' />", unsafe_allow_html=True)
                        st.success("Ulasan berhasil diklasifikasikan!")
                        st.write(f"🕒 Waktu klasifikasi: {durasi} detik")
                        st.write("Hasil Klasifikasi:")
                        st.write(hasil_klasifikasi[['ulasan', 'Hasil Klasifikasi SVM']])

    elif input_method == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload file CSV yang berisi ulasan", type=["csv"])
        if uploaded_file is not None:
            progress_text = "Memproses file..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1, text=progress_text)
            try:
                data = pd.read_csv(uploaded_file)
                if 'ulasan' not in data.columns:
                    st.error("File harus memiliki kolom 'ulasan'.")
                else:
                    st.info(f"⛓️ Jumlah data: {len(data)} ulasan")

                    with st.spinner("Sedang memproses..."):
                        start_time = time.time()
                        hasil_klasifikasi = klasifikasi_svm(data)
                        end_time = time.time()

                    if hasil_klasifikasi is not None:
                        durasi = round(end_time - start_time, 2)
                        st.markdown("<hr style='height:1px;border:none;color:#000;background-color:#fff;' />", unsafe_allow_html=True)
                        st.success("Klasifikasi berhasil!")
                        st.write(f"🕒 Waktu klasifikasi: {durasi} detik")
                        st.write("Hasil Klasifikasi:")
                        st.write(hasil_klasifikasi[['ulasan', 'Hasil Klasifikasi SVM']])
                        output = io.BytesIO()
                        hasil_klasifikasi.to_excel(output, index=False)
                        st.download_button(
                            label="Download Hasil Klasifikasi",
                            data=output.getvalue(),
                            file_name='hasil_klasifikasi_svm.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    else:
                        st.error("Gagal melakukan klasifikasi.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

page_names_to_funcs = {
    "Dashboard": intro,
    "Model K-Nearest Neighbor": model_knn,
    "Model Support Vector Machine": model_svm,
}

demo_name = st.sidebar.selectbox("Pilih Menu", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
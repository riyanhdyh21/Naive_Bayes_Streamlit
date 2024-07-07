import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import base64
import streamlit as st

def main():
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

    # Inisialisasi layout untuk konten utama di bagian kanan
    st.title("Website Data Mining Naive Bayes")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("### Data yang Diunggah:")
        st.write(data.head())

        label_encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = label_encoder.fit_transform(data[column])

        label = st.text_input("Masukkan nama kolom label:")

        if label:
            st.write("### Data Pemberian Label:")
            data['target'] = data[label]
            st.write(data.head())

            # Hapus kolom target dari data tabel
            data = data.drop(columns=['target'])

            X = data.drop(columns=[label])
            y = data[label]

            test_size = st.slider("Ukuran Data Uji (dalam persen):", min_value=10, max_value=100, step=10, value=20)

            # Bagi data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=0)

            st.write("### Ukuran Data Latih:")
            st.write(len(X_train))

            st.write("### Ukuran Data Uji:")
            st.write(len(X_test))

            if st.button("Latih dan Evaluasi Naive Bayes"):
                model = GaussianNB()
                model.fit(X_train, y_train)

                joblib.dump(model, 'naive_bayes_model.joblib')
                st.write("Model Naive Bayes telah disimpan.")

                y_pred = model.predict(X_test)

                # Evaluasi menggunakan akurasi
                accuracy = accuracy_score(y_test, y_pred)

                # Evaluasi menggunakan laporan klasifikasi
                report = classification_report(y_test, y_pred, output_dict=True)

                # Evaluasi menggunakan validasi silang
                cv_score = cross_val_score(model, X_train, y_train, cv=5)

                st.write("### Evaluasi Naive Bayes:")
                st.write(f"Akurasi: {accuracy:.2%}")
                st.write("Rata-rata Akurasi Cross-Validation: {:.2f}".format(cv_score.mean()))

                st.write("### Laporan Klasifikasi:")
                report_df = pd.DataFrame(report).transpose().round(2)
                st.dataframe(report_df)

                st.write("### Hasil Prediksi:")
                pred_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
                st.write(pred_df)

                fig, ax = plt.subplots(figsize=(8, 5))
                pred_df['Prediksi'].value_counts().plot(kind='bar', ax=ax)
                ax.set_title('Diagram Batang Hasil Prediksi')
                ax.set_xlabel('Kelas Prediksi')
                ax.set_ylabel('Jumlah')
                st.pyplot(fig)

                st.write("### Visualisasi Histogram Fitur:")
                fig, axes = plt.subplots(nrows=1, ncols=len(X_train.columns), figsize=(15, 5))
                for i, column in enumerate(X_train.columns):
                    axes[i].hist(X_train[column], bins=20)
                    axes[i].set_title(column)
                st.pyplot(fig)

                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">\
                        <button style="background-color:#4CAF50; border: none; color: white; padding: 15px 32px;\
                        text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px;\
                        cursor: pointer;">\
                        Unduh Dataset</button></a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan nama kolom label untuk data terlebih dahulu.")

if __name__ == "__main__":
    main()

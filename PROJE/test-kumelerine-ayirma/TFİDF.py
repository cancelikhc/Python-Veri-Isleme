import os
import numpy as np
import pandas as pd

# TF (Terim Frekansı) hesaplama fonksiyonu
def calculate_tf(document):
    words = document.split(" ")
    word_count = len(words)
    term_freq = {}
    for word in words:
        term_freq[word] = term_freq.get(word, 0) + 1
    return term_freq

# DF (Belge Frekansı) hesaplama fonksiyonu
def calculate_df(documents):
    df = {}
    for document in documents:
        words = set(document.split())
        for word in words:
            df[word] = df.get(word, 0) + 1
    return df

# IDF (Ters Belge Frekansı) hesaplama fonksiyonu
def calculate_idf(documents, df):
    num_documents = len(documents)
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log10(num_documents / (freq))
    return idf

# Belgeleri okuma fonksiyonu
def read_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file]
    return documents

# Ana klasör yolu
base_path = r"C:\Users\LENOVO\Desktop\odev1\Fold"

# Klasörler
folders = ["5"]

all_training_documents = []
all_test_documents = []

# Eğitim verisi için belgelerin birleştirilmesi ve TF değerlerinin hesaplanması
for folder in folders:
    output_folder = os.path.join(base_path, folder)
    
    # Eğitim belgelerini oku
    training_positive_documents = read_documents(os.path.join(output_folder, "egitim_pozitif.txt"))
    training_negative_documents = read_documents(os.path.join(output_folder, "egitim_negatif.txt"))

    all_training_documents.extend(training_positive_documents)
    all_training_documents.extend(training_negative_documents)

    # Test belgelerini oku
    test_positive_documents = read_documents(os.path.join(output_folder, "test_pozitif.txt"))
    test_negative_documents = read_documents(os.path.join(output_folder, "test_negatif.txt"))

    all_test_documents.extend(test_positive_documents)
    all_test_documents.extend(test_negative_documents)

# TF değerlerinin hesaplanması
tf_values = [calculate_tf(doc) for doc in all_training_documents]
tf_values_test = [calculate_tf(doc) for doc in all_test_documents]

# DF ve IDF değerlerinin hesaplanması
df_values = calculate_df(all_training_documents)
idf_values = calculate_idf(all_training_documents, df_values)

# TF-IDF matrisinin oluşturulması
tfidf_matrix = np.zeros((len(all_training_documents), len(df_values)))
tfidf_matrix_test = np.zeros((len(all_test_documents), len(df_values)))

for i, tf in enumerate(tf_values):
    for word, freq in tf.items():
        j = list(df_values.keys()).index(word)
        tfidf_matrix[i, j] = freq * idf_values[word]

for i, tf in enumerate(tf_values_test):
    for word, freq in tf.items():
        if word in df_values:
            j = list(df_values.keys()).index(word)
            tfidf_matrix_test[i, j] = freq * idf_values[word]

# Eğitim ve test verilerine sınıfların eklenmesi
num_positive_training = len(training_positive_documents)
num_negative_training = len(training_negative_documents)
num_positive_test = len(test_positive_documents)
num_negative_test = len(test_negative_documents)

training_classes = [1] * num_positive_training + [0] * num_negative_training
test_classes = [1] * num_positive_test + [0] * num_negative_test

# TF-IDF matrisine sınıfların eklenmesi (integer olarak)
tfidf_with_classes = np.column_stack((tfidf_matrix, np.array(training_classes, dtype=int)))
tfidf_test_with_classes = np.column_stack((tfidf_matrix_test, np.array(test_classes, dtype=int)))

# CSV dosyasına yazma
# CSV dosyasına yazma
column_names = list(df_values.keys())
column_names.append('class')

# Eğitim verisi
df = pd.DataFrame(tfidf_with_classes, columns=column_names)
df['class'] = df['class'].astype(int)  # Sınıf etiketlerini integer'a dönüştür
df.to_csv(r"C:\Users\LENOVO\Desktop\odev1\Fold\5\_training.csv", index=False)

# Test verisi
df_test = pd.DataFrame(tfidf_test_with_classes, columns=column_names)
df_test['class'] = df_test['class'].astype(int)  # Sınıf etiketlerini integer'a dönüştür
df_test.to_csv(r"C:\Users\LENOVO\Desktop\odev1\Fold\5\_test.csv", index=False)
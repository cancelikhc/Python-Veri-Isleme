import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Veri dosyalarını oku
eğitim_verisi = pd.read_csv(r"C:\Users\LENOVO\Desktop\MAK-ÖĞR\Fold\5\training.csv")
test_verisi = pd.read_csv(r"C:\Users\LENOVO\Desktop\MAK-ÖĞR\Fold\5\test.csv")

# Hedef sütunu al ve veriden kaldır
X_eğitim = eğitim_verisi.drop(columns=['class_label'])
y_eğitim = eğitim_verisi['class_label']
X_test = test_verisi.drop(columns=['class_label'])
y_test = test_verisi['class_label']

# SVM modelini tanımla ve eğit
svm_modeli = SVC()
svm_modeli.fit(X_eğitim, y_eğitim)

# Test verisi üzerinde tahmin yap

test_tahminleri = svm_modeli.predict(X_test)

# Test verisi için doğruluk ve f1 skorunu hesapla
test_doğruluk = accuracy_score(y_test, test_tahminleri)
test_f1_skoru = f1_score(y_test, test_tahminleri, average='weighted')

print("Test Doğruluk:", test_doğruluk)
print("Test F1 Skoru:", test_f1_skoru)

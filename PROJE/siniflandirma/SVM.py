import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Ortalama doğruluk ve F1 skorlarını depolamak için boş listeler oluştur
ortalama_dogruluklar = []
ortalama_f1_skorlari = []

# Klasörlerin listesi
klasorler = ["1", "2", "3", "4", "5"]

for klasor in klasorler:
    # Eğitim ve test verilerini oku
    eğitim_verisi = pd.read_csv(fr"C:\Users\LENOVO\Desktop\MAK-ÖĞR\Fold\{klasor}\training.csv")
    test_verisi = pd.read_csv(fr"C:\Users\LENOVO\Desktop\MAK-ÖĞR\Fold\{klasor}\test.csv")

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

    # Ortalama doğruluk ve F1 skorlarını listeye ekle
    ortalama_dogruluklar.append(test_doğruluk)
    ortalama_f1_skorlari.append(test_f1_skoru)

# Ortalama doğruluk ve F1 skorlarını hesapla
ortalama_dogruluk = sum(ortalama_dogruluklar) / len(ortalama_dogruluklar)
ortalama_f1_skoru = sum(ortalama_f1_skorlari) / len(ortalama_f1_skorlari)

print("Ortalama Test Doğruluk:", ortalama_dogruluk)
print("Ortalama Test F1 Skoru:", ortalama_f1_skoru)

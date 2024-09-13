import os
from sklearn.model_selection import KFold

# Dosyadan veriyi oku
with open(r"C:\Users\LENOVO\Desktop\odev1\islenmis_negatif_yorumlar.txt", "r") as file:
    data = file.readlines()

# Her satırın sonundaki newline karakterlerini temizle
data = [line.strip() for line in data]

# K-Fold çapraz doğrulama
kf = KFold(n_splits=5, shuffle=False)

# Klasör yolu
output_directory = r"C:\Users\LENOVO\Desktop\odev1\Fold"

# Klasörü oluştur
os.makedirs(output_directory, exist_ok=True)

fold_number = 1
for train_indices, test_indices in kf.split(data):
    print(f"Fold {fold_number}:")
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    # Eğitim verisini dosyaya yaz
    with open(os.path.join(output_directory, f"egitim_negatif_{fold_number}.txt"), "w") as train_file:
        for line in train_data:
            train_file.write(line + "\n")

    # Test verisini dosyaya yaz
    with open(os.path.join(output_directory, f"test_negatif_{fold_number}.txt"), "w") as test_file:
        for line in test_data:
            test_file.write(line + "\n")

    print("Training data written to train_fold_{}.txt".format(fold_number))
    print("Test data written to test_fold_{}.txt".format(fold_number))
    print()
    
    fold_number+=1
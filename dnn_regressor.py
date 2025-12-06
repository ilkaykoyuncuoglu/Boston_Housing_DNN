# dnn_regressor.py
import numpy as np
import pandas as pd

# =======================================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME (Normalization)
# =======================================================

# 1.1 Veri Yükleme
boston = pd.read_csv("boston.csv")
X = boston.data.astype(np.float32)
y = boston.target.astype(np.float32).reshape(-1, 1)

# 1.2 Normalizasyon (Z-Skor)
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
y_norm = (y - y.mean()) / y.std()

# =======================================================
# 2. MİMARİ VE AĞIRLIK BAŞLATMA
# =======================================================

INPUT_SIZE = X_norm.shape[1] # 13
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

# Ağırlık Başlatma (Küçük rastgele değerler)
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01
b1 = np.zeros((1, HIDDEN_SIZE))
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01
b2 = np.zeros((1, OUTPUT_SIZE))

# =======================================================
# 3. AKTİVASYON VE İLERİ YAYILIM
# =======================================================

def relu(Z):
    return np.maximum(0, Z)

def forward_propagation(X, W1, b1, W2, b2):
    # Katman 1 (Gizli)
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Katman 2 (Çıktı)
    Z2 = A1 @ W2 + b2
    A2 = Z2 # Regresyon için Lineer Aktivasyon
    
    cache = (Z1, A1, Z2, A2)
    return A2, cache

print("Kod Dosyası Hazırlandı.")
# =======================================================
# Eğitim döngüsü ve Geri Yayılım buradan devam edecek.
# =======================================================
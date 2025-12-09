import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing # Eğer Boston dosyanız yoksa alternatif olarak kullanılır

# ==============================================================================
# 1. VERİ YÜKLEME VE BÖLME
# ==============================================================================

print("=== VERİ YÜKLEME VE ÖN İŞLEME BAŞLIYOR ===")

# --- VERİ YÜKLEME YÖNTEMİ SEÇİMİ ---
try:
    # YÖNTEM A: Sizin manuel olarak yüklediğiniz 'boston.csv' dosyasını kullanır
    # NOT: Dosyanın, kodla aynı dizinde olması gerekir.
    data = pd.read_csv("boston.csv", sep='\s+', header=None)
    data.columns = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]
    
    # Özellikler (X) ve Hedef (y) ayırma
    X_raw = data.drop('MEDV', axis=1).values 
    y_raw = (data['MEDV'] * 100).values.reshape(-1, 1) # Fiyatları 1000'lik birimlere çevirdik
    feature_names = data.drop('MEDV', axis=1).columns.tolist()

except FileNotFoundError:
    # YÖNTEM B: Eğer 'boston.csv' bulunamazsa, scikit-learn'den California verisi kullanılır
    print("UYARI: 'boston.csv' dosyası bulunamadı. Yerine California Housing verisi kullanılıyor.")
    data_sklearn = fetch_california_housing()
    X_raw = data_sklearn.data
    y_raw = (data_sklearn.target * 100).reshape(-1, 1) # Fiyatları uyumlu hale getir
    feature_names = data_sklearn.feature_names.tolist()

# Veri setini Eğitim (%80) ve Test (%20) olarak ayırma
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Hedef ortalama ve sapmasını hesaplama
X_mean = np.mean(X_train_raw, axis=0)
X_std = np.std(X_train_raw, axis=0)
y_mean = np.mean(y_train_raw)
y_std = np.std(y_train_raw)

print(f"Eğitim Örnek Sayısı: {X_train_raw.shape[0]}, Özellik Sayısı: {X_raw.shape[1]}")
print(f"Ortalama Ev Fiyatı: {y_mean:.2f} bin Dolar")

# ==============================================================================
# 2. ÖN İŞLEME VE STANDARDİZASYON
# ==============================================================================

def normalize(X, mean, std):
    """Veriyi standardize eder."""
    return (X - mean) / std

# Özellikleri standardize etme
X_train = normalize(X_train_raw, X_mean, X_std)
X_test = normalize(X_test_raw, X_mean, X_std)

# Y değerleri (fiyatlar) normalize edilmez
y_train = y_train_raw
y_test = y_test_raw

# ==============================================================================
# 3. XGBOOST REGRESSOR EĞİTİMİ
# ==============================================================================

print("\n" + "="*50)
print("             XGBOOST REGRESSOR EĞİTİMİ")
print("="*50)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=1000,            # Ağaç sayısı
    learning_rate=0.05,           # Öğrenme hızı
    max_depth=6,                  # Ağaç derinliği
    random_state=42,
    use_label_encoder=False,
    eval_metric='mae'
)

# Modeli Eğitme (X_train standardize edilmiş, y_train orijinal)
xgb_model.fit(X_train, y_train.ravel()) 

print("XGBoost Eğitimi Tamamlandı.")

# ==============================================================================
# 4. TAHMİN VE DEĞERLENDİRME
# ==============================================================================

# Test Seti Üzerinde Tahmin Yapma
y_pred_xgb = xgb_model.predict(X_test)

# Ortalama Mutlak Hata (MAE) Hesaplama (bin Dolar cinsinden)
mae_xgb = mean_absolute_error(y_test.ravel(), y_pred_xgb)

print("\n" + "="*50)
print("             NİHAİ XGBOOST MODEL SONUÇLARI")
print("="*50)
print(f"Final MAE (Ortalama Mutlak Hata): {mae_xgb:.2f} bin Dolar 💰")

# ==============================================================================
# 5. HİPOTETİK TAHMİN
# ==============================================================================

# Ortalama özelliklere sahip hipotetik bir giriş oluşturma
X_new = X_mean.reshape(1, -1) 
X_new_standardized = normalize(X_new, X_mean, X_std)

# Tahmin yapma
y_new_pred_xgb = xgb_model.predict(X_new_standardized)

print(f"Hipotetik Evin Tahmini Fiyatı (Ortalama özellikler): {y_new_pred_xgb[0]:.2f} bin Dolar")
print(f"Gerçek Ortalama Fiyat: {y_mean:.2f} bin Dolar")

# Hipotetik tahminin ortalamadan sapması
sapma = np.abs(y_new_pred_xgb[0] - y_mean)
print(f"Sapma: {sapma:.2f} bin Dolar")
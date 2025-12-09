import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Veriyi bölmek için gerekli

# =======================================================
# 1. VERİ YÜKLEME VE BÖLME
# =======================================================

# 1.1 Veri Yükleme
boston = pd.read_csv("boston.csv", sep='\s+')
boston.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
X = boston.drop('MEDV', axis=1).values.astype(np.float32) 
y = boston['MEDV'].values.astype(np.float32).reshape(-1, 1)

print(f"Giriş Verisi (X) Boyutu: {X.shape}, Hedef Verisi (y) Boyutu: {y.shape}") 

# 1.2 Veriyi Bölme (Önce Test setini ayır: %10)
# Kalan %90, Train ve Validation için kullanılacak
X_remaining, X_test_raw, y_remaining, y_test_raw = train_test_split(
    X, y, test_size=0.1, random_state=42 
)

# Kalan %90, %80 Train ve %10 Validation olarak bölündü
# 0.1 / 0.9 = 0.1111... yani kalan verinin yaklaşık %11.1'i validation olur
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_remaining, y_remaining, test_size=(1/9), random_state=42
)

print(f"Eğitim Seti: {X_train_raw.shape[0]}, Doğrulama Seti: {X_val_raw.shape[0]}, Test Seti: {X_test_raw.shape[0]}")


# 1.3 NORMALİZASYON (SADECE EĞİTİM)

# Derin öğrenmede girdilerin ortalamasını 0, sapmasını 1 yapmak Gradyan İnişi'nin hedefi daha kolay bulmasını sağlar.
# Uç değerleri görmeyi sağlar
# ortalama ve sapmasını hesapla (Veri Sızıntısı Önlenir)

X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
y_mean, y_std = y_train_raw.mean(), y_train_raw.std()

# Setleri Normalize Et
X_train = (X_train_raw - X_mean) / X_std
y_train = (y_train_raw - y_mean) / y_std

X_val = (X_val_raw - X_mean) / X_std # Validation
y_val = (y_val_raw - y_mean) / y_std

X_test = (X_test_raw - X_mean) / X_std # Test
y_test = (y_test_raw - y_mean) / y_std


# =======================================================
# 2. MİMARİ VE AĞIRLIK BAŞLATMA
# =======================================================

input_size = X_train.shape[1] # Veri setiyle aynı boyutta olmalı.
hidden_size = 13 # Modelin acıklamasına baglı olarak değişebilir.
output_size = 1 # Sonuçta tek bir sayısal deger tahmin edilmek isteniyor.

# np.random.randn: Ağırlıkları tamamen sıfır başlatamayız, yoksa tüm nöronlar aynı şeyi öğrenir. Rastgele küçük sayılarla "simetriyi bozuyoruz".
# Sayıları çok küçük başlatıyoruz ki ağın başlangıçtaki hatası patlayıp eğitim bozulmasın.

# Bias (kayma) değerlerini sıfır başlatmakta sakınca yoktur, W (ağırlıklar), veriyi döndürürken b onu yukarı-aşağı öteler.

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01 
b2 = np.zeros((1, output_size))


# =======================================================
# 3. AKTİVASYON VE İLERİ YAYILIM (FORWARD PROPAGATION)
# =======================================================

# Doğrusal (Z = W.X + b) işlemini "doğrusal olmayan" bir hale getirir. Negatif her şeyi siler, pozitifleri olduğu gibi bırakır.
# Örn: Ev fiyatlarını sadece düz çizgilerle (lineer) açıklayamazsın. ReLU ağa "eğrileri" anlama gücü verir.

def relu(Z):
    return np.maximum(0, Z)

"""
Doğrusal Aralık (Linearity): Regresyon modellerinde tahmin etmek istediğimiz fiyat değerleri çok hassas olabilir.
Son katmana ReLU koyulursa, modelin öğrenme aşamasında "gerçek fiyatın altında" tahmin yapması gerektiği durumlarda
türevin aniden 0 olması (ölü nöronlar) veya modelin esnekliğini kaybetmesi riski doğar.
Biz modelin çıktı aralığını kısıtlamadan tam olarak hedef sayıya (22.4 gibi) odaklanmasını isteriz.
cache: Geri yayılım yaparken, "Hangi katmanda hangi değer vardı?" diye bakmamız gerekecek. Bu yüzden bu ara değerleri saklayıp geri gönderiyoruz.
"""

def forward_propagation(X, W1, b1, W2, b2):
    # --- 1. Katman: Lineer İşlem ---
    Z1 = X @ W1 + b1 # @ NumPy'da matris çarpımı demektir.
    # --- 1. Katman: Aktivasyon --
    A1 = relu(Z1)
    # --- 2. Katman (Çıktı): Lineer İşlem ---
    Z2 = A1 @ W2 + b2 
    A2 = Z2 # Aktivasyon yapılmadığı için Z2 A2'ye esit olacaktır. Atanmasının sebebi de iki değerin üstü üste binmeden ayrılabilmesi için.
    # --- Bilgileri Saklama ---
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Modelin ne kadar hatalı olduğunu ölçen o fonksiyondur

def calculate_loss(Y_pred, Y_true):
    # m = toplam örnek sayısı (dizinin / Matrisin satır sayısı)
    m = Y_pred.shape[0]
    # np.power(, 2) ile Farkın karesini alarak (squared error) negatiflerden de kurtuluruz.
    # hem de büyük hataları daha ağır cezalandırırız.
    squared_error = np.power((Y_pred - Y_true), 2)
    loss = np.sum(squared_error) / m
    return loss

def de_normalize(Y_norm, Y_mean, Y_std):
    """Normalize edilmiş tahmini orijinal fiyat aralığına çevirir."""
    return (Y_norm * Y_std) + Y_mean


# =======================================================
# 5. GERİ YAYILIM (BACKWARD PROPAGATION) - L2 DÜZENLİLEŞTİRME EKLENDİ
# =======================================================

# L2 Düzenlileştirme Katsayısı (Lambda). Çok küçük olmalıdır.
LAMBDA = 0.0019

def backward_propagation(X, y_true, Y_pred, W1, W2, cache):
    
    m = X.shape[0] 
    (Z1, A1, Z2, A2) = cache 

    # Çıktı Katmanı Hata Sinyali 
    # Not: Loss formülünü sade tutmak için L2 cezasını dW adımında uygulayacağız.
    dZ2 = (2/m) * (Y_pred - y_true)
    
    # Çıktı Katmanı Gradyanları
    dW2 = A1.T @ dZ2 
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # L2 CEZASI UYGULAMASI (Ağırlık Gradyanlarına eklenir)
    dW2 += LAMBDA * W2 # Gradyan + Lambda * W
    
    dA1 = dZ2 @ W2.T

    # Gizli Katman Gradyanları 
    dZ1 = dA1 * (Z1 > 0) 
    
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # L2 CEZASI UYGULAMASI (Ağırlık Gradyanlarına eklenir)
    dW1 += LAMBDA * W1 # Gradyan + Lambda * W
    
    return dW1, db1, dW2, db2


# =======================================================
# 6. OPTİMİZASYON VE EĞİTİM DÖNGÜSÜ (DENEME 29'UN PARAMETRELERİ)
# =======================================================

lr = 0.01          # Deneme 29'dan
epochs = 15000     # Önceki 20000 yerine Overfitting'i azaltmak için 15000
# Not: W1 ve W2 ağırlıkları kodun başında (Aşama 2'de) zaten belirlendi ve HIDDEN_SIZE=10.

print(f"\n=== L2 DÜZENLİLEŞTİRMELİ EĞİTİM BAŞLIYOR (LR: {lr}, Epochs: {epochs}, Gizli: 10) ===")

for epoch in range(epochs):
    
    # 1. İLERİ YAYILIM (EĞİTİM)
    Y_pred_train, cache = forward_propagation(X_train, W1, b1, W2, b2)
    
    # 2. GERİ YAYILIM VE AĞIRLIK GÜNCELLEMESİ (L2 cezası içerir)
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Y_pred_train, W1, W2, cache)
    
    # 3. AĞIRLIK GÜNCELLEMESİ
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    
    # İlerleme Kaydı ve VALIDATION KONTROLÜ
    if epoch % 3000 == 0: 
        # Loss hesaplamada L2 terimini göstermiyoruz, sadece MSE'ye odaklanıyoruz
        train_loss = calculate_loss(Y_pred_train, y_train) 
        Y_pred_val, _ = forward_propagation(X_val, W1, b1, W2, b2)
        val_loss = calculate_loss(Y_pred_val, y_val)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Validation Loss = {val_loss:.4f}")

# Eğitim bitti, son Train Loss'u kaydet
Y_pred_train, _ = forward_propagation(X_train, W1, b1, W2, b2)
final_train_loss = calculate_loss(Y_pred_train, y_train)


# =======================================================
# 7. MODEL DEĞERLENDİRMESİ (TEST)
# =======================================================

# 1. FİNAL TEST SETİ TAHMİNİ
Y_test_pred, _ = forward_propagation(X_test, W1, b1, W2, b2)

# 2. TEST HATA HESAPLAMA
final_test_loss = calculate_loss(Y_test_pred, y_test)

print(f"\nEğitim Sonrası Nihai TRAIN Loss: {final_train_loss:.4f}")
print(f"Eğitim Sonrası Nihai TEST Loss: {final_test_loss:.4f}")

# 3. OVERFITTING KONTROLÜ
oran = final_test_loss / final_train_loss
if oran > 1.2:
    print(f"UYARI: Test Loss, Train Loss'un {oran:.2f} katı. Overfitting riski devam ediyor!")
else:
    print(f"BAŞARILI: Test Loss, Train Loss'un {oran:.2f} katı. Model iyi genelleme yapıyor.")

# 4. TEST SETİ ÜZERİNDE MAE HESAPLAMA
Y_test_pred_original = de_normalize(Y_test_pred, y_mean, y_std)
y_test_true_original = de_normalize(y_test, y_mean, y_std)
MAE = np.mean(np.abs(Y_test_pred_original - y_test_true_original))
print(f"\nTEST Seti Ortalama Mutlak Hata (MAE): {MAE:.2f} bin Dolar")


# =======================================================
# 8. YENİ BİR VERİ İLE TAHMİN ETME (INFERENCE)
# =======================================================

X_new = X_mean.reshape(1, -1) 
print("\n=== Yeni Bir Veri İle Tahmin ===")
X_new_norm = (X_new - X_mean) / X_std 

Y_new_pred_norm, _ = forward_propagation(X_new_norm, W1, b1, W2, b2)
Y_new_pred_original = de_normalize(Y_new_pred_norm, y_mean, y_std)

print(f"Hipotetik Evin Tahmini Fiyatı: {Y_new_pred_original[0][0]:.2f} bin Dolar")
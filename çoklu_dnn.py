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


# =======================================================
# 5. GERİ YAYILIM (BACKWARD PROPAGATION)
# =======================================================
# biaslarda türev alınınca 1 olur formülden dolayı. X@W1+b1 türevi 1 olur.

def backward_propagation(X, y_true, Y_pred, W2, cache):
    
    m = X.shape[0] # Mini-Batch (Örnek) sayısı
    (Z1, A1, Z2, A2) = cache # İleri yayılımda sakladığımız ara değerler.

    # Çıktı Katmanı Hata Sinyali (Başlangıç)
    # dZ2: Loss'un Z2'ye göre türevi (Hata sinyalinin yönü ve büyüklüğü)
    # Math: Loss func(J) = (1/M)*(Toplam(Z2-y))kare, türevi alınınca kalan (2/m) * (Y_pred - y_true), çünkü Z2 = Y_pred (son cıktı).
    dZ2 = (2/m) * (Y_pred - y_true)
    
    # Çıktı Katmanı Gradyanları (dW2 ve db2
    # A. Ağırlık Gradyanı (dW2)
    # Mantık: Hata sinyalini (dZ2), önceki katmanın çıktısı (A1) kadar çarparız.
    dW2 = A1.T @ dZ2 
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T

    # Gizli Katman Gradyanları (dZ1, dW1 ve db1)
    # D. ReLU Türevi ve Hata Sinyali (dZ1)
    # Mantık: ReLU'nun türevi pozitife 1, negatife 0'dır. dA1 sinyalini sadece Z1'in pozitif olduğu yerlerde aktarırız.
    dZ1 = dA1 * (Z1 > 0) 
    
    # Mantık: Hata sinyalini (dZ1), en baştaki girdi (X) kadar çarparız.
    dW1 = X.T @ dZ1
    # Mantık: dZ1'lerin toplamı alınır.
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

def de_normalize(Y_norm, Y_mean, Y_std):
    """Normalize edilmiş tahmini orijinal fiyat aralığına çevirir."""
    return (Y_norm * Y_std) + Y_mean

# =======================================================
# 6. HİPERPARAMETRE IZGARA ARAMASI (GRID SEARCH)
# =======================================================

# Denenecek Hiperparametreler
lr_list = [0.1, 0.01, 0.005, 0.001]
epochs_list = [5000, 10000, 15000, 20000]
hidden_size_list = [8, 10, 15, 20]

results = [] # Sonuçları tablo halinde tutmak için

print("\n=== HİPERPARAMETRE IZGARA ARAMASI BAŞLIYOR ===")
deneme_sayisi = 0

# --- Ana Grid Search Döngüsü ---
for lr in lr_list:
    for epochs in epochs_list:
        for HIDDEN_SIZE in hidden_size_list:
            deneme_sayisi += 1
            print(f"\n--- Deneme {deneme_sayisi} | LR: {lr}, Epoch: {epochs}, Gizli: {HIDDEN_SIZE} ---")
            
            # 1. Ağırlıkları Yeniden Başlat (Her deneme bağımsız olmalı)
            INPUT_SIZE = X_train.shape[1] 
            OUTPUT_SIZE = 1
            
            W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01
            b1 = np.zeros((1, HIDDEN_SIZE))
            W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01 
            b2 = np.zeros((1, OUTPUT_SIZE))
            
            # 2. Eğitim Döngüsü
            for epoch in range(epochs):
                # İLERİ YAYILIM
                Y_pred_train, cache = forward_propagation(X_train, W1, b1, W2, b2)
                
                # GERİ YAYILIM VE GÜNCELLEME
                dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Y_pred_train, W2, cache)
                
                W1 = W1 - lr * dW1
                b1 = b1 - lr * db1
                W2 = W2 - lr * dW2
                b2 = b2 - lr * db2
            
            # Eğitim bitti, son Train Loss'u kaydet
            final_train_loss = calculate_loss(Y_pred_train, y_train) 
            
            
            # 3. TEST VE MAE HESAPLAMA (DEĞERLENDİRME)
            
            # TEST SETİ TAHMİNİ
            Y_test_pred, _ = forward_propagation(X_test, W1, b1, W2, b2)
            final_test_loss = calculate_loss(Y_test_pred, y_test)

            # MAE HESAPLAMA
            Y_test_pred_original = de_normalize(Y_test_pred, y_mean, y_std)
            y_test_true_original = de_normalize(y_test, y_mean, y_std)
            MAE = np.mean(np.abs(Y_test_pred_original - y_test_true_original))
            
            # 4. HİPOTETİK FİYAT TAHMİNİ
            
            # Giriş Verisini Normalize Et (X_mean)
            X_new = X_mean.reshape(1, -1) 
            X_new_norm = (X_new - X_mean) / X_std 
            
            # Tahmini Yap ve Geri Ölçeklendir
            Y_new_pred_norm, _ = forward_propagation(X_new_norm, W1, b1, W2, b2)
            Y_new_pred_original = de_normalize(Y_new_pred_norm, y_mean, y_std)
            hipotetik_fiyat = Y_new_pred_original[0][0]
            
            
            # 5. Sonuçları Kaydet ve Yazdır
            
            durum = "Başarılı" if final_test_loss < final_train_loss * 1.2 else "Overfitting"
            
            results.append({
                'LR': lr,
                'Epochs': epochs,
                'Gizli': HIDDEN_SIZE,
                'MAE': MAE,
                'Hip_Fiyat': hipotetik_fiyat,
                'T_Loss': final_train_loss,
                'V_Loss': final_test_loss,
                'Durum': durum
            })
            
            print(f"   Train Loss: {final_train_loss:.4f}, Test Loss: {final_test_loss:.4f}")
            print(f"   MAE: {MAE:.2f}K $, Hipotetik Fiyat: {hipotetik_fiyat:.2f}K $")


# =======================================================
# 7. DENEY SONUÇLARININ TABLOLANMASI
# =======================================================

print("\n\n=== NİHAİ PERFORMANS TABLOSU (Grid Search) ===")

# Pandas kullanarak tabloyu güzelleştirelim
results_df = pd.DataFrame(results)

# Hipotetik fiyatın ortalamaya (22.5K $) ne kadar yakın olduğunu ölçen bir metrik ekleyelim
avg_price_boston = 22.53 # Veri setinin yaklaşık ortalaması
results_df['Sapma'] = np.abs(results_df['Hip_Fiyat'] - avg_price_boston)

# Sonuçları en düşük MAE'ye göre sıralayalım
results_df_sorted = results_df.sort_values(by='MAE', ascending=True)

# Gerekli sütunları yuvarlayalım
results_df_sorted = results_df_sorted.round({
    'MAE': 2, 'Hip_Fiyat': 2, 'T_Loss': 4, 'V_Loss': 4, 'Sapma': 2
})

print(results_df_sorted)

print("\nEN İYİ MODEL: En düşük MAE ve Sapma değerine sahip olan modeldir.")
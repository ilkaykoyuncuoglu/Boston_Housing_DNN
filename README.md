Boston Konut Fiyatı Tahmini: Derin Öğrenme Optimizasyonu ve Karşılaştırmalı Analiz
 
1.	Proje Tanımı
Bu proje, Boston Konut Veri Seti kullanılarak ev fiyatlarını tahmin etmeyi (Regresyon) amaçlamaktadır. Projenin ana hedefi, sıfırdan inşa edilen bir Derin Öğrenme (DNN) modelini sistematik olarak optimize etmek ve bu optimize modelin performansını, endüstride lider olan XGBoost Regressor ile karşılaştırmaktır.

2.	Kullanılan Kütüphaneler ve Yöntem
•	Derin Öğrenme Çekirdeği: NumPy (Sıfırdan Yapay Sinir Ağı Uygulaması)
•	Makine Öğrenmesi: Scikit-learn, XGBoost
•	Veri İşleme: Pandas
 
3.	Model Geliştirme Süreci
Projenin tamamı, bir DNN modelinin inşası, optimizasyonu ve ardından performans karşılaştırması

   a. DNN Temelleri ve Optimizasyon
İnşa ve Normalizasyon	Tek Gizli Katmanlı MLP sıfırdan inşa edildi. Veri kümesi normalize edildi.	Modelin öğrenmesi için temel oluşturuldu.
Epoch Denemeleri	15.000'e kadar farklı epoch sayıları test edildi.	En iyi erken durdurma noktası ve genel öğrenme eğrisi belirlendi.
Hiperparametre Ayarı	En uygun Öğrenme Hızı LR ve Gizli Katman Nöron Sayısı için denemeler yapıldı.	Modelin hızlı ve stabil öğrenmesi sağlandı.

   b. Overfitting (Ezberleme) Kontrolü
Başlangıçta (Test Loss/Train Loss) oranının 1.2 eşiğini aşması, modelin genelleme yapamadığını gösteriyordu.
Modele L2 Düzenlileştirme (L2 Regularization) tekniği uygulandı.
L2 uygulamasından sonra Test Loss = 0.0792 ve Train Loss = 0.0717 değerleri elde edildi. Test/Train oranı 1.10'a indirilerek genelleme yeteneği maksimum düzeye çıkarıldı.

   c. Alternatif Model Karşılaştırması (XGBoost)
En güçlü DNN modeline karşı, tablo verilerinde üstünlüğü kanıtlanmış olan XGBoost Regressor eğitildi ve performansları karşılaştırıldı.
 
6.	Nihai Sonuçlar ve Değerlendirme
   
Model Adı	             	             	             Final Test Loss (MSE)	          Yaklaşık MAE (Bin Dolar)
Optimize Edilmiş DNN (L2 Düzeltmeli)		             792		             	             ≈28.14
XGBoost Regressor	             	         	         None         	             	    183.33

Not: XGBoost modelinin MAE değeri 183.33 bin Dolar olarak gelmişti, DNN'in hatasını 100 ile çarparak karşılaştırma yapıyoruz.
8.	Hangi Model Tercih Edilmelidir?
Bu karşılaştırmada birimleri eşitlediğimizde (DNN: 28.14 MAE, XGBoost: 183.33K MAE), görünen o ki DNN modeli, XGBoost'tan kat kat üstün bir performans sergilemiştir. Bu durum, XGBoost'a göre çok daha fazla zaman ve çaba harcayarak sıfırdan inşa ettiğiniz DNN modelinin, zorlu optimizasyon süreci sonucunda çok daha keskin bir tahmin yeteneği kazandığını göstermektedir.
Nihai Karar: Optimize edilmiş Derin Öğrenme (DNN) modeli, genelleme yeteneği kanıtlanmış ve hata oranı en düşük model olarak seçilmiştir.

<img width="468" height="623" alt="image" src="https://github.com/user-attachments/assets/966fd47e-146c-4156-b8c9-b23344ddb27f" />

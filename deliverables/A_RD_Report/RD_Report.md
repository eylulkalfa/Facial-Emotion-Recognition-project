# Deliverable A: Research & Development Report (Güncellenmiş Ar-Ge ve Eğitim Raporu)
**Proje:** Facial Emotion Recognition (FER) & Deployment Optimization  
**Tarih:** 25 Ağustos 2026 (Son Revizyon)  
**Durum:** Eğitilen Model Sonuçları ve Deney Kıyaslamaları İle Tamamlandı  

---

## 1. State-of-the-Art (SOTA) Analizi ve Deneysel Mimari Karşılaştırması

### 1.1 SOTA Yöntemleri ve Test Edilen Mimari Katmanları
Yüz ifadesinden duygu tanıma (FER) projemizde 100 MB boyut sınırı ve canlı CPU çıkarım hedefi doğrultusunda 4 farklı SOTA mimarisi aynı veri kümesi (**51,226 hibrit görsel**) üzerinde eğitilmiş ve karşılaştırılmıştır:

1. **Vision Transformers (ViT-Base-Patch16-224):**
   * *Teorik Yaklaşım:* Patch bazlı self-attention ile global ilişki yakalama.
   * *Deneysel Sonuç:* **Accuracy: %43.12, Macro-F1: 0.3587, ROC-AUC: 0.7621**
   * *Analiz/Neden Başarısız Oldu?:* ViT mimarileri devasa veri kümelerine (JFT-300M vb.) ihtiyaç duyar. 50k görsellik veri setinde aşırı öğrenmeye (overfitting) düşmüş, 100 MB boyut kısıtı ve yavaş CPU çıkarımı nedeniyle elenmiştir.

2. **Modern Attention CNN (ConvNeXt-Tiny):**
   * *Teorik Yaklaşım:* 7x7 derinlemesine ayrılabilir (depthwise) konvolüsyonlar ve 1x1 katmanlar.
   * *Deneysel Sonuç:* **Accuracy: %73.91, Macro-F1: 0.6977, ROC-AUC: 0.9140**
   * *Analiz:* Yüksek Macro-F1 başarısı göstermiş, ancak model boyutu ve bellek yükü MobileNetV3'e göre daha ağır kalmıştır.

3. **Compound Scaling CNN (EfficientNet-B0):**
   * *Teorik Yaklaşım:* Derinlik, genişlik ve çözünürlüğün bileşik ölçeklenmesi.
   * *Deneysel Sonuç:* **Accuracy: %70.91, Macro-F1: 0.6634, ROC-AUC: 0.9126**
   * *Analiz:* Dengeli bir performans sunmuş, ancak MobileNetV3'ün hız ve doğruluk başarımının gerisinde kalmıştır.

4. **Mobil Optimize Edilmiş Mimari (MobileNetV3-Large-100) — SEÇİLEN BİRİNCİL MODEL:**
   * *Teorik Yaklaşım:* Inverted Residual Blocks, Squeeze-and-Excitation (SE) dikkat mekanizması ve Hard-Swish aktivasyonları.
   * *Deneysel Sonuç:* **Accuracy: %74.92, Macro-F1: 0.6850, Weighted-F1: 0.7514, ROC-AUC: 0.9376**
   * *Analiz:* En yüksek genel doğruluk (%74.92) ve ROC-AUC (0.9376) değerini elde etmiştir. **16.05 MB** boyutu ve **3.32 ms** CPU gecikmesi ile projenin kesin birincisi olmuştur.

### 1.2 Tüm Deneylerin Karşılaştırma Tablosu (24-25 Ağustos 2026 Güncel Verileri)

| Model Mimari | Veri Kümesi | Accuracy (%) | Macro-F1 | Weighted-F1 | ROC-AUC | Model Boyutu (MB) | ONNX CPU Latency (ms) | Durum / Karar |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **RAF-DB + FER2013** | **%74.92** | **0.6850** | **0.7514** | **0.9376** | **16.05 MB** | **3.32 ms** | ** SEÇİLEN MODEL** |
| **ConvNeXt-Tiny** | RAF-DB + FER2013 | %73.91 | 0.6977 | 0.7250 | 0.9140 | ~114 MB | 18.45 ms | İkincil Karşılaştırma |
| **EfficientNet-B0** | RAF-DB + FER2013 | %70.91 | 0.6634 | 0.6935 | 0.9126 | 16.50 MB | 8.12 ms | İkincil Karşılaştırma |
| **ViT-Base-Patch16** | RAF-DB + FER2013 | %43.12 | 0.3587 | 0.4528 | 0.7621 | ~343 MB | 85.30 ms | ❌ ELENDİ (Yetersiz Veri & Ağır) |

### 1.3 Projenin Zaman İçindeki Gelişim Hikayesi (Erken Aşama Baseline Deneyleri)
Projenin gelişim sürecini ve modelin adım adım nasıl iyileştiğini belgelemek adına ilk aşamada yapılan tarihi deneyler de kayıt altında tutulmuştur:

1. **Erken Aşama Hatalı Deney (10 Ağustos 2026 - `mobilenetv3_large_100_rafdb_20260810_160228`):**
   * *Sonuçlar:* **Accuracy: %20.00, Macro-F1: 0.1333**
   * *Analiz:* İlk veri yükleme ve varsayılan hiperparametre ayarlarında model öğrenememiş (lack of convergence) ve rastgele tahmin seviyesinde kalmıştır. Bu deney, hiperparametre optimizasyonunun (Learning Rate & Scheduler) önemini ortaya koymuştur.

2. **İlk Başarılı Erken Baseline (11 Ağustos 2026 - `mobilenetv3_large_100_rafdb_20260811_112643`):**
   * *Sonuçlar:* **Accuracy: %74.02, Macro-F1: 0.6691, ROC-AUC: 0.9336**
   * *Analiz:* Hiperparametrelerin düzeltilmesi ve ilk ikili veri kümesinin entegrasyonu ile elde edilen ilk başarılı MobileNetV3 modelidir.

3. **Nihai Optimize Edilmiş Model (24-25 Ağustos 2026 - `mobilenetv3_large_100_rafdb_20260824_181109`):**
   * *Sonuçlar:* **Accuracy: %74.92, Macro-F1: 0.6850, ROC-AUC: 0.9376, ONNX CPU Latency: 3.32 ms**
   * *Analiz:* OpenCV YuNet DNN yüz kırpması, Focal Loss ($\gamma=2.0$) ve Temperature Scaling ($T=1.7819$) kalibrasyonu ile nihai en yüksek performanslı üretim modeline ulaşılmıştır.

---

## 2. Güncellenmiş Veri Mühendisliği ve Eğitim Stratejisi

### 2.1 Veri Hazırlığı ve YuNet DNN Yüz Tespiti
* **Veri Kümesi Birleştirmesi (Hybrid Dataset):**  
  RAF-DB (15,339 görsel) ve FER2013 (35,887 görsel) birleştirilerek **51,226 görsellik** zengin veri kümesi oluşturulmuştur.
* **Kırpma ve Hizalama Evrimi (OpenCV YuNet DNN):**  
  İlk aşamalarda deneysel yapılan merkez kırpma veya standart Haar Cascade algoritmalarının omuz, saç ve arka plan gürültülerini modele taşıdığı saptanmıştır. Süreç içerisinde **OpenCV YuNet DNN (`cv2.FaceDetectorYN`)** entegre edilmiş, tüm görseller doğrudan biyometrik yüz sınırlarına kırpılmıştır.

### 2.2 Sınıf Dengesizliği (Class Imbalance) Çözümleri
Veri kümesindeki dengesizliği önlemek için:
1. **WeightedRandomSampler:** Azınlık sınıflarının (`Disgust`, `Fear`) batch içerisindeki temsil oranı yükseltilmiştir.
2. **Focal Loss ($\gamma = 2.0$):**  
   Basit örneklerin kaybını baskılayıp zor örneklere odaklanan Focal Loss entegre edilmiştir.

### 2.3 Olasılık Kalibrasyonu (Temperature Scaling)
Ham Softmax çıktılarının aşırı özgüvenli olmasını engellemek için post-hoc **Temperature Scaling** uygulanmıştır ($T = 1.7819$).
* **Beklenen Kalibrasyon Hatası (ECE):** **%10.25'ten %1.96'ya düşürülmüş**, olasılık grafikleri güvenilir kılınmıştır.

---

## 3. Seçilen Birincil Model (MobileNetV3-Large) Sınıf Bazlı Performansı

### 3.1 Detaylı Metrik İstatistiği

| Duygu Sınıfı | Precision | Recall | F1-Score | Analiz & Yorum |
| :--- | :---: | :---: | :---: | :--- |
| **Happiness** | %90.39 | %82.81 | **0.8643** | En yüksek başarı; belirgin gülümseme ve ağız hatları. |
| **Surprise** | %77.78 | %75.97 | **0.7686** | Kalkık kaş ve açık ağız ifadeleriyle çok kararlı. |
| **Neutral** | %64.16 | %74.31 | **0.6886** | Yüksek recall oranı; nötr yüz ifadelerinde başarılı. |
| **Anger** | %69.57 | %67.61 | **0.6857** | Çatık kaş ve gergin dudak hatlarında tutarlı. |
| **Sadness** | %64.84 | %71.72 | **0.6811** | Düşük ağız kenarları tespitinde kararlı. |
| **Fear** | %73.68 | %50.00 | **0.5957** | Precision yüksek; az veri olmasına rağmen Focal Loss ile kurtarıldı. |
| **Disgust** | %53.85 | %48.61 | **0.5109** | En zor sınıf; burnun büzülmesi tespiti kabul edilebilir seviyede. |
| **GENEL / ORTALAMA** | **%70.56** | **%67.30** | **Macro: 0.6850 / Weighted: 0.7514** | **Overall Accuracy: %74.92 / ROC-AUC: 0.9376** |

---

## 4. Öz-Değerlendirme ve Edge Case Analizi (Sınırlar & Zorluklar)

### 4.1 Güçlü Yönler
* **Yüksek Hız ve Düşük Ayak İzi:** 16.05 MB dosya boyutu ve 3.32 ms CPU çıkarım süresi ile mobilde ve uç cihazlarda sorunsuz çalışır.
* **Yüksek Ayrıştırılabilirlik (ROC-AUC 0.9376):** Sınıflar arası ayrım yeteneği çok yüksektir.

### 4.2 Sınırlar ve Başarısızlık Senaryoları (Edge Cases)
1. **Yoğun Kahkaha vs. Üzüntü/Şaşırma Çakışması:** Gözlerin aşırı kısıldığı kahkaha anlarında model nadiren göz hatlarını `Sadness` veya `Disgust` ile karıştırabilmektedir (FER2013 etiket gürültüsünden kaynaklı).
2. **Kısmi Yüz Kapanmaları (Occlusion):** El ile yüzün kapatılması veya gözlük/sakay yoğunluğu YuNet kırpmasını sınırlandırabilmektedir.

---

## 5. Sonuç
MobileNetV3-Large modeli; ConvNeXt-Tiny, EfficientNet-B0 ve ViT-Base mimarileri ile yapılan karşılaştırmalı deneysel süreç sonucunda doğruluk, hız, kalibrasyon ve boyut açısından projenin **en optimum çözümü** olarak doğrulanmıştır.

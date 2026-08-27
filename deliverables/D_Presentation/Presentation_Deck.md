# Deliverable D: Final Technical Presentation (Hikaye Odaklı Teknik Sunum Dokümanı)
**Proje:** Facial Emotion Recognition (FER) & Deployment Optimization  
**Sunumu Yapan:** Mühendislik ve Ar-Ge Takımı  
**Tarih:** 26 Ağustos 2026 (Revize Edilmiş Versiyon)  

---

## 📋 Sunum Akış Yapısı (Agenda)

1. **Giriş ve Proje Amacı:** Kısıtlar (100 MB, Canlı CPU Hızı) ve 7 Duygu Sınıfı
2. **Süreç & Gelişim Hikayesi:** Karşılaşılan Engeller, Yapılan Değişiklikler ve Kazanılan Deneyimler
3. **Teknik Derinleşme & SOTA Mimarileri:** ViT neden elendi? MobileNetV3 vs ConvNeXt vs EfficientNet
4. **Veri Mühendisliği & Optimizasyon:** YuNet DNN Yüz Tespiti, Focal Loss ve Temperature Scaling
5. **ONNX Dağıtımı & Performans Metrikleri:** 11.48x Hızlanma, 3.32 ms Latency, ROC-AUC ve Sınıf Metrikleri
6. **Canlı Demo & Öz-Değerlendirme:** Gradio Web UI, Güçlü Yönler ve Edge Cases
7. **Sonuç ve Kapanış**

---

## SLAYT 1: Giriş ve Proje Amacı (Executive Summary)

### 📌 Başlık: Yüz İfadesinden Duygu Tanıma (FER) ve Canlı Dağıtım Optimizasyonu
* **Proje Amacı:** Gerçek zamanlı (real-time) kamera akışlarında çalışabilen, 100 MB boyut sınırını aşmayan ve CPU/Edge cihazlarda düşük gecikme sunan 7 duygu sınıflı FER sistemi geliştirmek.
* **Ana Hedef Metrikleri:**
  * **Sınıf Sayısı:** 7 Temel Duygu (`Anger`, `Disgust`, `Fear`, `Happiness`, `Sadness`, `Surprise`, `Neutral`).
  * **Boyut Kısıtı:** < 100 MB (**Elde Edilen: 16.05 MB**).
  * **Çıkarım Hızı:** CPU ortamında canlı akışa (real-time FPS) uygun olması (**Elde Edilen: 3.32 ms / 300+ FPS**).

> **🎤 Konuşmacı Notu:** "Sunumuma hoş geldiniz. Bu projede amacımız sadece yüksek doğruluklu bir duygu tanıma modeli eğitmek değil, aynı zamanda bu modeli 100 MB altı boyut ve 3.32 ms gibi canlı akışa uygun radikal bir hızla CPU üzerinde çalışabilir hale getirmekti."

---

## SLAYT 2: Süreç ve Yolculuk — Nelerle Karşılaştım? Neleri Değiştirdim?

### 🔄 Adım Adım Deneyim ve Gelişim Hikayesi

```
[10 Ağustos: İLK ENGEL]  ──► [11 Ağustos: İLK BASELINE] ──► [24-25 Ağustos: NİHAİ SOTA]
LR/Loss Uyumsuzluğu          Hiperparametre Düzeltmesi        YuNet DNN + Focal Loss +
%20.00 Accuracy (Rastgele)   %74.02 Accuracy Baseline        Temperature Scaling + ONNX
                                                              %74.92 Accuracy / 3.32 ms
```

### 🧠 Karşılaşılan Engeller ve Değişiklikler:
1. **İlkel Kırpma Yöntemlerinin Yetersizliği:**
   * *Karşılaşılan Durum:* İlk aşamada merkez kırpma (center crop) yapıldığında arka plan, saç ve omuzlar modele giriyordu.
   * *Değişiklik:* **OpenCV YuNet DNN (`cv2.FaceDetectorYN`)** entegre edilerek milimetrik yüz biyometrik hatlarına odaklanıldı.
2. **Sınıf Dengesizliği (Class Imbalance):**
   * *Karşılaşılan Durum:* FER2013 ve RAF-DB'de `Happiness` ve `Neutral` çok yüksek, `Disgust` ve `Fear` ise %5'in altındaydı.
   * *Değişiklik:* `WeightedRandomSampler` ve **Focal Loss ($\gamma=2.0$)** entegre edilerek azınlık sınıfların F1 skorları yükseltildi.
3. **Ham Softmax Özgüven Sorunu (Overconfidence):**
   * *Karşılaşılan Durum:* Ham model yanlış tahminlerde dahi %99 özgüven üretiyordu.
   * *Değişiklik:* **Temperature Scaling ($T=1.7819$)** uygulanarak ECE hatası %10.25'ten %1.96'ya düşürüldü.

> **🎤 Konuşmacı Notu:** "Projede ilk denememizde %20 doğrulukta kaldık. Ancak vazgeçmeyip hiperparametreleri düzelttik, YuNet yüz tespitini entegre ettik ve Focal Loss ile azınlık sınıfları kurtararak modeli adım adım %74.92 seviyesine taşıdık."

---

## SLAYT 3: SOTA Araştırması ve Mimari Seçim Kararları

### 📊 4 Farklı Mimari ile Deneysel Kıyaslama

| Model Mimari | Veri Kümesi | Accuracy (%) | Macro-F1 | ROC-AUC | Boyut (MB) | ONNX CPU Latency | Durum / Karar |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **RAF-DB + FER2013** | **%74.92** | **0.6850** | **0.9376** | **16.05 MB** | **3.32 ms** | ** SEÇİLEN MODEL** |
| **ConvNeXt-Tiny** | RAF-DB + FER2013 | %73.91 | 0.6977 | 0.9140 | ~114 MB | 18.45 ms | İkincil Karşılaştırma |
| **EfficientNet-B0** | RAF-DB + FER2013 | %70.91 | 0.6634 | 0.9126 | 16.50 MB | 8.12 ms | İkincil Karşılaştırma |
| **ViT-Base-Patch16** | RAF-DB + FER2013 | %43.12 | 0.3587 | 0.7621 | ~343 MB | 85.30 ms |  ELENDİ (Yetersiz Veri & Ağır) |

### 💡 Neler Öğrendim?
* **ViT (Vision Transformer) Neden Başarısız Oldu?:** ViT mimarileri devasa veri kümelerine (JFT-300M vb.) ihtiyaç duyar. 50k görsellik veri setimizde aşırı öğrenmeye (overfitting) düşmüş (%43.12 doğruluk) ve 85 ms süresiyle kısıtlarımızı ihlal etmiştir.
* **Neden MobileNetV3-Large?:** Inverted Residual bloklar, Hard-Swish aktivasyonu ve SE (Squeeze-and-Excitation) dikkat mekanizması sayesinde hem en yüksek doğruluğu (%74.92) hem de en yüksek hızı (3.32 ms) vermiştir.

---

## SLAYT 4: Veri Mühendisliği ve Yüz Tespiti Pipeline'ı

### 🛠️ 51,226 Görsellik Hibrit Veri Kümesi
1. **RAF-DB (15,339 Görsel):** Gerçek dikey/açılı yüzler.
2. **FER2013 (35,887 Görsel):** Vesikalık/deneysel yüzler.

```
 [ Ham Görsel ] ──► [ OpenCV YuNet DNN ] ──► [ 224x224 RGB Yüz Crop ] ──► [ Model Girdisi ]
 (Omuz/Arka Plan)     Milimetrik Tespit       Arka Plan Temizlendi          Saf Biyometri
```

> **🎤 Konuşmacı Notu:** "Veri tarafında en büyük atılımımız YuNet DNN yüz tespitini entegre etmek oldu. Model artık elbise veya arka plan renklerinden değil, doğrudan kaş, göz ve ağız büzülmelerinden duygu öğreniyor."

---

## SLAYT 5: ONNX Dönüşümü ve Performans Benchmark

### ⚡ PyTorch vs. ONNX Runtime CPU Karşılaştırması

```
 Ortalama Gecikme (Latency)               İşlem Hacmi (Throughput)
 PyTorch [38.15 ms]                       PyTorch [26.21 FPS]
 ONNX    [3.32 ms] 🚀 11.48x SPEEDUP      ONNX    [300.77 FPS] ⚡ 300+ FPS!
```

* **Model Boyutu:** `16.05 MB` (100 MB kısıtının yalnızca %16'sı).
* **Sayısal Tutarlılık:** $Max Abs Diff = 1.49 \times 10^{-6}$ ($< 10^{-5}$) ile PyTorch ve ONNX çıktıları birebir eşleşmiştir.

---

## SLAYT 6: Performans Metrikleri ve Sınıf Bazlı Analiz

### 📈 MobileNetV3 Sınıf Bazlı Performans Tablosu

| Duygu Sınıfı | Precision | Recall | F1-Score | Sınıf Değerlendirmesi |
| :--- | :---: | :---: | :---: | :--- |
| **Happiness** | %90.39 | %82.81 | **0.8643** | En yüksek doğruluk; açık gülümseme hatları. |
| **Surprise** | %77.78 | %75.97 | **0.7686** | Kalkık kaş ve açık ağız ifadeleri kararlı. |
| **Neutral** | %64.16 | %74.31 | **0.6886** | Yüksek recall oranı; nötr yüzlerde tutarlı. |
| **Anger** | %69.57 | %67.61 | **0.6857** | Çatık kaş hatlarında başarılı. |
| **Sadness** | %64.84 | %71.72 | **0.6811** | Düşük dudak kenarları tespiti kararlı. |
| **Fear** | %73.68 | %50.00 | **0.5957** | Precision yüksek; Focal Loss ile kurtarıldı. |
| **Disgust** | %53.85 | %48.61 | **0.5109** | En zor sınıf; burnun büzülmesi kabul edilebilir seviyede. |
| **GENEL METRİKLER** | **%70.56** | **%67.30** | **Macro-F1: 0.6850 / Weighted-F1: 0.7514** | **Overall Accuracy: %74.92 / ROC-AUC: 0.9376** |

---

## SLAYT 7: Canlı Demo ve Web Arayüzü (Gradio)

### 🎭 Web UI Özellikleri (`scripts/demo.py`)
1. **Çift Görsel Alanı:** Ham yüklenen görsel ile YuNet'in tespit ettiği kare yüz crop'unu yan yana gösterir.
2. **7 Sınıflı Bar Chart Dağılımı:** Kalibre edilmiş olasılık dağılımlarını anlık grafikle sunar.
3. **Webcam ve Canlı Çıkarım:** Kamera üzerinden anlık snapshot alma imkanı tanır.

---

## SLAYT 8: Öz-Değerlendirme (Self-Evaluation) ve Edge Case Analizi

### 💪 Güçlü Yönler
* **Süper Hızlı Çıkarım (3.32 ms):** Canlı akışta CPU tüketimi yaratmaz.
* **Yüksek Ayrıştırılabilirlik (0.9376 ROC-AUC):** Sınıfları birbirinden başarıyla ayırır.

### ⚠️ Başarısızlık Senaryoları (Edge Cases)
1. **Aşırı Kahkaha Kısıklığı:** Gözlerin kısılması FER2013 etiket gürültüsü nedeniyle nadiren `Sadness` veya `Disgust` ile karışabilmektedir.
2. **Yüz Kapanmaları (Occlusion):** El ile yüzün yarı yarıya kapatılması YuNet tespiti zorlaştırabilmektedir.

---

## SLAYT 9: Sonuç ve Kazanımlar (Conclusion & Takeaways)

### 🏁 Proje Sonucu ve Deneyimler
* 100 MB kısıtı altında **16.05 MB** boyut ve **3.32 ms** hız ile üretim ortamına (production) hazır canlı FER sistemi başarıyla geliştirilmiştir.
* **Kazanımlar:** Gerçek dünya verilerinde ön işlemenin (YuNet), loss fonksiyonu seçiminin (Focal Loss) ve kalibrasyonun (Temperature Scaling) model başarısındaki kritik rolü deneyimlenmiştir.

# Deliverable C: Demo Application & Visualization Guide (Demo Arayüz Dokümantasyonu)
**Proje:** Facial Emotion Recognition (FER) & Deployment Optimization  
**Tarih:** 25 Ağustos 2026  
**Durum:** Çalışır Durumda Test Edildi  

---

## 1. Genel Bakış

Deliverable C kapsamında geliştirilen canlı web arayüzü ve çıkarım (inference) sistemi, eğitilen **MobileNetV3 ONNX modelini** kullanıcı dostu bir grafiksel arayüz (GUI) ile sunar.

Arayüz **Gradio Blocks** mimarisi üzerinde inşa edilmiş olup aşağıdaki yetenekleri barındırır:
1. **Canlı YuNet Yüz Tespiti Kırpması:** Yüklenen resimde yüzü otomatik tespit ederek modele girdi sağlayan milimetrik yüz crop'unu sol/sağ sütunda gösterir.
2. **Kalibre Edilmiş Olasılık Dağılımı (Bar Chart):** 7 duygu sınıfının (`Anger`, `Disgust`, `Fear`, `Happiness`, `Sadness`, `Surprise`, `Neutral`) % olasılık değerlerini ve bar grafiğini sunar.
3. **Webcam ve Dosya Yükleme Desteği:** Kullanıcının hem bilgisayarından resim sürükleyip bırakmasına hem de canlı web kamerası görüntüsü almasına olanak tanır.

---

## 2. Çalıştırma Komutları ve Kullanım

### 2.1 Sanal Ortam ve Bağımlılıkların Hazırlanması
```bash
cd fer-project
source .venv/bin/activate
```

### 2.2 Demo Uygulamasını Başlatma

**ONNX Modeli ile Standart Web Arayüzünü Başlatma:**
```bash
python scripts/demo.py --model exports/mobilenetv3.onnx
```

**Webcam ve Paylaşılabilir Kamu Bağlantısı (Public Link) Açma:**
```bash
python scripts/demo.py --model exports/mobilenetv3.onnx --share
```

**Önceden Kırpılmış Görseller İçin Yüz Tespitini Devre Dışı Bırakma (Bypass Mode):**
```bash
python scripts/demo.py --model exports/mobilenetv3.onnx --bypass-face-detection
```

---

## 3. Ekran Arayüzü Bileşenleri ve Akış Grafiği

```
 +─────────────────────────────────────────────────────────────────────────+
 │                       🎭 Facial Emotion Recognition                      │
 ├────────────────────────────────────────┬────────────────────────────────┤
 │  [ Input Image / Webcam ]              │  [ Detected Face Crop (YuNet) ]│
 │  Girdi Görseli Yükleme Alanı           │  Modele Giren Hizalanmış Yüz   │
 │                                        ├────────────────────────────────┤
 │  [ Predict Emotion ] (Buton)           │  [ Probability Bar Chart ]     │
 │                                        │  • Happiness: 86.4%            │
 │                                        │  • Surprise:  10.2%            │
 │                                        │  • Neutral:    2.1%            │
 +────────────────────────────────────────┴────────────────────────────────+
```

---

## 4. Test Görselleri ve Örnek Çıkarım Sonuçları

Örnek test görselleri `fer-project/data/sample_images/` klasöründe yer almaktadır.
Test çalıştırması için:
```bash
python scripts/demo.py --model exports/mobilenetv3.onnx --port 7860
```
Tarayıcınızda `http://127.0.0.1:7860` adresine giderek testi doğrulayabilirsiniz.

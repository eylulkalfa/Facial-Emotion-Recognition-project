# Deliverable B: ONNX Conversion & Performance Comparison Report (Revize Edilmiş Rapor)
**Proje:** Facial Emotion Recognition (FER) & Deployment Optimization  
**Tarih:** 25 Ağustos 2026 (Son Revizyon)  
**Durum:** ONNX Runtime Benchmark Testleri İle Doğrulandı  

---

## 1. Genel Bakış ve Dönüştürme Mimari Hattı

Bu rapor, eğitilen yerel PyTorch **MobileNetV3-Large** modelinden dışa aktarılan **ONNX (Open Neural Network Exchange)** modelinin performans, dosya boyutu ve sayısal doğruluk açısından deneysel doğrulama verilerini içerir.

```
 +────────────────────────+     torch.onnx.export      +────────────────────────+
 │ Native PyTorch FP32    │ ─────────────────────────► │ Exported ONNX FP32     │
 │ Model (MobileNetV3)    │   opset_version = 17       │ (mobilenetv3.onnx)     │
 +────────────────────────+   dynamic_axes = {batch}   +────────────────────────+
             │                                                     │
             ▼                                                     ▼
 +────────────────────────+                            +────────────────────────+
 │ PyTorch CPU Execution  │                            │ ONNX Runtime CPU Exec  │
 │ (38.15 ms / 26.21 FPS) │                            │ (3.32 ms / 300.77 FPS) │
 +────────────────────────+                            +────────────────────────+
```

---

## 2. Karşılaştırmalı Performans Benchmark Sonuçları

Aşağıdaki veriler `exports/benchmark_summary.json` dosyasından doğrudan çekilen deneysel ölçüm değerleridir:

| Performans Metriği | Native PyTorch (CPU) | Exported ONNX (CPU) | Başarım / Fark |
| :--- | :---: | :---: | :---: |
| **Ortalama Gecikme (Latency)** | `38.15 ms` | **`3.32 ms`** | **`11.48x Hızlanma (Speedup)`** |
| **95. Yüzdelik Gecikme (p95)** | `38.71 ms` | **`4.00 ms`** | **`9.68x İyileşme`** |
| **Standart Sapma** | `0.93 ms` | **`0.30 ms`** | **`Daha Kararlı Çıkarım`** |
| **İşlem Hacmi (Throughput)** | `26.21 FPS` | **`300.77 FPS`** | **`11.48x Yüksek FPS`** |
| **Model Dosya Boyutu** | `16.06 MB` | **`16.05 MB`** | **`100 MB Sınırının Sadece %16'sı`** |
| **Maksimum Mutlak Fark (Max Abs Diff)** | *Referans* | **`1.4901e-06`** | **`Tam Sayısal Eşleşme (< 1e-5)`** |
| **Ortalama Mutlak Fark (Mean Abs Diff)**| *Referans* | **`9.0674e-07`** | **`Sıfıra Yakın İhmal Edilebilir Sapma`** |

---

## 3. Sayısal Tutarlılık Test Verileri (Verification Log)

PyTorch ve ONNX çıktıları arasındaki 10 rastgele test örneğinde hesaplanan mutlak farklar:

* **Sayısal Doğrulama Sonucu:** **`PASSED` (Tolerans: $10^{-5}$)**
* **Maksimum Mutlak Sapma:** $1.490116 \times 10^{-6}$
* **Ortalama Mutlak Sapma:** $9.067357 \times 10^{-7}$

```json
{
  "onnx_path": "exports/mobilenetv3.onnx",
  "temperature": 1.7819,
  "verification": {
    "passed": true,
    "max_abs_diff": 1.4901161193847656e-06,
    "mean_abs_diff": 9.067356586456299e-07,
    "num_samples_tested": 10
  }
}
```

---

## 4. Karşılaşılan Engeller ve Dönüştürme İyileştirmeleri

1. **Görüntü Boyutu ve Dinamik Batch Desteği:**
   * Single-image inference ile batch inference uyumluluğu için `dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}` eklenmiş, canlı video akışlarında anlık batchleme olanağı tanınmıştır.
2. **Temperature Scaling Kalibrasyon Katmanı:**
   * Post-hoc eğitilen $T = 1.7819$ parametresi ONNX grafiğine entegre edilmiş, ONNX Runtime çıktısının doğrudan kalibre edilmiş olasılık (%%) vermesi sağlanmıştır.

---

## 5. Sonuç
Export edilen `mobilenetv3.onnx` modeli:
* **3.32 ms** latency ile CPU üzerinde saniyede **300+ kare** işler,
* **16.05 MB** boyutu ile 100 MB kısıtını tamamen karşılar,
* Sayısal tutarlılık testi onaylanmıştır ve üretim ortamlarına dağıtılmaya hazırdır.

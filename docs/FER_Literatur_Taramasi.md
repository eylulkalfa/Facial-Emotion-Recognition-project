# Yüz İfadesinden Duygu Tanıma (FER) — Literatür Taraması

> **Proje odağı:** Gerçek dünya koşullarında çalışabilen, 100 MB altında kalan, ONNX formatına aktarılabilen ve duygu olasılık dağılımı üreten bir yüz ifadesi tanıma modeli geliştirmek.

## İçindekiler

- [1. Genel Bakış](#1-genel-bakış)
- [2. Temel Zorluklar](#2-temel-zorluklar)
- [3. FER Veri Kümeleri](#3-fer-veri-kümeleri)
- [4. Veri Stratejisi](#4-veri-stratejisi)
- [5. Derin Öğrenme Mimarileri](#5-derin-öğrenme-mimarileri)
- [6. Eğitim Stratejileri](#6-eğitim-stratejileri)
- [7. Değerlendirme Metrikleri](#7-değerlendirme-metrikleri)
- [8. Dağıtım ve ONNX Optimizasyonu](#8-dağıtım-ve-onnx-optimizasyonu)
- [9. Sonuç ve Önerilen Yaklaşım](#9-sonuç-ve-önerilen-yaklaşım)

---

## 1. Genel Bakış

Yüz İfadesinden Duygu Tanıma (**Facial Emotion Recognition — FER**), yüz görüntülerinden aşağıdaki duygusal durumların otomatik olarak sınıflandırılmasını amaçlayan bir bilgisayarlı görü problemidir:

- Mutluluk
- Üzüntü
- Öfke
- Korku
- Şaşkınlık
- İğrenme
- Nötr

Alanın ilk dönemlerinde yüz geometrisi, **Action Unit**'ler, **Local Binary Pattern (LBP)** ve **Gabor filtreleri** gibi elle tasarlanmış özellikler kullanılmıştır. FER2013 yarışması sonrasında evrişimli sinir ağları (**CNN**) temel yaklaşım hâline gelmiştir.

Son yıllarda ise şu mimari aileleri yaygınlaşmıştır:

- Dikkat mekanizmaları
- Vision Transformer (**ViT**) mimarileri
- CNN–Transformer hibritleri

Güncel çalışmalar, FER performansının yalnızca daha karmaşık modeller geliştirilmesine bağlı olmadığını göstermektedir. Başarı üzerinde aşağıdaki unsurlar da doğrudan etkilidir:

- Veri kalitesi
- Etiket tutarlılığı
- Sınıf dengesizliği
- Alan kayması
- Gerçek dünya koşullarına dayanıklılık

---

## 2. Temel Zorluklar

FER sistemlerinin karşılaştığı başlıca problemler şunlardır:

- Baş pozu değişiklikleri
- Yüzün kısmen kapanması
- Farklı aydınlatma koşulları
- Düşük görüntü çözünürlüğü
- Hareket bulanıklığı
- Kişiler arasındaki ifade farklılıkları
- Etiket gürültüsü
- Sınıf dengesizliği

Özellikle internet ortamından toplanan gerçek dünya veri kümelerinde etiket gürültüsü ve sınıf dengesizliği belirgindir. Mutluluk ve nötr sınıfları genellikle daha fazla örneğe sahipken, korku ve iğrenme sınıfları daha az temsil edilmekte ve birbirleriyle daha fazla karıştırılmaktadır.

Bu nedenle yalnızca genel doğruluk değerinin kullanılması, çoğunluk sınıflarına eğilimli modellerin başarılı görünmesine neden olabilir.

Literatürde aşağıdaki değerlendirme araçlarının birlikte kullanılması önerilmektedir:

- Macro-F1
- Sınıf bazlı precision ve recall
- Unweighted Average Recall (**UAR**)
- Karışıklık matrisi
- Model kalibrasyonu
- Tahmin güvenilirliği analizi

---

## 3. FER Veri Kümeleri

### 3.1. Veri Kümesi Karşılaştırması

| Veri kümesi | Temel özellikler | Avantajlar | Sınırlılıklar | Önerilen kullanım |
|---|---|---|---|---|
| **FER2013** | Yaklaşık 36 bin, 48×48 piksel, gri-seviyeli yüz görüntüsü; 7 duygu sınıfı | Kolay erişim, yaygın benchmark | Düşük çözünürlük, hatalı yüz kırpımları, gürültülü etiketler | Başlangıç deneyleri ve literatür karşılaştırması |
| **FER+** | FER2013 görüntülerinin çoklu değerlendirici oylarıyla yeniden etiketlenmiş sürümü | Yumuşak etiketlere ve duygu olasılıklarına uygun | FER2013 görüntü kalitesi sorunlarını devralır | Olasılık kalibrasyonu ve yardımcı değerlendirme |
| **RAF-DB** | İnternetten toplanmış; yaş, cinsiyet, etnik köken, poz ve aydınlatma çeşitliliği | Gerçek dünya çeşitliliği, görece güvenilir anotasyon | Veri erişimi ve kullanım koşulları ayrıca yönetilmelidir | İnce ayar ve ana benchmark |
| **AffectNet** | Yüz binlerce manuel etiketli görüntü; kategorik etiketler ile valans ve uyarılma değerleri | Büyük ölçek, yüksek çeşitlilik, transfer öğrenmeye uygun | Ciddi sınıf dengesizliği ve etiket belirsizliği | Ön eğitim |
| **CK+** | Kontrollü laboratuvar ortamı, poz verilmiş ifadeler | Temiz ve güvenilir etiketler | Sınırlı katılımcı çeşitliliği, düşük gerçek dünya temsili | Kontrollü deney ve ek değerlendirme |
| **JAFFE** | Kontrollü laboratuvar ortamı | Temiz veri ve güvenilir etiketler | Küçük ölçek ve sınırlı çeşitlilik | Yardımcı değerlendirme |
| **SFEW 2.0** | Film kareleri; yüksek poz, aydınlatma ve örtülme değişimi | Zorlu gerçek dünya koşulları | Küçük veri kümesi | Dış test ve genelleme analizi |

### 3.2. FER2013 ve FER+

FER2013, yaklaşık 36 bin adet 48×48 piksel gri-seviyeli yüz görüntüsünden oluşur ve yedi temel duygu sınıfını içerir. Kolay erişilebilir olması ve literatürde yaygın biçimde kullanılması önemli avantajlar sağlar.

Bununla birlikte aşağıdaki sorunlar gerçek dünya başarısını sınırlandırmaktadır:

- Düşük çözünürlük
- Hatalı yüz kırpımları
- Gürültülü etiketler

FER+, FER2013 görüntülerini yeniden etiketleyen ve her görüntü için birden fazla değerlendiricinin oylarını sunan geliştirilmiş bir sürümdür. Bu yapı, tek bir kesin etiket yerine duygu olasılıklarının veya yumuşak hedeflerin kullanılmasına olanak verir.

Bu nedenle FER+, olasılık dağılımı üretmesi beklenen FER sistemleri için özellikle değerlidir.

### 3.3. RAF-DB

RAF-DB, internet ortamından elde edilen ve aşağıdaki açılardan çeşitlilik gösteren yüz görüntülerinden oluşur:

- Yaş
- Cinsiyet
- Etnik köken
- Baş pozu
- Aydınlatma koşulları

Temel ifade alt kümesi yedi duygu sınıfı içerir ve görüntüler çok sayıda değerlendirici tarafından etiketlenir.

RAF-DB, gerçek dünya çeşitliliği ve görece güvenilir anotasyon yapısı bakımından güncel FER çalışmalarında önemli bir benchmark olarak kabul edilmektedir.

### 3.4. AffectNet

AffectNet, yüz binlerce manuel etiketli yüz görüntüsüyle FER alanındaki en büyük gerçek dünya veri kümelerinden biridir.

Veri kümesi şunları sağlar:

- Kategorik duygu etiketleri
- Valans değerleri
- Uyarılma değerleri

Büyük ölçeği ve çeşitliliği, AffectNet'i transfer öğrenme ve geniş kapsamlı ön eğitim için güçlü bir kaynak hâline getirir.

Bununla birlikte:

- Ciddi sınıf dengesizliği bulunmaktadır.
- Belirli düzeyde etiket belirsizliği vardır.

Bu nedenle AffectNet'in doğrudan nihai eğitim kümesi olarak kullanılmasındansa, ön eğitim amacıyla kullanılması ve modelin daha düzenli bir veri kümesi üzerinde ince ayarlanması daha uygun görülmektedir.

### 3.5. Kontrollü ve Gerçek Dünya Veri Kümeleri

CK+ ve JAFFE gibi kontrollü laboratuvar veri kümeleri daha temiz ve güvenilir etiketlere sahiptir. Ancak poz verilmiş ifadeler ve sınırlı katılımcı çeşitliliği nedeniyle gerçek dünya koşullarını yeterince temsil etmezler.

Bu veri kümelerinde elde edilen yüksek doğruluk değerleri, saha başarısını olduğundan yüksek gösterebilir.

SFEW 2.0 ise film karelerinden oluşturulmuştur ve aşağıdaki değişimlerin yüksek olduğu zor bir veri kümesidir:

- Baş pozu
- Aydınlatma
- Örtülme

Boyutunun küçük olması nedeniyle ana eğitim kümesi olarak uygun değildir. Buna karşın gerçek dünya genellemesini değerlendiren dış test kümesi olarak kullanılabilir.

---

## 4. Veri Stratejisi

Literatürde farklı veri kümelerinin doğrudan tek bir klasör altında birleştirilmesinin her zaman performans artışı sağlamadığı belirtilmektedir.

Veri kümeleri arasında şu farklılıklar bulunabilir:

- Duygu tanımları
- Etiketleme kriterleri
- Görüntü kalitesi
- Sınıf dağılımları
- Veri toplama ortamları

Bu nedenle doğrudan birleştirme, veri alanları ve etiket standartları arasında uyumsuzluk oluşturabilir.

### Önerilen veri akışı

1. **AffectNet üzerinde ön eğitim**
2. **RAF-DB üzerinde ince ayar**
3. **FER+ ile yumuşak etiket, kalibrasyon veya yardımcı değerlendirme**
4. **SFEW 2.0 ile dış test ve gerçek dünya genelleme analizi**

Veri kümelerinin doğrudan birleştirilmesi gerekiyorsa aşağıdaki işlemler uygulanmalıdır:

- Tüm sınıfların ortak bir etiket uzayına eşlenmesi
- Belirsiz örneklerin çıkarılması
- Yüz içermeyen görüntülerin temizlenmesi
- Dosya hash yöntemleriyle çift kayıtların bulunması
- Perceptual hash yöntemleriyle görsel kopyaların temizlenmesi

---

## 5. Derin Öğrenme Mimarileri

FER çalışmalarında kullanılan başlıca mimariler şunlardır:

- ResNet
- VGG
- DenseNet
- MobileNet
- EfficientNet
- ConvNeXt
- Vision Transformer
- Swin Transformer
- MobileViT
- Dikkat tabanlı hibrit modeller

Transformer tabanlı modeller, küresel yüz ilişkilerini modelleme açısından yüksek kapasite sunar. Ancak genellikle:

- Daha fazla eğitim verisine,
- Daha yüksek hesaplama gücüne,
- Daha karmaşık dağıtım süreçlerine

ihtiyaç duyar.

Küçük ve orta ölçekli FER veri kümelerinde CNN'lerin yerel yüz özelliklerini öğrenmeye yönelik yapısal eğilimi önemli bir avantaj sağlamaya devam etmektedir.

### 5.1. Aday Model Karşılaştırması

| Model | Yaklaşık parametre | Yaklaşık FP32 boyutu | Güçlü yön | Temel risk | Projedeki rol |
|---|---:|---:|---|---|---|
| **MobileNetV3-Large** | 5,5 milyon | 22 MB | Düşük gecikme, düşük maliyet, kolay ONNX aktarımı | Kapasitesi daha büyük modellere göre sınırlı olabilir | Ana model adayı |
| **EfficientNet-B0** | 5,3 milyon | 21 MB | Güçlü doğruluk–boyut dengesi | MobileNetV3'e göre gecikme daha yüksek olabilir | Ana model adayı |
| **MobileViT-XS** | 2,3 milyon | Çok küçük | Yerel ve küresel özellikleri birleştirir | FER ve endüstriyel dağıtım literatürü daha az olgun | Deneysel üçüncü model |
| **ConvNeXt-Tiny** | Daha yüksek | Yaklaşık 100 MB sınırını aşabilir | Güçlü temsil kapasitesi | Model boyutu ve dağıtım maliyeti | Düşük öncelik |
| **Swin-Tiny** | Daha yüksek | Yaklaşık 100 MB sınırını aşabilir | Güçlü küresel modelleme | Model boyutu, gecikme ve ONNX karmaşıklığı | Düşük öncelik |
| **ViT-Small** | Değişken | Teorik olarak sınır altında olabilir | Küresel ilişkileri modelleme | Çıkarım gecikmesi ve ONNX dağıtım riski | Alternatif deney |

### 5.2. MobileNetV3

MobileNetV3, düşük parametre sayısı, düşük hesaplama maliyeti ve standart evrişim tabanlı yapısı sayesinde mobil ve edge uygulamalarında yaygın biçimde tercih edilmektedir.

**MobileNetV3-Large:**

- Yaklaşık 5,5 milyon parametre
- Yaklaşık 22 MB FP32 ağırlık boyutu
- 100 MB sınırının oldukça altında
- Genellikle sorunsuz ONNX aktarımı

FER çalışmalarında MobileNetV3 üzerine hafif dikkat veya normalizasyon modülleri eklenerek başarılı sonuçlar elde edildiği görülmektedir.

### 5.3. EfficientNet

EfficientNet-B0 ve B1, model derinliği, genişliği ve görüntü çözünürlüğünü birlikte ölçekleyen **compound scaling** yaklaşımını kullanır.

**EfficientNet-B0:**

- Yaklaşık 5,3 milyon parametre
- Yaklaşık 21 MB model boyutu
- Güçlü doğruluk–boyut dengesi
- Transfer öğrenme için güvenilir başlangıç modeli

B1 sürümü daha yüksek kapasite sağlasa da hesaplama ve gecikme maliyeti artmaktadır.

### 5.4. MobileViT

MobileViT, evrişim katmanlarının yerel özellik çıkarma yeteneğini Transformer bloklarının küresel bağlam modelleme gücüyle birleştirir.

**MobileViT-XS:**

- Yaklaşık 2,3 milyon parametre
- Son derece küçük model boyutu
- Hafif FER çalışmaları için umut verici
- Ana modelden çok deneysel karşılaştırma modeli olmaya uygun

Endüstriyel dağıtım ve literatür olgunluğu açısından MobileNetV3 ve EfficientNet kadar yaygın değildir.

---

## 6. Eğitim Stratejileri

### 6.1. Transfer Öğrenme

FER çalışmalarında transfer öğrenme, sınırlı veri ve kısa eğitim süresi nedeniyle yaygın olarak kullanılmaktadır.

ImageNet üzerinde önceden eğitilmiş bir omurganın kullanılması, yüz ifadelerine ait düşük ve orta seviyeli görsel özelliklerin daha hızlı öğrenilmesini sağlar.

Önerilen eğitim akışı:

1. ImageNet ön eğitimli modelin yüklenmesi
2. AffectNet üzerinde alan odaklı ön eğitim
3. RAF-DB üzerinde ince ayar
4. FER+ ile yumuşak hedef veya kalibrasyon deneyi
5. SFEW 2.0 üzerinde dış test

### 6.2. Sınıf Dengesizliği

Sınıf dengesizliğiyle mücadele etmek için kullanılabilecek yöntemler:

- Ağırlıklı çapraz entropi
- Focal loss
- Dengeli örnekleme
- Sınıfa duyarlı veri artırma

Aşırı örnekleme, küçük sınıflarda ezberlemeye neden olabilir. Bu nedenle kayıp fonksiyonuna dayalı dengeleme yöntemleri çoğu durumda daha güvenlidir.

### 6.3. Veri Artırma

Uygun veri artırma teknikleri:

- Yatay çevirme
- Hafif döndürme
- Kırpma
- Parlaklık değişimi
- Kontrast değişimi

Kaçınılması gereken dönüşümler:

- Aşırı döndürme
- Agresif perspektif dönüşümü
- Yoğun kesme işlemleri
- Yüz geometrisini bozan dönüşümler
- Duygu anlamını değiştirebilecek dönüşümler

---

## 7. Değerlendirme Metrikleri

Değerlendirme sürecinde yalnızca accuracy değerine odaklanılmamalıdır.

| Amaç | Önerilen metrik |
|---|---|
| Genel sınıflandırma başarısı | Accuracy, weighted-F1 |
| Sınıflar arasında dengeli başarı | Macro-F1, UAR |
| Sınıf bazlı analiz | Precision, recall, F1-score |
| Hata analizi | Karışıklık matrisi |
| Olasılık güvenilirliği | Expected Calibration Error, Brier skoru |
| Görsel kalibrasyon analizi | Güvenilirlik diyagramı |

Modelin yüzde tabanlı duygu olasılıkları göstermesi beklendiğinden, softmax çıktılarının kalibrasyonu ayrıca değerlendirilmelidir.

### Raporlanması önerilen minimum metrik seti

- Accuracy
- Weighted-F1
- Macro-F1
- UAR
- Sınıf bazlı precision
- Sınıf bazlı recall
- Karışıklık matrisi
- Expected Calibration Error
- Brier skoru

---

## 8. Dağıtım ve ONNX Optimizasyonu

Projenin temel dağıtım gereksinimleri:

- Model boyutu **100 MB altında** olmalıdır.
- Model **ONNX** formatına aktarılmalıdır.
- Model bir demo uygulamasında çalışmalıdır.
- Çıktı olarak duygu olasılık dağılımı üretmelidir.

Mimari seçiminde yalnızca doğruluk değil, aşağıdaki özellikler birlikte değerlendirilmelidir:

- Parametre sayısı
- Model dosya boyutu
- Tek görüntü çıkarım gecikmesi
- Ortalama çıkarım gecikmesi
- CPU bellek tüketimi
- ONNX operatör uyumluluğu

### 8.1. Sayısal Tutarlılık Kontrolü

ONNX dönüşümünden sonra yerel PyTorch veya TensorFlow modeli ile ONNX Runtime modeli arasında sayısal tutarlılık kontrol edilmelidir.

Aynı giriş görüntüsü için şu değerler karşılaştırılmalıdır:

- Logit değerleri
- Softmax olasılıkları
- Maksimum mutlak fark
- Ortalama mutlak fark

### 8.2. Performans Ölçümleri

Dağıtım sonrası raporlanması önerilen ölçümler:

- Model dosya boyutu
- Tek görüntü çıkarım gecikmesi
- Ortalama çıkarım gecikmesi
- Standart sapma veya gecikme dağılımı
- CPU bellek tüketimi
- ONNX Runtime sağlayıcısı
- Kullanılan giriş görüntüsü boyutu

### 8.3. Nicemleme

Aşağıdaki nicemleme teknikleri model boyutunu ve çıkarım süresini azaltabilir:

- FP16
- INT8

Ancak nicemleme sonrasında şu etkiler ayrıca doğrulanmalıdır:

- Accuracy değişimi
- Macro-F1 değişimi
- Sınıf bazlı performans değişimi
- Olasılık kalibrasyonu
- PyTorch/TensorFlow–ONNX çıktı tutarlılığı

---

## 9. Sonuç ve Önerilen Yaklaşım

Literatür bulguları ve proje kısıtları birlikte değerlendirildiğinde, en düşük riskli ana omurga adayları şunlardır:

1. **MobileNetV3-Large**
2. **EfficientNet-B0**

MobileNetV3-Large daha düşük gecikme ve kolay dağıtım avantajı sunmaktadır. EfficientNet-B0 ise doğruluk ve model kapasitesi bakımından daha dengeli bir seçenek oluşturmaktadır.

**MobileViT-XS**, hibrit mimarilerin etkisini incelemek için deneysel üçüncü model olarak değerlendirilebilir.

### Önerilen deney planı

| Aşama | Öneri |
|---|---|
| Ön eğitim | AffectNet |
| İnce ayar | RAF-DB |
| Yumuşak etiket / kalibrasyon | FER+ |
| Dış test | SFEW 2.0 |
| Ana model 1 | MobileNetV3-Large |
| Ana model 2 | EfficientNet-B0 |
| Deneysel model | MobileViT-XS |
| Dağıtım formatı | ONNX |
| Boyut sınırı | 100 MB |
| Temel başarı metriği | Macro-F1 + accuracy |
| Kalibrasyon metriği | ECE + Brier skoru |

### Nihai öneri

Veri stratejisi bakımından:

> **AffectNet üzerinde ön eğitim, RAF-DB üzerinde ince ayar ve FER+ üzerinde olasılık kalibrasyonu veya yardımcı değerlendirme**

yaklaşımı; veri çeşitliliği, etiket güvenilirliği ve dağıtım maliyeti arasında dengeli bir çözüm sunmaktadır.

Model tarafında ilk karşılaştırmanın **MobileNetV3-Large** ve **EfficientNet-B0** arasında yapılması, **MobileViT-XS** modelinin ise deneysel alternatif olarak değerlendirilmesi önerilmektedir.

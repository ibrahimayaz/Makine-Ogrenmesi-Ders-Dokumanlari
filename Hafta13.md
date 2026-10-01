# Hafta 13 — Derin Öğrenmeye Bakış

## Bu Haftanın Hedefleri

- Görüntülerde tam bağlı ağların neden yetersiz kaldığını anlamak
- **Evrişim** (convolution) işlemini elle hesaplamak; filtre, adım (stride) ve dolgu (padding) kavramlarını öğrenmek
- **Havuzlama** (pooling) katmanının rolünü görmek
- Bir CNN'in katman katman nasıl öznitelik hiyerarşisi öğrendiğini anlamak
- **Transfer öğrenme** ile az veriyle güçlü modeller kurmak
- Dizi modelleri (RNN, LSTM) ve **Transformer**'ın temel fikri olan dikkat (attention) mekanizmasını tanımak
- Önceden eğitilmiş bir CNN ile görüntü sınıflandırma yapmak

---

# BÖLÜM A: Evrişimli Sinir Ağları (CNN)

## 1. Görüntüler İçin Neden Özel Bir Mimari?

### 1.1 Bilgisayar bir görüntüyü nasıl görür?

Bir görüntü, sayılardan oluşan bir **tensördür**:
- Gri tonlamalı 28×28 görüntü: 28 × 28 matris, her piksel 0 (siyah) – 255 (beyaz)
- Renkli 224×224 görüntü: **224 × 224 × 3** (kırmızı, yeşil, mavi kanallar)

### 1.2 MLP'nin sorunları

Hafta 12'de Fashion-MNIST görüntülerini düzleştirip (flatten) MLP'ye verdik. Bu yaklaşımın sorunları:

**1. Parametre patlaması:** 224×224×3 bir görüntü = 150.528 girdi. İlk gizli katmanda 1.000 nöron olsa:

```
150.528 × 1.000 = 150 milyon parametre — sadece ilk katmanda!
```

**2. Uzamsal yapının kaybolması:** Düzleştirince komşu pikseller arasındaki ilişki kaybolur. MLP için piksel (10, 10) ile (10, 11) komşu değildir; sadece farklı iki girdidir.

**3. Konuma bağımlılık:** Görüntünün sol üst köşesindeki bir kediyi tanımayı öğrenen MLP, sağ alt köşedeki kediyi tanımak için **tamamen farklı ağırlıklar** öğrenmek zorundadır.

### 1.3 CNN'in üç temel fikri

| Fikir | Açıklama | Sonuç |
|---|---|---|
| **Yerel bağlantı** | Her nöron görüntünün sadece küçük bir bölgesine (ör. 3×3) bakar | Kenar, köşe gibi yerel desenler yakalanır |
| **Ağırlık paylaşımı** | Aynı filtre görüntünün **her yerinde** kullanılır | Parametre sayısı çok azalır |
| **Öteleme eşdeğerliği** | Desen kayarsa, çıktıdaki tepki de aynı miktarda kayar | Nesne nerede olursa olsun tanınır |

---

## 2. Evrişim (Convolution) İşlemi

### 2.1 Mekanizma

Küçük bir **filtre (kernel)** — örneğin 3×3'lük bir ağırlık matrisi — görüntünün üzerinde **kaydırılır**. Her konumda filtre ile altındaki görüntü parçası **eleman eleman çarpılıp toplanır**. Sonuç, çıktıdaki tek bir pikseldir. Tüm çıktıya **öznitelik haritası (feature map)** denir.

```
   Girdi (5×5)                  Filtre (3×3)        Çıktı (3×3)
 ┌───┬───┬───┬───┬───┐
 │ a │ b │ c │   │   │         ┌───┬───┬───┐       ┌───┬───┬───┐
 ├───┼───┼───┼───┼───┤         │w₁ │w₂ │w₃ │       │ ● │   │   │
 │ d │ e │ f │   │   │    ⊛    ├───┼───┼───┤   =   ├───┼───┼───┤
 ├───┼───┼───┼───┼───┤         │w₄ │w₅ │w₆ │       │   │   │   │
 │ g │ h │ i │   │   │         ├───┼───┼───┤       ├───┼───┼───┤
 ├───┼───┼───┼───┼───┤         │w₇ │w₈ │w₉ │       │   │   │   │
 │   │   │   │   │   │         └───┴───┴───┘       └───┴───┴───┘
 ├───┼───┼───┼───┼───┤
 │   │   │   │   │   │         ● = a·w₁ + b·w₂ + c·w₃ + d·w₄ + ... + i·w₉ + bias
 └───┴───┴───┴───┴───┘
```

### 2.2 Elle örnek: Dikey kenar tespiti

**Girdi (6×6):** Sol yarısı parlak (10), sağ yarısı karanlık (0):

```
10 10 10  0  0  0
10 10 10  0  0  0
10 10 10  0  0  0
10 10 10  0  0  0
10 10 10  0  0  0
10 10 10  0  0  0
```

**Filtre (3×3) — dikey kenar dedektörü:**

```
 1  0  −1
 1  0  −1
 1  0  −1
```

**Sol üst konum (satır 0–2, sütun 0–2):**

```
10·1 + 10·0 + 10·(−1)
+ 10·1 + 10·0 + 10·(−1)      =  0     (düz bölge: değişim yok)
+ 10·1 + 10·0 + 10·(−1)
```

**Bir sağa kaydır (sütun 1–3):**

```
10·1 + 10·0 + 0·(−1)
+ 10·1 + 10·0 + 0·(−1)       = 30     (kenar! soldan sağa parlaklık düşüyor)
+ 10·1 + 10·0 + 0·(−1)
```

**Tam çıktı (4×4):**

```
 0  30  30  0
 0  30  30  0
 0  30  30  0
 0  30  30  0
```

Çıktıdaki yüksek değerler (30), görüntüdeki **dikey kenarın** tam yerini gösteriyor. Filtre bir "desen dedektörü" gibi çalışır: Aradığı desenle eşleşen bölgelerde güçlü tepki verir.

```python
import numpy as np

def evrisim2d(goruntu, filtre, adim=1, dolgu=0):
    if dolgu > 0:
        goruntu = np.pad(goruntu, dolgu)                  # kenarlara sıfır ekle
    H, W = goruntu.shape
    k = filtre.shape[0]
    cH, cW = (H - k) // adim + 1, (W - k) // adim + 1
    cikti = np.zeros((cH, cW))
    for i in range(cH):
        for j in range(cW):
            parca = goruntu[i * adim:i * adim + k, j * adim:j * adim + k]
            cikti[i, j] = np.sum(parca * filtre)          # eleman eleman çarp, topla
    return cikti

goruntu = np.array([[10, 10, 10, 0, 0, 0]] * 6, dtype=float)
dikey = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
print(evrisim2d(goruntu, dikey))
```

> **Not:** Matematikte "evrişim" filtrenin 180° çevrilmesini içerir. Derin öğrenme kütüphaneleri aslında çevirmeden **çapraz korelasyon** yapar ama yine de "evrişim" der. Filtre zaten öğrenildiği için fark etmez.

### 2.3 Klasik filtreler

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").mean(axis=2)[::2, ::2]   # gri tonlama, küçült

filtreler = {
    "Dikey kenar (Sobel)": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    "Yatay kenar (Sobel)": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "Bulanıklaştırma": np.ones((3, 3)) / 9,
    "Keskinleştirme": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Orijinal")
for a, (ad, f) in zip(ax[1:], filtreler.items()):
    a.imshow(evrisim2d(img, f.astype(float)), cmap="gray"); a.set_title(ad)
for a in ax: a.axis("off")
plt.show()
```

**CNN'in devrimci fikri:** Bu filtreleri insanlar elle tasarlamaz. Filtre ağırlıkları, geri yayılım ile **veriden öğrenilir**. Ağ, görevi için hangi desenlerin faydalı olduğunu kendisi keşfeder.

### 2.4 Dolgu (padding) ve adım (stride)

**Dolgu (padding):** Görüntünün kenarlarına sıfırlar eklenir.
- Dolgu olmadan çıktı her katmanda küçülür (6×6 → 4×4) ve kenar pikselleri daha az "görülür".
- **"same" dolgu:** 3×3 filtre için 1 piksel dolgu → çıktı girdiyle aynı boyutta kalır.

**Adım (stride):** Filtrenin her seferinde kaç piksel kaydığı. Adım 2 → çıktı yaklaşık yarı boyuta iner.

**Çıktı boyutu formülü:**

```
         W − K + 2P
Çıktı = ──────────── + 1        W: girdi boyutu, K: filtre boyutu, P: dolgu, S: adım
             S
```

| Girdi | Filtre | Dolgu | Adım | Çıktı |
|---|---|---|---|---|
| 6 | 3 | 0 | 1 | (6 − 3 + 0)/1 + 1 = **4** |
| 6 | 3 | 1 | 1 | (6 − 3 + 2)/1 + 1 = **6** ("same") |
| 32 | 5 | 0 | 1 | **28** |
| 224 | 7 | 3 | 2 | (224 − 7 + 6)/2 + 1 = **112** |

### 2.5 Çok kanallı evrişim

Renkli bir görüntüde 3 kanal vardır. Bu durumda filtre de **3 boyutludur**: 3×3×3. Her kanal kendi 3×3'lük dilimiyle evrişilir ve sonuçlar **toplanır** → tek bir öznitelik haritası.

Bir katmanda genellikle **birçok filtre** (ör. 64) kullanılır; her biri farklı bir desen arar. Çıktı: 64 kanallı bir tensör.

```
 Girdi: H × W × 3   ──[64 filtre, her biri 3×3×3]──►   Çıktı: H × W × 64
```

**Parametre sayısı:** 64 filtre × (3 × 3 × 3 ağırlık + 1 bias) = 64 × 28 = **1.792**

Karşılaştırın: Aynı 224×224×3 girdi için tam bağlı katman 150 milyon parametre gerektiriyordu. Ağırlık paylaşımının gücü!

---

## 3. Havuzlama (Pooling)

Havuzlama, öznitelik haritasını **küçültür** (alt örnekleme). Öğrenilecek parametresi yoktur.

**Maksimum havuzlama (max pooling), 2×2, adım 2:**

```
 Girdi (4×4)                       Çıktı (2×2)
 ┌───┬───┬───┬───┐
 │ 1 │ 3 │ 2 │ 1 │                 ┌───┬───┐
 │ 4 │ 6 │ 5 │ 0 │    max 2×2      │ 6 │ 5 │
 ├───┼───┼───┼───┤   ─────────►    ├───┼───┤
 │ 3 │ 1 │ 1 │ 2 │                 │ 3 │ 4 │
 │ 0 │ 2 │ 4 │ 3 │                 └───┴───┘
 └───┴───┴───┴───┘
```

- Sol üst blok {1, 3, 4, 6} → **6**
- Sağ üst blok {2, 1, 5, 0} → **5**
- Sol alt blok {3, 1, 0, 2} → **3**
- Sağ alt blok {1, 2, 4, 3} → **4**

**Ortalama havuzlama** aynı bloklar için: 3.5, 2.0, 1.5, 2.5

**Neden havuzlama?**
1. **Hesaplama ve bellek** azalır (her boyut yarıya → 4 kat küçülme)
2. **Küçük kaymalara dayanıklılık:** Kenar bir piksel kaysa bile bloğun maksimumu genelde değişmez
3. **Alıcı alan (receptive field) büyür:** Sonraki katmanlar görüntünün daha geniş bir bölgesini "görür"

```python
def max_havuz(X, k=2):
    H, W = X.shape
    return X[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).max(axis=(1, 3))

X = np.array([[1, 3, 2, 1], [4, 6, 5, 0], [3, 1, 1, 2], [0, 2, 4, 3]])
print(max_havuz(X))     # [[6 5] [3 4]]
```

---

## 4. CNN Mimarisi

### 4.1 Tipik yapı

```
Girdi → [Evrişim → ReLU → (Evrişim → ReLU) → Havuzlama] × N → Düzleştir → Tam bağlı → Softmax
         └─────────── öznitelik çıkarıcı (feature extractor) ───────────┘   └─ sınıflandırıcı ─┘
```

Derinlere doğru:
- Uzamsal boyut **küçülür** (224 → 112 → 56 → 28 → 14 → 7)
- Kanal sayısı **artar** (3 → 64 → 128 → 256 → 512)

### 4.2 Öznitelik hiyerarşisi

CNN'ler katman katman giderek soyutlaşan öznitelikler öğrenir:

```
 Katman 1          Katman 2-3           Katman 4-5            Son katmanlar
 ─────────         ──────────           ──────────            ─────────────
 kenarlar          dokular, köşeler     nesne parçaları       bütün nesneler
  │ ─ ╱ ╲          ◠ ▦ ░ ∷               👁 👃 🐾 🛞             🐱 🐶 🚗
```

Bu hiyerarşi, **transfer öğrenmenin** (Bölüm 6) neden işe yaradığını açıklar: İlk katmanlardaki kenar ve doku dedektörleri hemen her görsel görevde faydalıdır.

### 4.3 PyTorch ile basit bir CNN

```python
import torch
import torch.nn as nn

class BasitCNN(nn.Module):
    def __init__(self, sinif_sayisi=10):
        super().__init__()
        self.ozellik = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28×1  → 28×28×32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                               # → 14×14×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # → 14×14×64
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                               # → 7×7×64
        )
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),                                  # → 3136
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, sinif_sayisi),
        )

    def forward(self, x):
        return self.siniflandirici(self.ozellik(x))

model = BasitCNN()
x = torch.randn(8, 1, 28, 28)                 # 8 görüntülük sahte bir batch
print(model(x).shape)                          # torch.Size([8, 10])
print("Parametre:", sum(p.numel() for p in model.parameters()))
```

Hafta 12'deki eğitim döngüsü (`egit` fonksiyonu) bu modelle **aynen** kullanılabilir. Fashion-MNIST'te MLP ~%88–89 verirken bu küçük CNN ~%91–92'ye ulaşır.

### 4.4 Önemli mimariler (tarihsel gelişim)

| Yıl | Mimari | Yenilik | ImageNet top-5 hata |
|---|---|---|---|
| 1998 | **LeNet-5** | İlk başarılı CNN (el yazısı rakamlar) | — |
| 2012 | **AlexNet** | ReLU, dropout, GPU ile eğitim → derin öğrenme devrimi | %15.3 |
| 2014 | **VGG** | Sadece 3×3 filtreler, çok derin (16–19 katman) | %7.3 |
| 2014 | **GoogLeNet/Inception** | Paralel farklı boyutlu filtreler | %6.7 |
| 2015 | **ResNet** | **Artık bağlantılar** → 152 katman | %3.6 |
| 2019 | **EfficientNet** | Derinlik/genişlik/çözünürlüğü dengeli ölçekleme | — |
| 2020 | **Vision Transformer (ViT)** | Görüntüyü yamalara bölüp Transformer ile işleme | — |

**ResNet'in artık bağlantısı (residual / skip connection):**

```
     x ──────────────────────╮
     │                        │ (kısayol: x olduğu gibi aktarılır)
     ▼                        │
  [Evrişim → BN → ReLU]       │
     ▼                        │
  [Evrişim → BN]              │
     ▼                        │
     ⊕ ◄──────────────────────╯
     ▼
   ReLU     çıktı = F(x) + x
```

Katman, girdinin tamamını değil sadece **farkı (artığı)** F(x) öğrenir. Gradyan, kısayol üzerinden doğrudan geriye akabildiği için **kaybolan gradyan** sorunu (Hafta 12) büyük ölçüde çözülür ve yüzlerce katmanlı ağlar eğitilebilir hale gelir.

---

## 5. Veri Artırma (Data Augmentation)

Elimizdeki görüntülerden **gerçekçi varyasyonlar** üreterek eğitim setini yapay olarak büyütmek:

```
 Orijinal 🐱 → döndürülmüş, aynalanmış, kırpılmış, parlaklığı değiştirilmiş 🐱 🐱 🐱 🐱
```

- Yatay çevirme, küçük döndürme, rastgele kırpma, ölçekleme
- Parlaklık/kontrast/renk değişimi
- Bunlar **sadece eğitim** sırasında, her epoch'ta rastgele uygulanır; doğrulama/test görüntüleri değiştirilmez.

Dikkat: Dönüşüm etiketi değiştirmemeli. Rakamlarda 180° döndürme "6"yı "9" yapar!

```python
from torchvision import transforms
egitim_donusum = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## 6. Transfer Öğrenme (Transfer Learning)

### 6.1 Fikir

ImageNet (1,2 milyon görüntü, 1.000 sınıf) üzerinde günlerce eğitilmiş bir ağ, genel görsel öznitelikleri (kenarlar, dokular, şekiller, nesne parçaları) zaten öğrenmiştir. Kendi küçük veri setimiz (ör. 500 görüntü) için **sıfırdan başlamak yerine bu bilgiyi yeniden kullanırız**.

Benzetme: Bisiklet sürmeyi bilen biri motosiklet kullanmayı sıfırdan başlayan birinden çok daha hızlı öğrenir.

### 6.2 İki strateji

```
 Önceden eğitilmiş ağ (ör. ResNet-18):
 [Evrişim blokları ............................]  [FC 1000 sınıf]
                                                        │
                                                        ▼ değiştir
 Bizim ağımız:
 [Evrişim blokları ............................]  [FC: bizim sınıf sayımız]

 Strateji 1 — Öznitelik çıkarma:  ❄ dondurulmuş (eğitilmez)        🔥 eğitilir
 Strateji 2 — İnce ayar:          ❄ ilk katmanlar  🔥 son bloklar   🔥 eğitilir
```

| Strateji | Ne yapılır? | Ne zaman? |
|---|---|---|
| **Öznitelik çıkarma** (feature extraction) | Tüm evrişim katmanları dondurulur; sadece yeni sınıflandırıcı katman eğitilir | Az veri; veriler ImageNet'e benzer |
| **İnce ayar** (fine-tuning) | Yeni katman eğitildikten sonra son bloklar da **küçük öğrenme oranıyla** eğitilir | Orta/çok veri; veriler farklı (tıbbi görüntü vb.) |

**İnce ayarda küçük öğrenme oranı neden önemli?** Büyük adımlar, önceden öğrenilmiş değerli ağırlıkları bozar ("felaket unutma"). Tipik: Yeni katman için 1e-3, önceden eğitilmiş katmanlar için 1e-4 – 1e-5.

**Önemli:** Önceden eğitilmiş modelin beklediği ön işleme **aynen** uygulanmalı (aynı görüntü boyutu ve aynı ortalama/std normalizasyonu).

---

# BÖLÜM B: Dizi Modelleri ve Transformer'a Kısa Giriş

## 7. Dizi Verisi

Metin, konuşma, zaman serileri, DNA dizileri... Bu verilerde **sıra önemlidir** ve uzunluk değişkendir.

> "Köpek adamı ısırdı" ≠ "Adam köpeği ısırdı"

MLP ve CNN sabit boyutlu girdi bekler ve sıra bilgisini doğal olarak modellemez.

## 8. Tekrarlayan Sinir Ağları (RNN)

### 8.1 Fikir: Hafıza

RNN, diziyi **adım adım** işler ve bir **gizli durum (hidden state)** hₜ taşır. Bu durum, o ana kadar görülenlerin özetidir:

```
hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)
```

```
 "Ben"     "okula"     "gidiyorum"
   x₁        x₂           x₃
   │         │            │
   ▼         ▼            ▼
 [RNN] ──► [RNN] ──► [RNN] ──► h₃ → tahmin
   h₁   aynı   h₂  aynı
       ağırlıklar   ağırlıklar
```

Tüm zaman adımlarında **aynı ağırlıklar** kullanılır (CNN'deki ağırlık paylaşımına benzer, ama zaman ekseninde).

### 8.2 Sorunlar

- **Uzun vadeli bağımlılıklar:** Geri yayılım zaman boyunca (BPTT) gradyanları her adımda çarptığı için gradyanlar **kaybolur veya patlar**. RNN, 50 kelime önceki bir bilgiyi hatırlamakta zorlanır.
  > "**Fransa**'da büyüdüm, ... (uzun bir paragraf) ... bu yüzden akıcı bir şekilde **Fransızca** konuşurum."
- **Sıralı hesaplama:** hₜ, hₜ₋₁'e bağlı → paralelleştirilemez → yavaş eğitim.

### 8.3 LSTM ve GRU

**LSTM (Long Short-Term Memory)**, ayrı bir **hücre durumu (cell state)** ve bilgiyi kontrol eden **kapılar (gates)** ekler:

| Kapı | Görevi |
|---|---|
| Unutma kapısı | Eski bilginin ne kadarı silinsin? |
| Girdi kapısı | Yeni bilginin ne kadarı eklensin? |
| Çıktı kapısı | Hücre durumunun ne kadarı dışarı verilsin? |

Hücre durumu, bilginin neredeyse değişmeden uzun süre akabildiği bir "taşıyıcı bant" gibidir (ResNet'teki kısayol fikrine benzer). **GRU**, LSTM'in daha basit (iki kapılı) bir versiyonudur.

## 9. Dikkat Mekanizması ve Transformer

### 9.1 Dikkat (attention) fikri

Bir cümleyi çevirirken her kelime için kaynak cümlenin **farklı kısımlarına** odaklanırız. Dikkat mekanizması, her pozisyonun dizideki **tüm diğer pozisyonlara doğrudan bakmasını** ve hangilerinin önemli olduğuna karar vermesini sağlar.

> "Hayvan sokağı geçmedi çünkü **o** çok yorgundu."
> "o" kelimesi en çok "hayvan" kelimesine dikkat etmelidir.

### 9.2 Ölçekli nokta çarpım dikkati (scaled dot-product attention)

Her kelime (token) üç vektöre dönüştürülür (öğrenilen matrislerle):
- **Sorgu (Query, Q):** "Ben neyi arıyorum?"
- **Anahtar (Key, K):** "Ben neyi içeriyorum?"
- **Değer (Value, V):** "Seçilirsem hangi bilgiyi aktarırım?"

```
                        ⎛ Q·Kᵀ ⎞
Dikkat(Q, K, V) = softmax⎜──────⎟ · V
                        ⎝ √d_k ⎠
```

1. **Q·Kᵀ:** Her sorgunun her anahtarla benzerliği (iç çarpım) → skor matrisi
2. **÷ √d_k:** Boyut büyüdükçe iç çarpımlar büyür ve softmax doygunlaşır; ölçekleme bunu önler
3. **softmax:** Skorları ağırlıklara (toplamı 1) dönüştür
4. **· V:** Değerlerin ağırlıklı ortalaması → her token için bağlamla zenginleşmiş yeni temsil

### 9.3 Elle küçük bir örnek

3 token, d_k = 2. Basitlik için Q = K = V = X:

```
x₁ = (1, 0)     x₂ = (0, 1)     x₃ = (1, 1)
```

**Token 1'in çıktısını hesaplayalım:**

**Adım 1 — Skorlar:** q₁ · kⱼ

```
q₁·k₁ = 1·1 + 0·0 = 1
q₁·k₂ = 1·0 + 0·1 = 0
q₁·k₃ = 1·1 + 0·1 = 1
```

**Adım 2 — Ölçekleme (√2 = 1.414):** (0.707, 0, 0.707)

**Adım 3 — Softmax:**

```
e^0.707 = 2.028,  e^0 = 1.000,  e^0.707 = 2.028   →  toplam = 5.056
ağırlıklar = (0.401, 0.198, 0.401)
```

Token 1, kendisine ve benzer olan token 3'e daha çok "dikkat ediyor".

**Adım 4 — Değerlerin ağırlıklı toplamı:**

```
çıktı₁ = 0.401·(1, 0) + 0.198·(0, 1) + 0.401·(1, 1) = (0.802, 0.599)
```

```python
def dikkat(Q, K, V):
    skor = Q @ K.T / np.sqrt(K.shape[1])
    skor = skor - skor.max(axis=1, keepdims=True)
    agirlik = np.exp(skor) / np.exp(skor).sum(axis=1, keepdims=True)
    return agirlik @ V, agirlik

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
cikti, A = dikkat(X, X, X)
print("Dikkat ağırlıkları:\n", A.round(3))
print("Çıktılar:\n", cikti.round(3))
```

### 9.4 Transformer mimarisi

"Attention Is All You Need" (2017) makalesi RNN'leri tamamen kaldırıp yalnızca dikkat mekanizmasına dayanan bir mimari önerdi:

```
          Girdi tokenları
                │
       Gömme (embedding) + Konum kodlaması
                │
     ┌──────────▼──────────┐
     │ Çok başlı öz-dikkat  │ ◄── her token tüm tokenlara bakar
     │   + Artık bağlantı   │
     │   + Katman normu     │      × N blok
     │ İleri beslemeli ağ   │
     │   + Artık bağlantı   │
     │   + Katman normu     │
     └──────────┬──────────┘
                ▼
             Çıktı
```

- **Öz-dikkat (self-attention):** Q, K, V aynı diziden gelir; her token diğer tüm tokenlarla etkileşir.
- **Çok başlı dikkat (multi-head):** Birden çok dikkat paralel çalışır; her "baş" farklı bir ilişki türüne odaklanabilir (sözdizimi, eş gönderim, ...).
- **Konum kodlaması:** Dikkat işlemi sıradan bağımsızdır; kelimelerin sırasını belirtmek için her tokene konum bilgisi eklenir.

**RNN'e göre avantajları:**
- **Paralel hesaplama:** Tüm tokenlar aynı anda işlenir → GPU'larda çok hızlı eğitim
- **Uzun mesafeli bağımlılıklar:** Herhangi iki token arasındaki yol uzunluğu 1'dir
- **Ölçeklenebilirlik:** Milyarlarca parametreye ve devasa veri setlerine ölçeklenebilir

**Dezavantajı:** Dikkat maliyeti dizi uzunluğunun **karesiyle** (O(n²)) artar.

### 9.5 Transformer'ın etkisi

| Model ailesi | Yapı | Örnek kullanım |
|---|---|---|
| **BERT** | Sadece kodlayıcı (encoder); iki yönlü bağlam | Metin sınıflandırma, soru-cevap |
| **GPT** | Sadece kod çözücü (decoder); bir sonraki tokeni tahmin | Metin üretme, sohbet (büyük dil modelleri) |
| **T5, BART** | Kodlayıcı + kod çözücü | Çeviri, özetleme |
| **ViT** | Görüntü yamalarına uygulanan Transformer | Görüntü sınıflandırma |

Bu modeller de **transfer öğrenme** ile kullanılır: Devasa veride önceden eğitilir, sonra kendi görevimize ince ayar yapılır.

```python
# pip install transformers
from transformers import pipeline
siniflandirici = pipeline("sentiment-analysis")
print(siniflandirici(["I love this course!", "The exam was terrible."]))
```

---

## 10. Laboratuvar: Önceden Eğitilmiş Bir CNN ile Görüntü Sınıflandırma

**Görev:** Az sayıda görüntüyle (her sınıftan ~120 eğitim görüntüsü) karınca ve arı fotoğraflarını sınıflandırmak. Sıfırdan eğitilen bir CNN bu kadar az veriyle zorlanır; transfer öğrenme ise yüksek doğruluk verir.

### 10.1 Hazırlık

```python
import os, zipfile, urllib.request, time, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

torch.manual_seed(0)
cihaz = "cuda" if torch.cuda.is_available() else "cpu"

# Veri: hymenoptera (karınca / arı), PyTorch eğitim materyallerinden
if not os.path.exists("hymenoptera_data"):
    urllib.request.urlretrieve(
        "https://download.pytorch.org/tutorial/hymenoptera_data.zip", "hymenoptera.zip")
    zipfile.ZipFile("hymenoptera.zip").extractall(".")

ORT, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet istatistikleri
donusumler = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(ORT, STD)]),
    "val": transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(ORT, STD)]),
}
veri = {k: datasets.ImageFolder(f"hymenoptera_data/{k}", donusumler[k]) for k in ["train", "val"]}
yukleyici = {k: DataLoader(veri[k], batch_size=16, shuffle=(k == "train")) for k in veri}
siniflar = veri["train"].classes
print(siniflar, {k: len(v) for k, v in veri.items()})     # ['ants', 'bees'] {'train': 244, 'val': 153}
```

### 10.2 Hazır modeli doğrudan kullanmak (ImageNet sınıfları)

```python
agirliklar = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=agirliklar).to(cihaz).eval()
imagenet_siniflari = agirliklar.meta["categories"]

X, y = next(iter(yukleyici["val"]))
with torch.no_grad():
    olasilik = resnet(X.to(cihaz)).softmax(dim=1)
en_iyi5 = olasilik.topk(5, dim=1)

def goster(img_tensor, ax, baslik):
    img = img_tensor.numpy().transpose(1, 2, 0) * STD + ORT     # normalizasyonu geri al
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(baslik, fontsize=8); ax.axis("off")

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    etiketler = [f"{imagenet_siniflari[j]} ({p:.2f})"
                 for p, j in zip(en_iyi5.values[i].cpu(), en_iyi5.indices[i].cpu())][:3]
    goster(X[i], ax[i], f"Gerçek: {siniflar[y[i]]}\n" + "\n".join(etiketler))
plt.show()
```

ImageNet'te "ant" ve "bee" sınıfları zaten var; model önceden eğitilmiş haliyle bile çoğunu tanır.

### 10.3 Eğitim fonksiyonu

```python
def egit(model, optimizer, epoch=10, zamanlayici=None):
    kayip_fn = nn.CrossEntropyLoss()
    en_iyi_agirlik, en_iyi_dog = copy.deepcopy(model.state_dict()), 0.0
    gecmis = {"train": [], "val": []}
    for e in range(epoch):
        for faz in ["train", "val"]:
            model.train() if faz == "train" else model.eval()
            dogru, n = 0, 0
            for X, y in yukleyici[faz]:
                X, y = X.to(cihaz), y.to(cihaz)
                with torch.set_grad_enabled(faz == "train"):
                    logit = model(X)
                    kayip = kayip_fn(logit, y)
                    if faz == "train":
                        optimizer.zero_grad(); kayip.backward(); optimizer.step()
                dogru += (logit.argmax(1) == y).sum().item(); n += len(y)
            gecmis[faz].append(dogru / n)
        if zamanlayici: zamanlayici.step()
        print(f"Epoch {e + 1}: eğitim {gecmis['train'][-1]:.3f}  doğrulama {gecmis['val'][-1]:.3f}")
        if gecmis["val"][-1] > en_iyi_dog:
            en_iyi_dog, en_iyi_agirlik = gecmis["val"][-1], copy.deepcopy(model.state_dict())
    model.load_state_dict(en_iyi_agirlik)
    print(f"En iyi doğrulama doğruluğu: {en_iyi_dog:.3f}")
    return gecmis
```

### 10.4 Deney 1 — Sıfırdan eğitim (karşılaştırma için)

```python
sifirdan = models.resnet18(weights=None)               # rastgele ağırlıklar
sifirdan.fc = nn.Linear(sifirdan.fc.in_features, 2)
sifirdan = sifirdan.to(cihaz)
g_sifir = egit(sifirdan, torch.optim.Adam(sifirdan.parameters(), lr=1e-3), epoch=10)
```

### 10.5 Deney 2 — Öznitelik çıkarma (dondurulmuş gövde)

```python
model_oc = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_oc.parameters():
    p.requires_grad = False                            # tüm katmanları dondur
model_oc.fc = nn.Linear(model_oc.fc.in_features, 2)    # yeni katman (varsayılan: eğitilebilir)
model_oc = model_oc.to(cihaz)

egitilebilir = sum(p.numel() for p in model_oc.parameters() if p.requires_grad)
print("Eğitilebilir parametre:", egitilebilir)          # sadece 1026 (512·2 + 2)
g_oc = egit(model_oc, torch.optim.Adam(model_oc.fc.parameters(), lr=1e-3), epoch=10)
```

### 10.6 Deney 3 — İnce ayar

```python
model_ia = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_ia.fc = nn.Linear(model_ia.fc.in_features, 2)
model_ia = model_ia.to(cihaz)

# Farklı katmanlara farklı öğrenme oranları
optimizer = torch.optim.Adam([
    {"params": [p for n, p in model_ia.named_parameters() if not n.startswith("fc")], "lr": 1e-4},
    {"params": model_ia.fc.parameters(), "lr": 1e-3},
])
zamanlayici = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
g_ia = egit(model_ia, optimizer, epoch=10, zamanlayici=zamanlayici)
```

### 10.7 Karşılaştırma ve hata analizi

```python
for ad, g in [("Sıfırdan", g_sifir), ("Öznitelik çıkarma", g_oc), ("İnce ayar", g_ia)]:
    plt.plot(g["val"], "o-", label=ad)
plt.xlabel("Epoch"); plt.ylabel("Doğrulama doğruluğu"); plt.legend(); plt.show()

# Yanlış sınıflandırılan görüntüler
model_ia.eval(); yanlislar = []
with torch.no_grad():
    for X, y in yukleyici["val"]:
        tahmin = model_ia(X.to(cihaz)).argmax(1).cpu()
        yanlislar += [(x, t, g) for x, t, g in zip(X, tahmin, y) if t != g]
print("Yanlış sayısı:", len(yanlislar))
fig, ax = plt.subplots(1, min(6, len(yanlislar)), figsize=(18, 3), squeeze=False)
for a, (x, t, g) in zip(ax[0], yanlislar):
    goster(x, a, f"Tahmin: {siniflar[t]} / Gerçek: {siniflar[g]}")
plt.show()
```

Beklenen sonuç: Sıfırdan eğitim ~%60–70, öznitelik çıkarma ~%90+, ince ayar ~%93–95 doğrulama doğruluğu.

### 10.8 Ek görevler

1. **Filtreleri görselleştirin:** `model_ia.conv1.weight` (64 × 3 × 7 × 7) ilk katman filtrelerini 8×8'lik bir ızgarada renkli görüntü olarak çizin. Kenar ve renk dedektörlerini görebiliyor musunuz?
2. **Öznitelik haritaları:** Bir görüntüyü `model_ia.layer1` çıkışına kadar geçirip ilk 16 kanalı çizin (`torch.nn.Sequential(*list(model_ia.children())[:5])`).
3. **Kendi veri setiniz:** Telefonunuzla iki/üç sınıftan 30'ar fotoğraf çekin (`veri/train/sinif_adi/` klasör yapısı), aynı ince ayar kodunu çalıştırın.
4. **Farklı model:** `models.mobilenet_v3_small` veya `models.efficientnet_b0` deneyin. Son katmanın adı farklıdır (`classifier`) — modeli `print` ederek bulun.

---

## 11. Özet

- **Evrişim:** Küçük bir filtre görüntü üzerinde kaydırılarak eleman eleman çarpım-toplam yapılır. Yerel bağlantı + ağırlık paylaşımı → az parametre, konumdan bağımsız desen tanıma.
- **Çıktı boyutu** = (W − K + 2P)/S + 1. Dolgu boyutu korur, adım küçültür.
- **Havuzlama** boyutu küçültür, küçük kaymalara dayanıklılık sağlar.
- CNN'ler katman katman **kenar → doku → parça → nesne** hiyerarşisi öğrenir.
- **ResNet** artık bağlantılarıyla çok derin ağları eğitilebilir kıldı.
- **Transfer öğrenme:** Önceden eğitilmiş ağın gövdesini yeniden kullan; az veriyle yüksek başarı. Öznitelik çıkarma (dondur) veya ince ayar (küçük öğrenme oranı).
- **RNN/LSTM** diziyi adım adım gizli durumla işler; uzun bağımlılıklarda ve paralelleştirmede zorlanır.
- **Transformer:** softmax(QKᵀ/√d)·V dikkat mekanizmasıyla her token diğer tüm tokenlara doğrudan bakar; günümüz büyük dil modellerinin temelidir.

## Kendinizi Sınayın

1. 32×32×3 bir girdiye 5×5 boyutunda 16 filtre (dolgu 2, adım 1) uygulanırsa çıktı boyutu ve parametre sayısı nedir?
2. Bölüm 2.2'deki görüntüye yatay kenar filtresi [[1,1,1],[0,0,0],[−1,−1,−1]] uygulansaydı çıktı ne olurdu? Neden?
3. Havuzlama katmanının öğrenilebilir parametresi var mıdır?
4. 100 görüntülük bir tıbbi görüntü veri seti için transfer öğrenmede hangi stratejiyi seçersiniz? Görüntüler ImageNet'e çok benzemiyorsa ne değişir?
5. Bölüm 9.3'teki örnekte token 2'nin dikkat ağırlıklarını ve çıktısını hesaplayın.
6. Transformer'lar neden RNN'lere göre daha hızlı eğitilir?

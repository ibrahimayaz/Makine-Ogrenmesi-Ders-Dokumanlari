# Hafta 6 — Örnek Tabanlı ve Olasılıksal Modeller

## Bu Haftanın Hedefleri

- k-En Yakın Komşu (k-NN) algoritmasının çalışma mantığını adım adım uygulamak
- Farklı mesafe ölçülerini (Öklid, Manhattan, Minkowski, kosinüs) hesaplamak ve doğru olanı seçmek
- Boyutluluk lanetinin mesafe tabanlı yöntemleri neden zorladığını deneysel olarak görmek
- Bayes teoremini gerçek hayat örnekleriyle yorumlamak
- Naive Bayes sınıflandırıcısını elle hesaplamak, sıfırdan kodlamak ve metin sınıflandırmada kullanmak

---

# BÖLÜM A: k-En Yakın Komşu (k-NN)

## 1. Fikir: "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim"

k-NN en sezgisel ML algoritmalarından biridir. Yeni bir örneği sınıflandırmak için:

1. Eğitim verisindeki **tüm örneklere olan mesafesini** hesapla.
2. En yakın **k** komşuyu bul.
3. **Sınıflandırma:** Komşular arasında en çok görülen sınıfı seç (çoğunluk oylaması).
   **Regresyon:** Komşuların hedef değerlerinin ortalamasını al.

**k-NN'in ilginç özelliği:** Eğitim aşamasında hiçbir şey öğrenmez! Sadece veriyi saklar. Tüm hesaplama tahmin anında yapılır. Bu yüzden **tembel öğrenici (lazy learner)** veya **örnek tabanlı (instance-based)** öğrenme denir.

| | Doğrusal/lojistik regresyon | k-NN |
|---|---|---|
| Eğitim | Parametreleri (w, b) öğrenir — yavaş olabilir | Sadece veriyi saklar — anında |
| Tahmin | wᵀx + b — çok hızlı | n mesafe hesabı — yavaş |
| Bellek | Sadece parametreler | Tüm eğitim verisi |
| Karar sınırı | Doğrusal | İstenilen kadar karmaşık |

## 2. Elle Bir Örnek

Bir meyve sınıflandırması: x₁ = ağırlık (100 g), x₂ = tatlılık skoru.

| Meyve | x₁ | x₂ | Sınıf |
|---|---|---|---|
| A | 1 | 1 | 🔴 Elma |
| B | 2 | 1 | 🔴 Elma |
| C | 4 | 3 | 🔵 Armut |
| D | 5 | 4 | 🔵 Armut |
| E | 3 | 2 | 🔴 Elma |

**Yeni meyve:** q = (3.5, 3). Hangi sınıf?

**Adım 1 — Öklid mesafelerini hesapla:** d = √((x₁ − 3.5)² + (x₂ − 3)²)

| Meyve | Hesap | Mesafe | Sınıf |
|---|---|---|---|
| A | √(2.5² + 2²) = √10.25 | 3.20 | 🔴 |
| B | √(1.5² + 2²) = √6.25 | 2.50 | 🔴 |
| C | √(0.5² + 0²) = √0.25 | **0.50** | 🔵 |
| D | √(1.5² + 1²) = √3.25 | 1.80 | 🔵 |
| E | √(0.5² + 1²) = √1.25 | 1.12 | 🔴 |

**Adım 2 — Sırala:** C (0.50, 🔵) → E (1.12, 🔴) → D (1.80, 🔵) → B (2.50, 🔴) → A (3.20, 🔴)

**Adım 3 — Oyla:**

| k | Komşular | Oylar | Tahmin |
|---|---|---|---|
| 1 | C | 🔵1 | **Armut** |
| 3 | C, E, D | 🔵2 – 🔴1 | **Armut** |
| 5 | Hepsi | 🔵2 – 🔴3 | **Elma** |

**Önemli gözlem:** k değeri sonucu değiştirebilir! k = 5'te tüm veri kullanıldı ve sonuç sadece çoğunluk sınıfı oldu.

```
 x₂
  4 │              🔵D
  3 │          🔵C ★q
  2 │      🔴E
  1 │ 🔴A 🔴B
    └─────────────────── x₁
      1   2   3   4   5
```

## 3. k Değerinin Seçimi

```
 k = 1                         k = 15                         k = n
 Çok girintili sınır           Yumuşak sınır                  Tek sınıf
 ╭╮ ╭─╮╭╮                       ╭─────╮                        ─────────
 ╰╯╭╯ ╰╯╰─                     ╱       ╰──
 Aşırı öğrenme                 Dengeli                        Eksik öğrenme
 (yüksek varyans)                                             (yüksek yanlılık)
```

- **k küçük** → Model her gürültülü noktaya tepki verir → aşırı öğrenme. k = 1'de eğitim doğruluğu her zaman %100'dür (her nokta kendinin en yakın komşusu).
- **k büyük** → Model çok "ortalama" düşünür → eksik öğrenme.
- İkili sınıflandırmada **tek sayılı k** seçmek beraberliği önler.
- En iyi k **çapraz doğrulama** ile bulunur. Başlangıç için sık kullanılan bir kural: k ≈ √n.

### Ağırlıklı oylama

Yakın komşuların oyu daha değerli olsun: ağırlık = 1 / mesafe.

Örneğimizde k = 5 için:
- 🔵: 1/0.50 + 1/1.80 = 2.00 + 0.56 = **2.56**
- 🔴: 1/1.12 + 1/2.50 + 1/3.20 = 0.89 + 0.40 + 0.31 = **1.61**
- Tahmin: **Armut** (düz oylamada Elma'ydı)

scikit-learn: `KNeighborsClassifier(weights="distance")`

## 4. Mesafe Ölçüleri

p = (1, 2) ve q = (4, 6) noktaları için:

| Ölçü | Formül | Sonuç | Sezgi |
|---|---|---|---|
| **Öklid (L2)** | √Σ(pᵢ − qᵢ)² | √(9 + 16) = **5** | Kuş uçuşu mesafe |
| **Manhattan (L1)** | Σ\|pᵢ − qᵢ\| | 3 + 4 = **7** | Şehir blokları arasında yürümek |
| **Chebyshev (L∞)** | max\|pᵢ − qᵢ\| | max(3, 4) = **4** | Satrançta şahın hamle sayısı |
| **Minkowski (Lp)** | (Σ\|pᵢ − qᵢ\|ᵖ)^(1/p) | p=1 → Manhattan, p=2 → Öklid | Genelleştirilmiş form |

```
     q(4,6)
       ●         Öklid: düz çizgi (5)
      ╱│         Manhattan: önce sağa 3, sonra yukarı 4 (7)
     ╱ │4
    ╱  │
   ●───┘
 p(1,2) 3
```

**Kosinüs benzerliği:** Vektörlerin **yönüne** bakar, büyüklüğüne değil.

```
                 p · q          (1·4 + 2·6)       16
cos(θ) = ───────────────── = ──────────────── = ─────── = 0.992
           ‖p‖ · ‖q‖          √5 · √52          16.12

Kosinüs mesafesi = 1 − cos(θ) = 0.008
```

Metin verisinde çok kullanılır: Bir belgede "futbol" kelimesi 2 kez, diğerinde 20 kez geçse bile ikisi de futbol hakkındaysa yönleri benzerdir.

**Hamming mesafesi:** Kategorik/ikili verilerde farklı olan pozisyon sayısı. "karar" ve "kurar" → 1.

```python
import numpy as np

p, q = np.array([1, 2]), np.array([4, 6])
print("Öklid    :", np.sqrt(((p - q) ** 2).sum()))
print("Manhattan:", np.abs(p - q).sum())
print("Chebyshev:", np.abs(p - q).max())
print("Minkowski (p=3):", (np.abs(p - q) ** 3).sum() ** (1 / 3))
print("Kosinüs benzerliği:", p @ q / (np.linalg.norm(p) * np.linalg.norm(q)))
```

> **Ölçekleme şarttır!** Hafta 2'de gördüğümüz gibi, büyük ölçekli bir öznitelik (maaş) küçük ölçekli olanları (yaş) mesafe hesabında ezer. k-NN'den önce mutlaka `StandardScaler` uygulayın.

## 5. Sıfırdan k-NN

```python
from collections import Counter

class BenimKNN:
    def __init__(self, k=3, mesafe="oklid", agirlikli=False):
        self.k, self.mesafe, self.agirlikli = k, mesafe, agirlikli

    def fit(self, X, y):
        self.X_train, self.y_train = np.asarray(X), np.asarray(y)   # sadece sakla!
        return self

    def _mesafeler(self, x):
        fark = self.X_train - x                     # broadcasting: tüm örneklere fark
        if self.mesafe == "oklid":
            return np.sqrt((fark ** 2).sum(axis=1))
        if self.mesafe == "manhattan":
            return np.abs(fark).sum(axis=1)
        raise ValueError("Bilinmeyen mesafe")

    def _tahmin_tek(self, x):
        d = self._mesafeler(x)
        komsu_idx = np.argsort(d)[:self.k]          # en yakın k komşu
        if not self.agirlikli:
            return Counter(self.y_train[komsu_idx]).most_common(1)[0][0]
        oylar = {}
        for i in komsu_idx:
            oylar[self.y_train[i]] = oylar.get(self.y_train[i], 0) + 1 / (d[i] + 1e-9)
        return max(oylar, key=oylar.get)

    def predict(self, X):
        return np.array([self._tahmin_tek(x) for x in np.asarray(X)])

# Elle yaptığımız örnek
X = np.array([[1, 1], [2, 1], [4, 3], [5, 4], [3, 2]])
y = np.array(["Elma", "Elma", "Armut", "Armut", "Elma"])
for k in [1, 3, 5]:
    print(k, BenimKNN(k=k).fit(X, y).predict([[3.5, 3]]))
print("Ağırlıklı k=5:", BenimKNN(k=5, agirlikli=True).fit(X, y).predict([[3.5, 3]]))
```

**Karar sınırının k ile değişimini görelim:**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, k in zip(ax, [1, 15, 100]):
    m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    a.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
    a.set_title(f"k = {k}  (eğitim doğruluğu: {m.score(X, y):.2f})")
plt.show()
```

**Hesaplama maliyeti:** Her tahmin için n mesafe hesabı gerekir: O(n·d). Büyük verilerde **KD-Tree** veya **Ball-Tree** gibi veri yapıları aramayı hızlandırır (scikit-learn `algorithm="auto"` ile otomatik seçer).

## 6. Boyutluluk Laneti (Curse of Dimensionality)

Boyut sayısı arttıkça uzay o kadar hızlı büyür ki veri **seyrekleşir** ve mesafeler **anlamını yitirir**.

### 6.1 Sezgi 1: Komşuluk artık "yerel" değil

Veri [0, 1]^d birim küpte düzgün dağılmış olsun. En yakın %10'luk veriyi yakalamak için gereken küçük küpün kenar uzunluğu: **0.1^(1/d)**

| Boyut d | Gereken kenar uzunluğu | Yorum |
|---|---|---|
| 1 | 0.10 | Eksenin %10'u |
| 2 | 0.32 | Her eksenin %32'si |
| 10 | 0.79 | Her eksenin %79'u! |
| 100 | 0.98 | Neredeyse tüm eksen |

100 boyutta "en yakın %10'luk komşuluk", her eksende neredeyse tüm aralığı kapsıyor. Komşular artık "yakın" değil.

### 6.2 Sezgi 2: Tüm noktalar birbirine eşit uzaklıkta

```python
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    X = rng.random((500, d))
    q = rng.random(d)
    dist = np.sqrt(((X - q) ** 2).sum(axis=1))
    print(f"d = {d:4d}   en yakın/en uzak oranı = {dist.min() / dist.max():.3f}")
# d =    2   en yakın/en uzak oranı ≈ 0.0x
# d = 1000   en yakın/en uzak oranı ≈ 0.9   <- en yakın ve en uzak neredeyse aynı!
```

Yüksek boyutta en yakın komşu ile en uzak komşu arasındaki fark yok olur. "En yakın" kavramı anlamsızlaşır ve k-NN'in performansı düşer.

### 6.3 Çözümler

- **Öznitelik seçimi:** Gereksiz öznitelikleri at (Hafta 11)
- **Boyut indirgeme:** PCA vb. (Hafta 11)
- **Daha fazla veri:** Gerekli veri miktarı boyutla üstel olarak artar
- Yüksek boyutta (metin, görüntü) daha uygun modeller seçmek: doğrusal modeller, Naive Bayes, sinir ağları

---

# BÖLÜM B: Bayes Teoremi ve Naive Bayes

## 7. Olasılık Hatırlatması ve Bayes Teoremi

**Koşullu olasılık:** B olduğu bilindiğinde A'nın olasılığı:

```
P(A | B) = P(A ∩ B) / P(B)
```

**Bayes teoremi:**

```
              P(B | A) · P(A)
P(A | B) = ─────────────────────
                  P(B)
```

ML diliyle:

```
                     Olabilirlik · Önsel
Sonsal olasılık = ─────────────────────────
                         Kanıt

P(sınıf | veri) = P(veri | sınıf) · P(sınıf) / P(veri)
```

- **Önsel (prior) P(sınıf):** Veriyi görmeden önceki inancımız. Ör. e-postaların %30'u spam.
- **Olabilirlik (likelihood) P(veri | sınıf):** Bu sınıftan olsaydı bu veriyi görme olasılığı. Ör. spam e-postalarda "bedava" kelimesinin geçme olasılığı.
- **Sonsal (posterior) P(sınıf | veri):** Veriyi gördükten sonraki güncellenmiş inancımız.
- **Kanıt P(veri):** Tüm sınıflar için aynıdır; sınıfları karşılaştırırken ihmal edilebilir.

### 7.1 Klasik örnek: Tıbbi test paradoksu

Bir hastalık nüfusun **%1**'inde görülüyor. Test:
- Hastaların **%99**'unu doğru tespit ediyor (duyarlılık)
- Sağlıklıların **%5**'ine yanlışlıkla "pozitif" diyor

Testiniz **pozitif** çıktı. Gerçekten hasta olma olasılığınız nedir?

Çoğu insan "%99" der. Hesaplayalım:

```
P(H) = 0.01           P(+|H) = 0.99
P(¬H) = 0.99          P(+|¬H) = 0.05

P(+) = P(+|H)·P(H) + P(+|¬H)·P(¬H) = 0.99·0.01 + 0.05·0.99 = 0.0099 + 0.0495 = 0.0594

           0.99 · 0.01
P(H|+) = ────────────── = 0.167   →   sadece %16.7 !
             0.0594
```

**Sezgisel açıklama — 10.000 kişi düşünün:**

```
10.000 kişi
 ├── 100 hasta       → 99 pozitif, 1 negatif
 └── 9.900 sağlıklı  → 495 pozitif (yanlış alarm), 9.405 negatif

Toplam pozitif = 99 + 495 = 594
Bunlardan gerçekten hasta = 99   →   99 / 594 = %16.7
```

Hastalık nadir olduğu için yanlış alarmlar gerçek vakalardan sayıca fazladır. **Önsel olasılık çok önemlidir!** (Hafta 5'teki dengesiz veri ve precision konusuyla aynı fenomen.)

```python
def bayes(prior, duyarlilik, yanlis_pozitif):
    kanit = duyarlilik * prior + yanlis_pozitif * (1 - prior)
    return duyarlilik * prior / kanit

print(bayes(0.01, 0.99, 0.05))     # 0.1667
# İkinci bir test de pozitif çıkarsa? Yeni önsel = 0.1667
print(bayes(0.1667, 0.99, 0.05))   # 0.798
```

Bayes'in gücü: **Yeni kanıt geldikçe inancımızı güncelleyebiliriz.**

## 8. Naive Bayes Sınıflandırıcı

### 8.1 Problem

Bir e-posta d = (w₁, w₂, ..., wₙ) kelimelerinden oluşuyor. Hangi sınıf daha olası?

```
P(spam | w₁, ..., wₙ)  ∝  P(w₁, ..., wₙ | spam) · P(spam)
```

**Sorun:** P(w₁, ..., wₙ | spam) — yani bu kelimelerin **tam olarak bu birleşiminin** spam'de görülme olasılığı — tahmin edilemez. Binlerce kelimelik sözlükte olası kombinasyon sayısı astronomiktir; hiçbir veri seti yeterli olmaz.

### 8.2 "Naif" varsayım

**Sınıf bilindiğinde, öznitelikler birbirinden bağımsızdır** varsayımını yaparız:

```
P(w₁, ..., wₙ | sınıf) ≈ P(w₁ | sınıf) · P(w₂ | sınıf) · ... · P(wₙ | sınıf)
```

Bu varsayım gerçekte doğru değildir ("New" ve "York" bağımsız değildir). Ama sınıflandırma için çoğu zaman şaşırtıcı derecede iyi çalışır, çünkü bizim için önemli olan olasılıkların tam değeri değil, **hangi sınıfın daha yüksek** olduğudur.

**Karar kuralı:**

```
ŷ = argmax  P(sınıf) · Π P(wᵢ | sınıf)
     sınıf
```

### 8.3 Elle bir örnek: Spam filtresi

**Eğitim verisi:**

| # | E-posta | Sınıf |
|---|---|---|
| 1 | bedava para kazan | spam |
| 2 | bedava hediye kazan | spam |
| 3 | para kazan | spam |
| 4 | toplantı yarın | normal |
| 5 | proje toplantı yarın | normal |
| 6 | para yarın | normal |

**Adım 1 — Önsel olasılıklar:** P(spam) = 3/6 = 0.5, P(normal) = 3/6 = 0.5

**Adım 2 — Kelime sayıları:**

| Kelime | spam'de | normal'de |
|---|---|---|
| bedava | 2 | 0 |
| para | 2 | 1 |
| kazan | 3 | 0 |
| hediye | 1 | 0 |
| toplantı | 0 | 2 |
| yarın | 0 | 3 |
| proje | 0 | 1 |
| **Toplam** | **8** | **7** |

Sözlük boyutu V = 7.

**Adım 3 — Sıfır olasılık sorunu ve Laplace düzeltmesi:**

Doğrudan oranlarla P(bedava | normal) = 0/7 = 0 olur. O zaman "bedava" geçen her e-postanın normal olma olasılığı, diğer kelimeler ne olursa olsun **sıfır** olur! Tek bir görülmemiş kelime tüm çarpımı sıfırlar.

**Laplace (add-one) düzeltmesi:** Her kelime sayısına 1 ekle:

```
                 sayı(w, sınıf) + 1
P(w | sınıf) = ──────────────────────────
                toplam(sınıf) + V
```

- P(w | spam) = (sayı + 1) / (8 + 7) = (sayı + 1) / 15
- P(w | normal) = (sayı + 1) / (7 + 7) = (sayı + 1) / 14

**Adım 4 — Yeni e-posta: "bedava para"**

```
spam:   0.5 × P(bedava|spam) × P(para|spam)   = 0.5 × (3/15) × (3/15) = 0.5 × 0.200 × 0.200 = 0.02000
normal: 0.5 × P(bedava|normal) × P(para|normal) = 0.5 × (1/14) × (2/14) = 0.5 × 0.071 × 0.143 = 0.00510
```

Normalize edersek:

```
P(spam | "bedava para") = 0.02000 / (0.02000 + 0.00510) = 0.797  →  %79.7 SPAM
```

### 8.4 Sayısal taşma sorunu: Logaritma

Gerçek bir e-postada yüzlerce kelime olur; 0.01 gibi küçük sayıların yüzlercesini çarpmak bilgisayarda **0'a yuvarlanır** (underflow). Çözüm: **logaritma** kullanmak. Çarpım toplama dönüşür, sıralama değişmez:

```
log P(sınıf | d) ∝ log P(sınıf) + Σ log P(wᵢ | sınıf)
```

### 8.5 Naive Bayes türleri

| Tür | Öznitelik tipi | P(xᵢ \| sınıf) | Kullanım |
|---|---|---|---|
| **Multinomial NB** | Sayımlar (kelime frekansı) | Kelime oranları + Laplace | Metin sınıflandırma |
| **Bernoulli NB** | İkili (kelime var/yok) | Kelimenin geçme olasılığı | Kısa metinler |
| **Gaussian NB** | Sürekli sayılar | Normal dağılım yoğunluğu | Sayısal veriler |

**Gaussian NB:** Her sınıf ve öznitelik için ortalama μ ve varyans σ² hesaplanır:

```
                      1              (x − μ)²
P(x | sınıf) = ─────────────  exp(− ─────────)
                 √(2πσ²)              2σ²
```

Örnek: Erkeklerin boyu μ = 177, σ = 7; kadınların μ = 163, σ = 6. Boyu 170 olan biri için olabilirlikler N(170; 177, 7) ve N(170; 163, 6) ile hesaplanır ve önsellerle çarpılır.

### 8.6 Sıfırdan Multinomial Naive Bayes

```python
import numpy as np
from collections import defaultdict

class BenimMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha                      # Laplace düzeltmesi

    def fit(self, belgeler, etiketler):
        self.siniflar = sorted(set(etiketler))
        self.sozluk = sorted({k for b in belgeler for k in b.split()})
        V = len(self.sozluk)
        self.log_prior, self.log_olabilirlik = {}, {}
        for s in self.siniflar:
            sinif_belgeleri = [b for b, e in zip(belgeler, etiketler) if e == s]
            self.log_prior[s] = np.log(len(sinif_belgeleri) / len(belgeler))
            sayac = defaultdict(int)
            for b in sinif_belgeleri:
                for kelime in b.split():
                    sayac[kelime] += 1
            toplam = sum(sayac.values())
            self.log_olabilirlik[s] = {
                k: np.log((sayac[k] + self.alpha) / (toplam + self.alpha * V))
                for k in self.sozluk
            }
        return self

    def skorlar(self, belge):
        sonuc = {}
        for s in self.siniflar:
            skor = self.log_prior[s]
            for kelime in belge.split():
                if kelime in self.log_olabilirlik[s]:     # sözlükte olmayanı atla
                    skor += self.log_olabilirlik[s][kelime]
            sonuc[s] = skor
        return sonuc

    def predict_proba(self, belge):
        s = self.skorlar(belge)
        m = max(s.values())
        ustel = {k: np.exp(v - m) for k, v in s.items()}  # log'dan olasılığa (kararlı)
        top = sum(ustel.values())
        return {k: v / top for k, v in ustel.items()}

    def predict(self, belgeler):
        return [max(self.skorlar(b), key=self.skorlar(b).get) for b in belgeler]

belgeler = ["bedava para kazan", "bedava hediye kazan", "para kazan",
            "toplantı yarın", "proje toplantı yarın", "para yarın"]
etiketler = ["spam", "spam", "spam", "normal", "normal", "normal"]

nb = BenimMultinomialNB().fit(belgeler, etiketler)
print(nb.predict_proba("bedava para"))           # {'normal': 0.203, 'spam': 0.797}
print(nb.predict(["yarın toplantı var mı", "hemen bedava kazan"]))
```

**scikit-learn ile aynı sonuç:**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = CountVectorizer()
X = vec.fit_transform(belgeler)
print(vec.get_feature_names_out())
print(X.toarray())                    # belge-terim matrisi
sk_nb = MultinomialNB(alpha=1.0).fit(X, etiketler)
print(sk_nb.predict_proba(vec.transform(["bedava para"])))   # [[0.203 0.797]]
```

### 8.7 Naive Bayes'in güçlü ve zayıf yönleri

**Güçlü:**
- Çok hızlı (eğitim tek geçişte sayım yapmaktan ibaret)
- Az veriyle bile iyi çalışır
- Yüksek boyutlu verilerde (on binlerce kelime) başarılı — boyutluluk lanetinden k-NN kadar etkilenmez
- Yorumlanabilir: Hangi kelimeler spam'e işaret ediyor görülebilir

**Zayıf:**
- Bağımsızlık varsayımı nedeniyle olasılık **değerleri** genelde aşırı emindir (0.999...). Sıralama iyi, kalibrasyon kötü.
- Öznitelikler arası etkileşimleri modelleyemez

---

## 9. Laboratuvar: Metin Sınıflandırma (Spam Tespiti)

**Veri:** SMS Spam Collection (5.572 SMS, İngilizce, "ham" / "spam").

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["etiket", "mesaj"])
df["y"] = (df["etiket"] == "spam").astype(int)
print(df["etiket"].value_counts(normalize=True))   # ~%13 spam -> dengesiz!

X_train, X_test, y_train, y_test = train_test_split(
    df["mesaj"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])
```

**Görev 1 — Metni sayılara çevirme (Bag of Words)**

```python
vec = CountVectorizer(lowercase=True, stop_words="english")
Xtr = vec.fit_transform(X_train)      # sözlük SADECE eğitimden öğrenilir
Xte = vec.transform(X_test)
print("Sözlük boyutu:", len(vec.vocabulary_))
print("Matris boyutu:", Xtr.shape, " Sıfır olmayan oranı:", Xtr.nnz / np.prod(Xtr.shape))
```

Belge-terim matrisi çok **seyrektir** (sparse): Elemanların %99'dan fazlası sıfırdır.

**Görev 2 — Naive Bayes**

```python
nb = MultinomialNB().fit(Xtr, y_train)
print(classification_report(y_test, nb.predict(Xte), target_names=["ham", "spam"], digits=3))
ConfusionMatrixDisplay.from_estimator(nb, Xte, y_test, display_labels=["ham", "spam"])
plt.show()
```

**Görev 3 — Model ne öğrendi? En "spam" kelimeler**

```python
kelimeler = vec.get_feature_names_out()
# log P(w|spam) − log P(w|ham): pozitif ve büyükse kelime spam'e işaret ediyor
oran = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
print("Spam göstergeleri:", kelimeler[np.argsort(oran)[-15:]][::-1])
print("Ham göstergeleri :", kelimeler[np.argsort(oran)[:15]])
```

**Görev 4 — k-NN, Naive Bayes ve lojistik regresyon karşılaştırması**

```python
modeller = {
    "Naive Bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
    "k-NN (k=5, Öklid)": make_pipeline(CountVectorizer(), KNeighborsClassifier(5)),
    "k-NN (k=5, kosinüs)": make_pipeline(TfidfVectorizer(),
                                         KNeighborsClassifier(5, metric="cosine")),
    "Lojistik Regresyon": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
}
for ad, m in modeller.items():
    f1 = cross_val_score(m, X_train, y_train, cv=5, scoring="f1")
    print(f"{ad:25s} F1 = {f1.mean():.3f} ± {f1.std():.3f}")
```

Sorular: k-NN'de Öklid yerine kosinüs mesafesi kullanmak neden sonucu değiştirdi? Binlerce boyutlu bu veride boyutluluk lanetinin etkisini görüyor musunuz?

**Görev 5 — Kendi mesajlarınızı deneyin**

```python
ornekler = ["Congratulations! You won a FREE iPhone. Click here to claim now",
            "Are we still meeting for lunch tomorrow?",
            "URGENT: your account has been suspended, call 0800 now"]
p = nb.predict_proba(vec.transform(ornekler))[:, 1]
for m, pr in zip(ornekler, p):
    print(f"{pr:.3f}  {m}")
```

**Görev 6 — Sıfırdan yazdığınız `BenimMultinomialNB` sınıfını** (küçük harfe çevirdikten sonra) bu veride eğitin ve scikit-learn sonucu ile karşılaştırın.

---

## 10. Özet

| | k-NN | Naive Bayes |
|---|---|---|
| Tür | Örnek tabanlı, tembel | Olasılıksal, üretici (generative) |
| Eğitim | Veriyi saklar | Olasılıkları sayar |
| Tahmin | k en yakın komşunun oyu | argmax P(sınıf)·ΠP(xᵢ\|sınıf) |
| Hiperparametre | k, mesafe ölçüsü, ağırlıklandırma | Laplace α |
| Ölçekleme | Şart | Gerekmez |
| Yüksek boyut | Zayıf (boyutluluk laneti) | İyi |
| Hız | Eğitim hızlı, tahmin yavaş | İkisi de çok hızlı |

- **k-NN:** Mesafe hesapla → en yakın k'yı bul → oyla. k küçükse aşırı, büyükse eksik öğrenme.
- **Boyutluluk laneti:** Yüksek boyutta mesafeler birbirine yaklaşır, komşuluk anlamını yitirir.
- **Bayes teoremi:** Sonsal ∝ olabilirlik × önsel. Önsel olasılık sonuçları güçlü biçimde etkiler.
- **Naive Bayes:** Koşullu bağımsızlık varsayımıyla çarpımı basitleştirir; Laplace düzeltmesi sıfır olasılığı, logaritma sayısal taşmayı önler.

## Kendinizi Sınayın

1. Bölüm 2'deki veride q = (2, 2) noktasını k = 3 ile sınıflandırın.
2. Neden k-NN'de eğitim doğruluğu k = 1 için her zaman %100'dür?
3. Bir hastalığın yaygınlığı %10 olsaydı Bölüm 7.1'deki pozitif test sonucunun anlamı ne olurdu? Hesaplayın.
4. Bölüm 8.3'teki veride "toplantı kazan" e-postası için her iki sınıfın skorunu hesaplayın.
5. Laplace düzeltmesi olmadan, eğitimde spam'de hiç görülmemiş bir kelime içeren bir spam e-postaya ne olur?

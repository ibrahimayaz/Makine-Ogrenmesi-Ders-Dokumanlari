# Hafta 8 — Topluluk Yöntemleri (Ensemble Methods)

## Bu Haftanın Hedefleri

- Birden çok zayıf modeli birleştirmenin neden güçlü bir model ürettiğini anlamak
- **Bagging** ve **Random Forest**'ın varyansı nasıl azalttığını görmek
- **AdaBoost**'un ağırlık güncelleme mekanizmasını elle hesaplamak
- **Gradient Boosting**'in "artıklar üzerinde öğrenme" fikrini adım adım uygulamak
- XGBoost ve LightGBM'in temel yeniliklerini tanımak
- Tablo verisinde farklı modelleri adil biçimde karşılaştırmak

---

## 1. Kalabalığın Bilgeliği

1906'da bir panayırda 787 kişi bir öküzün ağırlığını tahmin etmeye çalıştı. Tahminlerin hiçbiri tam doğru değildi, ama **ortalaması** (1197 pound) gerçek değere (1198 pound) neredeyse eşitti. Bireysel hatalar birbirini götürmüştü.

Topluluk yöntemleri bu fikri ML'e uygular: **Birçok modelin tahminlerini birleştirmek, tek bir modelden genellikle daha iyidir.**

### 1.1 Neden işe yarar? Basit bir hesap

Her biri bağımsız olarak **%70** doğrulukla çalışan 5 sınıflandırıcımız olsun. Çoğunluk oylamasıyla karar verirsek, en az 3'ünün doğru olma olasılığı:

```
P(≥3 doğru) = C(5,3)·0.7³·0.3² + C(5,4)·0.7⁴·0.3 + C(5,5)·0.7⁵
            = 10·0.343·0.09 + 5·0.2401·0.3 + 0.16807
            = 0.3087 + 0.3602 + 0.1681 = 0.837
```

Tek model %70 → 5 modelin oylaması **%83.7**!

```python
from scipy.stats import binom
for n in [1, 5, 11, 25, 101]:
    print(f"{n:3d} model: {1 - binom.cdf(n // 2, n, 0.7):.3f}")
#   1 model: 0.700
#   5 model: 0.837
#  11 model: 0.922
#  25 model: 0.983
# 101 model: 1.000
```

**Kritik şart:** Modellerin hataları **bağımsız** (en azından birbirinden farklı) olmalı. 5 kopya model aynı hatayı yapar; oylamanın faydası olmaz. Topluluk yöntemlerinin tüm sanatı, **çeşitli (diverse)** modeller üretmektir.

### 1.2 İki ana strateji

| | **Bagging** | **Boosting** |
|---|---|---|
| Model eğitimi | **Paralel**, birbirinden bağımsız | **Sıralı**, her model öncekinin hatasına odaklanır |
| Temel model | Güçlü ama yüksek varyanslı (derin ağaç) | Zayıf, yüksek yanlılıklı (sığ ağaç, "kütük") |
| Azalttığı | **Varyans** | **Yanlılık** (ve varyans) |
| Örnek | Random Forest | AdaBoost, Gradient Boosting, XGBoost |

```
BAGGING:                              BOOSTING:
   Veri                                  Veri
  ╱ │ │ ╲                                 │
 B₁ B₂ B₃ B₄  (bootstrap örnekleri)       M₁ → hatalar → M₂ → hatalar → M₃ → ...
 │  │  │  │                               │            │             │
 M₁ M₂ M₃ M₄  (paralel)                   └─────── ağırlıklı toplam ───┘
  ╲ │  │ ╱
   Oylama / Ortalama
```

---

## 2. Bagging (Bootstrap Aggregating)

### 2.1 Bootstrap örnekleme

n örnekli bir veri setinden **yerine koyarak** n örnek çekeriz. Bazı örnekler birden çok kez, bazıları hiç seçilmez.

**Örnek:** Orijinal veri [A, B, C, D, E]

```
Bootstrap 1: [A, A, C, D, E]   → B dışarıda
Bootstrap 2: [B, C, C, C, E]   → A, D dışarıda
Bootstrap 3: [A, B, D, D, E]   → C dışarıda
```

**Bir örneğin hiç seçilmeme olasılığı:** Her çekişte seçilmeme olasılığı (1 − 1/n). n çekişte:

```
(1 − 1/n)ⁿ  →  e⁻¹ ≈ 0.368   (n büyüdükçe)
```

Yani her bootstrap örneği orijinal verinin yaklaşık **%63.2**'sini içerir. Kalan **%36.8**'e **OOB (out-of-bag)** örnekleri denir.

```python
import numpy as np
rng = np.random.default_rng(0)
n = 10000
secilen = rng.integers(0, n, n)                     # yerine koyarak n çekiş
print("Benzersiz oranı:", len(np.unique(secilen)) / n)   # ≈ 0.632
```

### 2.2 Bagging algoritması

```
Eğitim:
  B kez tekrarla (b = 1...B):
     1. Eğitim verisinden bootstrap örneği Dᵦ çek
     2. Dᵦ üzerinde bir model Mᵦ eğit (genelde budanmamış derin ağaç)

Tahmin:
  Sınıflandırma: ŷ = çoğunluk oyu(M₁(x), ..., M_B(x))
  Regresyon:     ŷ = (1/B) Σ Mᵦ(x)
```

### 2.3 Neden varyansı azaltır?

Varyansı σ² olan B **bağımsız** tahminin ortalamasının varyansı σ²/B'dir. Ancak bootstrap örnekleri aynı veriden geldiği için modeller **ilişkilidir** (korelasyon ρ). Bu durumda:

```
Var(ortalama) = ρσ² + (1 − ρ)σ²/B
```

- B → ∞ olunca ikinci terim yok olur, ama **ρσ²** kalır.
- Varyansı daha da düşürmenin yolu: **modeller arası korelasyonu (ρ) azaltmak** → Random Forest'ın fikri!

### 2.4 OOB hatası: Bedava doğrulama

Her örnek, ağaçların yaklaşık %37'sinin eğitiminde kullanılmamıştır. O örneği sadece bu ağaçlarla tahmin edip hatayı ölçersek, ayrı bir doğrulama seti olmadan güvenilir bir performans tahmini elde ederiz.

### 2.5 Sıfırdan Bagging

```python
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class BenimBagging:
    def __init__(self, n_model=50, seed=0):
        self.n_model, self.seed = n_model, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.modeller = []
        for _ in range(self.n_model):
            idx = rng.integers(0, n, n)                    # bootstrap
            m = DecisionTreeClassifier().fit(X[idx], y[idx])  # derin ağaç
            self.modeller.append(m)
        return self

    def predict(self, X):
        tum = np.array([m.predict(X) for m in self.modeller])   # (n_model, n_örnek)
        return np.array([Counter(sutun).most_common(1)[0][0] for sutun in tum.T])
```

---

## 3. Random Forest

### 3.1 Fikir: Ağaçları birbirinden farklılaştır

Bagging'de bir sorun var: Verideki **çok güçlü bir öznitelik** varsa, tüm ağaçlar köke onu koyar ve birbirine benzer. Korelasyon (ρ) yüksek kalır.

**Random Forest'ın çözümü:** Her **bölmede**, tüm öznitelikler yerine **rastgele seçilmiş m tanesine** bak.

```
Random Forest = Bagging + Her düğümde rastgele öznitelik alt kümesi
```

Tipik değerler (d = toplam öznitelik sayısı):
- Sınıflandırma: m = √d
- Regresyon: m = d/3 (scikit-learn'de varsayılan artık m = d, yani `max_features=1.0`)

Güçlü öznitelik bazı bölmelerde "aday listesinde" olmadığı için diğer öznitelikler de şans bulur, ağaçlar çeşitlenir, ρ düşer.

### 3.2 Algoritma

```
B ağaç için:
  1. Bootstrap örneği çek
  2. Ağacı büyüt; HER düğümde:
       a. d öznitelikten rastgele m tanesini seç
       b. Sadece bu m öznitelik arasından en iyi bölmeyi bul
  3. Ağacı budamadan bırak (veya hafif sınırla)
Tahmin: Oylama / ortalama
```

### 3.3 Elle küçük bir örnek

4 öznitelik var: [Gelir, Yaş, Borç, Çalışma süresi]. √4 = 2 öznitelik seçilecek.

```
Ağaç 1, kök düğüm:  aday {Gelir, Borç}         → en iyi: Gelir ≤ 45K
Ağaç 2, kök düğüm:  aday {Yaş, Çalışma süresi} → en iyi: Çalışma ≤ 3 yıl
Ağaç 3, kök düğüm:  aday {Borç, Yaş}           → en iyi: Borç ≤ 20K
```

Gelir en güçlü öznitelik olsa bile Ağaç 2 ve 3 farklı yapılar öğrenir. Yeni bir müşteri için tahminler: Ağaç 1 → Onay, Ağaç 2 → Red, Ağaç 3 → Onay ⇒ **Oylama: Onay (2–1)**

### 3.4 Sıfırdan Random Forest

scikit-learn'ün `DecisionTreeClassifier`'ı `max_features` parametresiyle her düğümde rastgele öznitelik seçimini zaten destekliyor. Bunu bagging ile birleştirmek yeterli:

```python
class BenimRandomForest:
    def __init__(self, n_agac=100, max_features="sqrt", max_depth=None, seed=0):
        self.n_agac, self.max_features = n_agac, max_features
        self.max_depth, self.seed = max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.agaclar, self.oob_idx = [], []
        for i in range(self.n_agac):
            idx = rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), idx)          # bu ağacın görmediği örnekler
            agac = DecisionTreeClassifier(max_features=self.max_features,
                                          max_depth=self.max_depth, random_state=i)
            self.agaclar.append(agac.fit(X[idx], y[idx]))
            self.oob_idx.append(oob)
        self.siniflar = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.mean([a.predict_proba(X) for a in self.agaclar], axis=0)

    def predict(self, X):
        return self.siniflar[np.argmax(self.predict_proba(X), axis=1)]

    def oob_skoru(self, X, y):
        oylar = np.zeros((len(X), len(self.siniflar)))
        for agac, oob in zip(self.agaclar, self.oob_idx):
            oylar[oob] += agac.predict_proba(X[oob])
        gecerli = oylar.sum(axis=1) > 0
        tahmin = self.siniflar[np.argmax(oylar[gecerli], axis=1)]
        return (tahmin == y[gecerli]).mean()
```

**Ağaç sayısının etkisi:**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

tek_agac = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
sayilar = [1, 5, 10, 25, 50, 100, 200]
rf_skor = [(BenimRandomForest(n_agac=k).fit(X_tr, y_tr).predict(X_te) == y_te).mean()
           for k in sayilar]
plt.plot(sayilar, rf_skor, "o-", label="Random Forest")
plt.axhline(tek_agac, color="r", ls="--", label="Tek karar ağacı")
plt.xscale("log"); plt.xlabel("Ağaç sayısı"); plt.ylabel("Test doğruluğu"); plt.legend()
plt.show()

rf = BenimRandomForest(n_agac=200).fit(X_tr, y_tr)
print("OOB skoru :", rf.oob_skoru(X_tr, y_tr))
print("Test skoru:", (rf.predict(X_te) == y_te).mean())
```

**Önemli:** Ağaç sayısını artırmak Random Forest'ta **aşırı öğrenmeye yol açmaz**; performans bir noktadan sonra sabitlenir. Sadece hesaplama maliyeti artar.

### 3.5 scikit-learn ile Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True,
                            n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print("OOB:", rf.oob_score_, " Test:", rf.score(X_te, y_te))
```

| Parametre | Etkisi |
|---|---|
| `n_estimators` | Ağaç sayısı; fazlası zarar vermez, sadece yavaşlatır |
| `max_features` | Her bölmede bakılan öznitelik sayısı; küçükse ağaçlar daha çeşitli |
| `max_depth`, `min_samples_leaf` | Tek tek ağaçların karmaşıklığı |
| `n_jobs=-1` | Tüm işlemci çekirdeklerini kullan (ağaçlar paralel eğitilir) |

---

## 4. Boosting: Hatalardan Öğrenmek

Boosting'de modeller **sırayla** eğitilir. Her yeni model, önceki modellerin **zorlandığı örneklere** odaklanır. Sonunda tüm modellerin **ağırlıklı** toplamı alınır.

Benzetme: Sınava hazırlanan bir öğrenci, her deneme sınavından sonra **yanlış yaptığı konulara** daha çok çalışır.

---

## 5. AdaBoost (Adaptive Boosting)

### 5.1 Algoritma

Zayıf öğrenici olarak genellikle **karar kütüğü** (decision stump: derinliği 1 olan ağaç) kullanılır. Etiketler y ∈ {−1, +1}.

```
1. Tüm örneklere eşit ağırlık ver: wᵢ = 1/n

2. t = 1...T için:
   a. Ağırlıklı veride bir zayıf öğrenici hₜ eğit
   b. Ağırlıklı hatayı hesapla:     εₜ = Σ wᵢ · [hₜ(xᵢ) ≠ yᵢ]
   c. Modelin söz hakkını hesapla:  αₜ = ½ · ln((1 − εₜ) / εₜ)
   d. Ağırlıkları güncelle:         wᵢ ← wᵢ · exp(−αₜ · yᵢ · hₜ(xᵢ))
        → doğru bilinen örnek: yᵢ·hₜ(xᵢ) = +1 → ağırlık × e^(−α) (azalır)
        → yanlış bilinen örnek: yᵢ·hₜ(xᵢ) = −1 → ağırlık × e^(+α) (artar)
   e. Ağırlıkları normalize et (toplamı 1 olsun)

3. Son model:  H(x) = işaret( Σ αₜ · hₜ(x) )
```

αₜ'nin davranışı:

| εₜ (hata) | αₜ | Yorum |
|---|---|---|
| 0.01 | 2.30 | Çok iyi model → büyük söz hakkı |
| 0.30 | 0.42 | |
| 0.50 | 0 | Yazı-tura kadar → söz hakkı yok |

### 5.2 Elle bir tur hesaplayalım

10 örnek, tek boyutlu x:

| i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| x | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| y | + | + | + | − | − | − | + | + | + | − |

**Tur 1:** Tüm ağırlıklar wᵢ = 0.1

En iyi kütüklerden biri: "x ≤ 3.5 ise +, değilse −" (aynı hatayı veren "x ≤ 9.5 ise +" da var; birini seçiyoruz). Yanlış bildikleri: 7, 8, 9 (gerçekte + ama − dedi).

```
ε₁ = 0.1 + 0.1 + 0.1 = 0.3
α₁ = ½ · ln(0.7 / 0.3) = ½ · ln(2.333) = 0.424
```

Ağırlık güncelleme:
- Doğru bilinen 7 örnek: 0.1 · e^(−0.424) = 0.1 · 0.655 = 0.0655
- Yanlış bilinen 3 örnek: 0.1 · e^(+0.424) = 0.1 · 1.528 = 0.1528

Toplam = 7 · 0.0655 + 3 · 0.1528 = 0.458 + 0.458 = 0.917. Normalize:
- Doğrular: 0.0655 / 0.917 = **0.0714**
- Yanlışlar (7, 8, 9): 0.1528 / 0.917 = **0.1667**

> İlginç bir özellik: Normalizasyondan sonra yanlış bilinen örneklerin toplam ağırlığı tam **0.5** olur (3 × 0.1667). Yani bir sonraki model için önceki modelin başarısı tam "yazı-tura" seviyesine indirilir; yeni model mecburen farklı bir şey öğrenir.

**Tur 2:** Artık 7, 8, 9 numaralı örnekler 2.3 kat daha önemli. Yeni kütük muhtemelen "x ≤ 9.5 ise +" gibi bir şey seçer: Yanlış bildikleri 4, 5, 6 → ε₂ = 3 × 0.0714 = 0.214 → α₂ = ½·ln(0.786/0.214) = 0.650.

**Son model:** H(x) = işaret(0.424·h₁(x) + 0.650·h₂(x) + α₃·h₃(x) + ...)

Kütüklerin her biri tek bir eşik koyabilir; ama ağırlıklı toplamları karmaşık bir karar sınırı oluşturur.

### 5.3 Sıfırdan AdaBoost

```python
class BenimAdaBoost:
    def __init__(self, T=50):
        self.T = T

    def fit(self, X, y):                  # y ∈ {-1, +1}
        n = len(X)
        w = np.full(n, 1 / n)
        self.kutukler, self.alfalar = [], []
        for _ in range(self.T):
            kutuk = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
            h = kutuk.predict(X)
            eps = np.clip(w[h != y].sum(), 1e-10, 1 - 1e-10)
            alfa = 0.5 * np.log((1 - eps) / eps)
            w = w * np.exp(-alfa * y * h)
            w /= w.sum()
            self.kutukler.append(kutuk); self.alfalar.append(alfa)
        return self

    def karar_fonksiyonu(self, X):
        return sum(a * k.predict(X) for a, k in zip(self.alfalar, self.kutukler))

    def predict(self, X):
        return np.sign(self.karar_fonksiyonu(X))

# Elle yaptığımız örnek
X = np.arange(1, 11).reshape(-1, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
ada = BenimAdaBoost(T=3).fit(X, y)
print("α değerleri:", np.round(ada.alfalar, 3))
print("Tahmin:", ada.predict(X), " Doğruluk:", (ada.predict(X) == y).mean())
```

**Zayıf yönü:** Yanlış etiketlenmiş (gürültülü) örneklere ağırlık sürekli artar → gürültüye ve aykırı değerlere hassastır.

---

## 6. Gradient Boosting

### 6.1 Sezgi: Hataları adım adım düzeltmek

Golf oynadığınızı düşünün. İlk vuruşta top deliğe 50 m uzağa düşer. İkinci vuruşta **kalan mesafeyi** hedeflersiniz, sonra yine kalanı... Her vuruş bir önceki vuruşun **hatasını** düzeltir.

Gradient Boosting (regresyon için):

```
F₀(x) = ȳ                                  (başlangıç: ortalama tahmin)
t = 1...T için:
    rᵢ = yᵢ − Fₜ₋₁(xᵢ)                     (artıklar = şu anki hatalar)
    hₜ = artıkları (r) tahmin eden küçük bir ağaç eğit
    Fₜ(x) = Fₜ₋₁(x) + η · hₜ(x)            (η: öğrenme oranı, ör. 0.1)
Son model: F_T(x) = F₀ + η·h₁(x) + η·h₂(x) + ... + η·h_T(x)
```

### 6.2 Elle bir örnek

Ev fiyatları (milyon TL), η = 0.5, ağaçlar derinlik 1 (kütük):

| Ev | Metrekare | Fiyat (y) |
|---|---|---|
| A | 70 | 2.0 |
| B | 90 | 2.4 |
| C | 120 | 3.6 |
| D | 150 | 4.0 |

**Adım 0:** F₀ = ortalama = (2.0 + 2.4 + 3.6 + 4.0) / 4 = **3.0**

**Adım 1:** Artıklar r = y − 3.0

| Ev | y | F₀ | r₁ |
|---|---|---|---|
| A | 2.0 | 3.0 | −1.0 |
| B | 2.4 | 3.0 | −0.6 |
| C | 3.6 | 3.0 | +0.6 |
| D | 4.0 | 3.0 | +1.0 |

Artıklara kütük uydur: En iyi bölme "Metrekare ≤ 105"
- Sol (A, B): ortalama artık = −0.8
- Sağ (C, D): ortalama artık = +0.8

F₁ = F₀ + 0.5 · h₁:

| Ev | F₁ | Yeni artık r₂ = y − F₁ |
|---|---|---|
| A | 3.0 + 0.5·(−0.8) = 2.6 | −0.6 |
| B | 2.6 | −0.2 |
| C | 3.0 + 0.5·(0.8) = 3.4 | +0.2 |
| D | 3.4 | +0.6 |

MSE: 0.68 → 0.20

**Adım 2:** Artıklara (−0.6, −0.2, 0.2, 0.6) yeni kütük: Yine "≤ 105" en iyisi → sol −0.4, sağ +0.4

F₂: A, B = 2.6 − 0.2 = 2.4; C, D = 3.4 + 0.2 = 3.6 → artıklar: −0.4, 0.0, 0.0, 0.4 → MSE: **0.08**

Bir sonraki ağaç artık A ve D'yi B ve C'den ayırmayı (ör. "≤ 80" veya "≤ 135") öğrenecek. Her adım kalan hatayı küçültür.

### 6.3 Neden "gradyan"?

MSE kaybı L = ½(y − F)² için kaybın F'ye göre negatif gradyanı:

```
−∂L/∂F = y − F = artık
```

Yani artıklara ağaç uydurmak = **fonksiyon uzayında gradyan iniş** yapmak! Hafta 3'te parametreleri (w) gradyanın tersi yönünde güncelliyorduk; burada **tahmin fonksiyonunun kendisini** güncelliyoruz.

Bu bakış açısı genellenebilir: Başka bir kayıp fonksiyonu (MAE, log-loss, ...) için **negatif gradyanı** (sözde-artıklar, pseudo-residuals) hesaplayıp ağacı onlara uydururuz. Sınıflandırmada log-loss kullanılır; sözde-artıklar `y − p` olur (Hafta 5'teki lojistik regresyon gradyanını hatırlayın).

### 6.4 Sıfırdan Gradient Boosting (regresyon)

```python
from sklearn.tree import DecisionTreeRegressor

class BenimGradientBoosting:
    def __init__(self, n_agac=100, eta=0.1, max_depth=3):
        self.n_agac, self.eta, self.max_depth = n_agac, eta, max_depth

    def fit(self, X, y):
        self.F0 = y.mean()
        F = np.full(len(y), self.F0)
        self.agaclar, self.kayiplar = [], []
        for _ in range(self.n_agac):
            artik = y - F                                   # negatif gradyan (MSE)
            agac = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, artik)
            F += self.eta * agac.predict(X)
            self.agaclar.append(agac)
            self.kayiplar.append(((y - F) ** 2).mean())
        return self

    def predict(self, X):
        return self.F0 + self.eta * sum(a.predict(X) for a in self.agaclar)

# Elle yaptığımız örnek
X = np.array([[70], [90], [120], [150]])
y = np.array([2.0, 2.4, 3.6, 4.0])
gb = BenimGradientBoosting(n_agac=2, eta=0.5, max_depth=1).fit(X, y)
print(gb.predict(X), gb.kayiplar)    # [2.4 2.4 3.6 3.6] [0.2, 0.08]
```

**Adımları görselleştirelim:**

```python
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 80))
y = np.sin(x) + rng.normal(0, 0.2, 80)
X = x[:, None]
xc = np.linspace(0, 6, 300)[:, None]

fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for a, T in zip(ax, [1, 5, 20, 100]):
    m = BenimGradientBoosting(n_agac=T, eta=0.3, max_depth=2).fit(X, y)
    a.scatter(x, y, s=10, color="gray"); a.plot(xc, m.predict(xc), "r", lw=2)
    a.set_title(f"{T} ağaç")
plt.show()
```

### 6.5 Önemli hiperparametreler

| Parametre | Etkisi |
|---|---|
| `n_estimators` (T) | Ağaç sayısı. **Random Forest'ın aksine fazlası aşırı öğrenmeye yol açar!** |
| `learning_rate` (η) | Her ağacın katkısı. Küçük η + çok ağaç = genelde daha iyi genelleme ("shrinkage") |
| `max_depth` | Ağaç derinliği; tipik olarak 3–8. Etkileşim derecesini belirler |
| `subsample` | Her ağaç için verinin bir kısmını kullan (< 1.0) → **Stokastik Gradient Boosting**, varyansı düşürür |

η ve T birbirine bağlıdır: η'yı yarıya indirirseniz yaklaşık iki kat ağaç gerekir.

**Erken durdurma:** Doğrulama hatası belirli sayıda turda iyileşmezse eğitimi durdur.

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05,
                                validation_fraction=0.2, n_iter_no_change=20,
                                random_state=0).fit(X_tr, y_tr)
print("Kullanılan ağaç sayısı:", gb.n_estimators_)
```

---

## 7. XGBoost ve LightGBM'e Genel Bakış

Gradient Boosting'in endüstride ve Kaggle yarışmalarında en çok kullanılan hızlı, optimize edilmiş uygulamaları.

### 7.1 XGBoost (eXtreme Gradient Boosting)

- **Düzenlileştirilmiş amaç fonksiyonu:** Kayba ağaç karmaşıklığı için ceza ekler: `Ω = γ·(yaprak sayısı) + ½λ·Σ(yaprak değeri)²`
- **İkinci derece türev (Newton adımı):** Sadece gradyanı değil, ikinci türevi (Hessian) de kullanarak daha isabetli adımlar atar. Yaprak değeri: `w* = −Σg / (Σh + λ)`
- **Eksik değerleri otomatik ele alma:** Her bölmede eksik değerlerin hangi dala gideceğini öğrenir
- **Paralel bölme arama, önbellek optimizasyonu**

### 7.2 LightGBM (Microsoft)

- **Histogram tabanlı bölme:** Sürekli değerleri ~255 kutuya (bin) ayırır; aday eşik sayısı büyük ölçüde azalır → çok hızlı
- **Yaprak bazlı büyüme (leaf-wise):** Seviye seviye büyümek yerine en çok kazanç sağlayan **yaprağı** böler → daha derin, asimetrik ağaçlar, daha düşük kayıp (ama aşırı öğrenme riski; `num_leaves` ile kontrol edilir)
- **GOSS ve EFB:** Küçük gradyanlı örnekleri örnekleme ve seyrek öznitelikleri birleştirme ile hızlanma
- Kategorik öznitelikleri doğrudan destekler

```
Seviye bazlı (XGBoost varsayılanı):     Yaprak bazlı (LightGBM):
         ●                                       ●
       ╱   ╲                                   ╱   ╲
      ●     ●                                 ●     ○
     ╱ ╲   ╱ ╲                               ╱ ╲
    ○  ○  ○   ○                             ●   ○
                                           ╱ ╲
                                          ○   ○
```

**CatBoost** (Yandex): Kategorik özniteliklerle çalışmada özellikle güçlüdür (sızıntısız hedef kodlama).

scikit-learn'ün kendi hızlı uygulaması: `HistGradientBoostingClassifier` (LightGBM'den esinlenmiştir, eksik değerleri destekler).

```python
# pip install xgboost lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, random_state=0)
lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      random_state=0, verbose=-1)
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
```

### 7.3 Ne zaman hangisi?

| Durum | Öneri |
|---|---|
| Hızlı, sağlam bir baseline; az ayar | **Random Forest** |
| Tablo verisinde en yüksek doğruluk | **Gradient Boosting** (XGBoost / LightGBM / CatBoost) |
| Çok büyük veri (milyonlarca satır) | **LightGBM** / `HistGradientBoosting` |
| Çok sayıda kategorik öznitelik | **CatBoost** veya LightGBM |
| Gürültülü etiketler | Random Forest (boosting gürültüye daha hassas) |

> Tablo (yapılandırılmış) veride gradient boosting yöntemleri, derin öğrenme dahil çoğu yöntemi genellikle geride bırakır.

---

## 8. Diğer Topluluk Yöntemleri (Kısa)

**Oylama (Voting):** Farklı türde modelleri birleştir.
- *Hard voting:* Çoğunluk sınıfı
- *Soft voting:* Olasılıkların ortalaması (genelde daha iyi)

**Yığınlama (Stacking):** Temel modellerin tahminlerini **girdi** olarak kullanan bir **meta-model** eğit. Meta-model, hangi temel modele ne zaman güveneceğini öğrenir. Sızıntıyı önlemek için temel modellerin tahminleri çapraz doğrulama ile (out-of-fold) üretilir.

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

temel = [("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
         ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
         ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))]
oylama = VotingClassifier(temel, voting="soft")
yigin = StackingClassifier(temel, final_estimator=LogisticRegression(), cv=5)
```

---

## 9. Laboratuvar: Tablo Verisinde Model Karşılaştırması

**Veri:** Adult Census Income — bir kişinin yıllık gelirinin 50K$'ı aşıp aşmadığını tahmin etmek (≈ 48.000 satır, sayısal + kategorik öznitelikler, dengesiz).

```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)

adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = (adult.target == ">50K").astype(int)
print(X.shape, "  Pozitif oranı:", y.mean().round(3))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

sayisal = make_column_selector(dtype_include="number")
kategorik = make_column_selector(dtype_exclude="number")

def on_isleme(olcekle=True):
    say = [SimpleImputer(strategy="median")] + ([StandardScaler()] if olcekle else [])
    return ColumnTransformer([
        ("say", make_pipeline(*say), sayisal),
        ("kat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
         kategorik),
    ])
```

**Görev 1 — Modelleri tanımla**

```python
modeller = {
    "Lojistik Regresyon": make_pipeline(on_isleme(), LogisticRegression(max_iter=2000)),
    "Karar Ağacı": make_pipeline(on_isleme(False), DecisionTreeClassifier(max_depth=8, random_state=0)),
    "Random Forest": make_pipeline(on_isleme(False),
                                   RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                          n_jobs=-1, random_state=0)),
    "AdaBoost": make_pipeline(on_isleme(False), AdaBoostClassifier(n_estimators=200, random_state=0)),
    "Gradient Boosting": make_pipeline(on_isleme(False),
                                       GradientBoostingClassifier(n_estimators=200, random_state=0)),
    "HistGradientBoosting": make_pipeline(on_isleme(False),
                                          HistGradientBoostingClassifier(random_state=0)),
}
```

**Görev 2 — Adil karşılaştırma (aynı CV katları, birden çok metrik, süre)**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
sonuclar = []
for ad, m in modeller.items():
    t = time.time()
    s = cross_validate(m, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
    sonuclar.append({"Model": ad,
                     "Accuracy": s["test_accuracy"].mean(),
                     "F1": s["test_f1"].mean(),
                     "ROC-AUC": s["test_roc_auc"].mean(),
                     "AUC std": s["test_roc_auc"].std(),
                     "Süre (s)": time.time() - t})
tablo = pd.DataFrame(sonuclar).sort_values("ROC-AUC", ascending=False)
print(tablo.round(4).to_string(index=False))
```

**Görev 3 — XGBoost ve LightGBM'i ekleyin** (kuruluysa) ve tabloyu güncelleyin.

**Görev 4 — Öznitelik önemleri:** En iyi ağaç tabanlı modelin öznitelik önemlerini çubuk grafikle gösterin.

```python
rf = modeller["Random Forest"].fit(X_tr, y_tr)
isimler = rf[0].get_feature_names_out()
onem = pd.Series(rf[-1].feature_importances_, index=isimler).nlargest(15)
onem.sort_values().plot(kind="barh"); plt.title("Random Forest öznitelik önemi"); plt.show()
```

**Görev 5 — Gradient Boosting'de aşırı öğrenme:** `learning_rate` = 0.5, 0.1, 0.02 için ağaç sayısına göre eğitim ve doğrulama log-loss eğrilerini çizin (`staged_predict_proba` kullanın). Hangi η ile en düşük doğrulama kaybına ulaşılıyor ve kaç ağaç gerekiyor?

**Görev 6 — Son değerlendirme:** CV'de en iyi çıkan modeli tüm eğitim setinde eğitip test setinde **bir kez** değerlendirin.

**Rapor soruları:**
1. Tek karar ağacı ile Random Forest arasındaki fark yanlılık-varyans açısından nasıl açıklanır?
2. Hangi model doğruluk/süre dengesinde en iyiydi?
3. Accuracy sıralaması ile ROC-AUC sıralaması aynı mı? Neden farklı olabilir?

---

## 10. Özet

| Yöntem | Temel fikir | Azalttığı | Paralel mi? | Ağaç sayısı artarsa |
|---|---|---|---|---|
| **Bagging** | Bootstrap örnekleri + oylama | Varyans | Evet | Aşırı öğrenme yok |
| **Random Forest** | Bagging + her düğümde rastgele öznitelikler | Varyans (korelasyonu da düşürür) | Evet | Aşırı öğrenme yok |
| **AdaBoost** | Yanlış bilinen örneklerin ağırlığını artır | Yanlılık | Hayır | Aşırı öğrenebilir |
| **Gradient Boosting** | Her ağaç artıklara (negatif gradyana) uydurulur | Yanlılık | Hayır | Aşırı öğrenir → erken durdurma |

- Topluluklar, **çeşitli** modellerin hatalarının birbirini götürmesinden güç alır.
- Bootstrap örnekleri verinin ~%63.2'sini içerir; kalan OOB örnekleri ücretsiz doğrulama sağlar.
- AdaBoost: εₜ → αₜ = ½ln((1−ε)/ε) → yanlışların ağırlığını e^α ile artır.
- Gradient Boosting = fonksiyon uzayında gradyan iniş; küçük öğrenme oranı + çok ağaç + erken durdurma.
- XGBoost/LightGBM: düzenlileştirme, ikinci türev, histogram bölme ile hızlı ve güçlü uygulamalar.

## Kendinizi Sınayın

1. Neden Random Forest'ta derin (budanmamış) ağaçlar, boosting'de ise sığ ağaçlar kullanılır?
2. Bir AdaBoost turunda ε = 0.2 ise α kaçtır? Yanlış bilinen bir örneğin ağırlığı (normalizasyondan önce) kaç katına çıkar?
3. Bölüm 6.2'deki örnekte üçüncü ağaç hangi bölmeyi seçer? F₃'ü hesaplayın.
4. Random Forest'ta `max_features=None` (tüm öznitelikler) seçilirse model neye dönüşür?
5. Gradient Boosting'de öğrenme oranını 0.1'den 0.01'e düşürürseniz ağaç sayısını nasıl ayarlamalısınız?

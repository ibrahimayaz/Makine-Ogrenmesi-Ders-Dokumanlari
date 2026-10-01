# Hafta 4 — Genelleme ve Model Değerlendirme

## Bu Haftanın Hedefleri

- Aşırı öğrenme (overfitting) ve eksik öğrenme (underfitting) kavramlarını tanımak ve teşhis etmek
- Yanlılık-varyans ödünleşimini (bias-variance tradeoff) sezgisel ve deneysel olarak anlamak
- k-katlı çapraz doğrulamanın nasıl çalıştığını görmek ve sıfırdan kodlamak
- Ridge (L2) ve Lasso (L1) düzenlileştirmenin modeli nasıl değiştirdiğini anlamak
- Öğrenme eğrileri ve doğrulama eğrileriyle model teşhisi yapmak

---

## 1. Genelleme: Asıl Amaç

Makine öğrenmesinin amacı eğitim verisini ezberlemek değil, **daha önce görülmemiş verilerde** iyi tahmin yapmaktır. Buna **genelleme** denir.

İki tür hata tanımlarız:

- **Eğitim hatası:** Modelin eğitildiği verideki hatası
- **Test (genelleme) hatası:** Yeni verideki hatası — asıl önemsediğimiz

**Benzetme:** Sınava hazırlanan üç öğrenci:

| Öğrenci | Davranış | Deneme sınavı | Gerçek sınav | ML karşılığı |
|---|---|---|---|---|
| Ali | Konuyu yüzeysel çalıştı | Düşük | Düşük | **Eksik öğrenme** |
| Ayşe | Konuyu anlayarak çalıştı | Yüksek | Yüksek | **İyi genelleme** |
| Can | Deneme sorularını cevaplarıyla ezberledi | %100 | Düşük | **Aşırı öğrenme** |

---

## 2. Eksik Öğrenme ve Aşırı Öğrenme

```
  Eksik öğrenme            İyi uyum               Aşırı öğrenme
  (derece 1)               (derece 3)             (derece 15)
  y                        y                      y
  │  ●    ●   ●            │  ●    ●   ●          │  ●╮  ╭●╮ ╭●
  │ ●───────────●          │ ●╭──╮ ╭──●           │ ●╯╰──╯ ╰─╯
  │●  ●     ●              │●╯ ●╰─╯               │●
  └────────── x            └────────── x          └────────── x
  Eğitim hatası: YÜKSEK    Eğitim hatası: düşük   Eğitim hatası: ~0
  Test hatası:   YÜKSEK    Test hatası:   düşük   Test hatası:   YÜKSEK
```

| | Eksik öğrenme (underfitting) | Aşırı öğrenme (overfitting) |
|---|---|---|
| **Belirti** | Eğitim ve test hatası ikisi de yüksek | Eğitim hatası düşük, test hatası yüksek |
| **Neden** | Model çok basit | Model çok karmaşık / veri az |
| **Çözümler** | Daha karmaşık model, daha fazla öznitelik, düzenlileştirmeyi azalt | Daha fazla veri, daha basit model, düzenlileştirme, öznitelik seçimi, erken durdurma |

### 2.1 Deneyle görelim

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def gercek_fonk(x):
    return np.sin(1.5 * x)

rng = np.random.default_rng(0)
x_train = np.sort(rng.uniform(0, 4, 20))
y_train = gercek_fonk(x_train) + rng.normal(0, 0.2, 20)
x_test = np.sort(rng.uniform(0, 4, 200))
y_test = gercek_fonk(x_test) + rng.normal(0, 0.2, 200)

dereceler = range(1, 16)
egitim_hata, test_hata = [], []
for d in dereceler:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression())
    m.fit(x_train[:, None], y_train)
    egitim_hata.append(mean_squared_error(y_train, m.predict(x_train[:, None])))
    test_hata.append(mean_squared_error(y_test, m.predict(x_test[:, None])))

plt.plot(dereceler, egitim_hata, "o-", label="Eğitim MSE")
plt.plot(dereceler, test_hata, "s-", label="Test MSE")
plt.yscale("log"); plt.xlabel("Polinom derecesi (model karmaşıklığı)")
plt.ylabel("MSE"); plt.legend(); plt.show()
```

Beklenen grafik:

```
 Hata
  │╲                                    ╱  Test hatası
  │ ╲                               ╱──
  │  ╲___                     ___╱
  │      ╲________________╱
  │ ╲                ↑ en iyi karmaşıklık
  │  ╲____
  │       ╲___________________________   Eğitim hatası
  └──────────────────────────────────── Karmaşıklık
    eksik öğrenme  │  uygun  │  aşırı öğrenme
```

**Kritik gözlem:** Eğitim hatası karmaşıklık arttıkça **hep azalır**. Bu yüzden model seçimini eğitim hatasına bakarak yapamayız.

---

## 3. Yanlılık-Varyans Ödünleşimi (Bias-Variance Tradeoff)

### 3.1 Sezgi: Dart tahtası

Aynı modeli farklı eğitim setleriyle defalarca eğittiğimizi düşünün. Her eğitim bir dart atışı:

```
 Düşük yanlılık          Düşük yanlılık          Yüksek yanlılık         Yüksek yanlılık
 Düşük varyans           Yüksek varyans          Düşük varyans           Yüksek varyans
   ┌───────┐               ┌───────┐               ┌───────┐               ┌───────┐
   │   ◎   │               │ •  ◎ •│               │ ••◎   │               │•  ◎   │
   │  •••  │               │•     •│               │ ••    │               │      •│
   │   •   │               │   •   │               │       │               │ •    •│
   └───────┘               └───────┘               └───────┘               └───────┘
   İDEAL                   AŞIRI ÖĞRENME           EKSİK ÖĞRENME           EN KÖTÜ
```

- **Yanlılık (bias):** Modelin ortalama tahmininin gerçek değerden **sistematik** sapması. Modelin varsayımları çok kısıtlayıcıysa (ör. eğrisel bir ilişkiye düz çizgi uydurmak) yanlılık yüksektir.
- **Varyans (variance):** Eğitim verisi değiştiğinde modelin tahminlerinin **ne kadar değiştiği**. Model çok esnekse, verideki gürültüyü de öğrenir ve her veri setinde çok farklı bir model çıkar.

### 3.2 Matematiksel ayrıştırma

Bir x noktasında, beklenen test hatası (MSE) üç parçaya ayrılır:

```
E[(y − ŷ)²] = Yanlılık²  +  Varyans  +  İndirgenemez hata (σ²)
```

- **İndirgenemez hata:** Verideki doğal gürültü. Hiçbir model bunu azaltamaz.
- Basit modeller: yüksek yanlılık, düşük varyans
- Karmaşık modeller: düşük yanlılık, yüksek varyans
- Amaç: **toplamı** en aza indiren karmaşıklığı bulmak

### 3.3 Simülasyonla yanlılık ve varyansı ölçelim

Aynı gerçek fonksiyondan 100 farklı eğitim seti üretip her biriyle model eğitiyoruz. Sonra sabit test noktalarında tahminlerin ortalamasını (yanlılık) ve dağılımını (varyans) ölçüyoruz.

```python
x_grid = np.linspace(0, 4, 50)
f_gercek = gercek_fonk(x_grid)

def yanlilik_varyans(derece, deneme=100, n=20, gurultu=0.2):
    tahminler = np.zeros((deneme, len(x_grid)))
    for t in range(deneme):
        r = np.random.default_rng(t)
        x = r.uniform(0, 4, n)
        y = gercek_fonk(x) + r.normal(0, gurultu, n)
        m = make_pipeline(PolynomialFeatures(derece), LinearRegression()).fit(x[:, None], y)
        tahminler[t] = m.predict(x_grid[:, None])
    ort_tahmin = tahminler.mean(axis=0)
    yanlilik2 = ((ort_tahmin - f_gercek) ** 2).mean()
    varyans = tahminler.var(axis=0).mean()
    return yanlilik2, varyans, tahminler

for d in [1, 3, 10]:
    b2, v, _ = yanlilik_varyans(d)
    print(f"Derece {d:2d}: Yanlılık² = {b2:.4f}   Varyans = {v:.4f}   Toplam = {b2 + v:.4f}")
```

Beklenen sonuç:
- Derece 1: Yanlılık² yüksek, varyans düşük
- Derece 3: İkisi de düşük → en iyi toplam
- Derece 10: Yanlılık² çok düşük, varyans çok yüksek

Görselleştirme:

```python
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, d in enumerate([1, 3, 10]):
    _, _, T = yanlilik_varyans(d, deneme=30)
    for t in T:
        ax[i].plot(x_grid, t, color="gray", alpha=0.3)
    ax[i].plot(x_grid, f_gercek, "r", lw=2, label="Gerçek f(x)")
    ax[i].plot(x_grid, T.mean(axis=0), "b--", lw=2, label="Ortalama tahmin")
    ax[i].set_title(f"Derece {d}"); ax[i].set_ylim(-2, 2)
ax[0].legend(); plt.show()
```

Gri çizgilerin **dağınıklığı = varyans**, mavi kesikli çizginin kırmızıdan **uzaklığı = yanlılık**.

---

## 4. Çapraz Doğrulama (Cross-Validation)

### 4.1 Tek bir doğrulama setinin sorunu

Veriyi bir kez eğitim/doğrulama olarak ayırdığımızda sonuç **hangi örneklerin hangi tarafa düştüğüne** bağlıdır. Şanslı bir bölmede model iyi, şanssız bir bölmede kötü görünebilir. Ayrıca veri azsa %20'yi doğrulamaya ayırmak, eğitim verisini azaltır.

### 4.2 k-katlı çapraz doğrulama — çalışma mantığı

1. Veriyi rastgele karıştır ve **k eşit parçaya (kat, fold)** böl.
2. Her i = 1...k için:
   - i'nci katı **doğrulama**, kalan k−1 katı **eğitim** olarak kullan.
   - Modeli eğit, doğrulama katında skorunu hesapla.
3. k skorun **ortalamasını** (ve standart sapmasını) raporla.

```
k = 5 için:
            Kat 1    Kat 2    Kat 3    Kat 4    Kat 5
Tur 1:    [ DOĞR ] [ eğit ] [ eğit ] [ eğit ] [ eğit ]  → skor₁
Tur 2:    [ eğit ] [ DOĞR ] [ eğit ] [ eğit ] [ eğit ]  → skor₂
Tur 3:    [ eğit ] [ eğit ] [ DOĞR ] [ eğit ] [ eğit ]  → skor₃
Tur 4:    [ eğit ] [ eğit ] [ eğit ] [ DOĞR ] [ eğit ]  → skor₄
Tur 5:    [ eğit ] [ eğit ] [ eğit ] [ eğit ] [ DOĞR ]  → skor₅

Sonuç = ortalama(skor₁ ... skor₅) ± std
```

Her örnek **tam bir kez** doğrulamada, k−1 kez eğitimde kullanılır.

### 4.3 Sıfırdan k-katlı çapraz doğrulama

```python
from sklearn.base import clone

def k_katli_cv(model, X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    indeksler = rng.permutation(len(X))
    katlar = np.array_split(indeksler, k)
    skorlar = []
    for i in range(k):
        dogr_idx = katlar[i]
        egit_idx = np.concatenate([katlar[j] for j in range(k) if j != i])
        m = clone(model)                         # her turda sıfırdan model
        m.fit(X[egit_idx], y[egit_idx])
        hata = mean_squared_error(y[dogr_idx], m.predict(X[dogr_idx]))
        skorlar.append(hata)
    return np.array(skorlar)

X_all = x_train[:, None]
for d in [1, 3, 5, 10]:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    s = k_katli_cv(model, X_all, y_train, k=5)
    print(f"Derece {d:2d}: CV MSE = {s.mean():.4f} ± {s.std():.4f}")
```

### 4.4 scikit-learn ile

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skor = cross_val_score(model, X_all, y_train, cv=kf, scoring="neg_mean_squared_error")
print("CV MSE:", -skor.mean())
```

> scikit-learn'de skorlar her zaman "büyük daha iyi" mantığıyla çalışır. Bu yüzden MSE negatif olarak döner (`neg_mean_squared_error`).

### 4.5 Çapraz doğrulama türleri

| Tür | Ne zaman? |
|---|---|
| `KFold` | Regresyon, genel kullanım |
| `StratifiedKFold` | Sınıflandırma — her katta sınıf oranları korunur |
| `LeaveOneOut` (k = n) | Çok küçük veri; hesaplama maliyeti yüksek |
| `GroupKFold` | Aynı gruba ait örnekler (aynı hasta) aynı katta kalmalı |
| `TimeSeriesSplit` | Zaman serisi — her zaman geçmişle eğit, gelecekle doğrula |

```
TimeSeriesSplit:
Tur 1: [eğit][doğr]
Tur 2: [eğit  eğit][doğr]
Tur 3: [eğit  eğit  eğit][doğr]
```

### 4.6 Doğru iş akışı

```
Tüm veri
   ├── Test seti (%20) ─────────────────────── en sonda BİR KEZ kullan
   └── Eğitim seti (%80)
          └── k-katlı CV ile model/hiperparametre seçimi
                 └── En iyi ayarla tüm eğitim setinde yeniden eğit
```

---

## 5. Düzenlileştirme (Regularization)

### 5.1 Fikir

Aşırı öğrenen modellerin ortak bir özelliği vardır: **Ağırlıkları çok büyüktür.** Derece 15 polinomun katsayılarına bakarsanız binlerce, milyonlarca büyüklüğünde değerler görürsünüz; model, noktaların arasından geçmek için bu büyük katsayılarla aşırı salınım yapar.

Düzenlileştirme, kayıp fonksiyonuna **büyük ağırlıklar için bir ceza terimi** ekler:

```
Yeni kayıp = MSE  +  λ · (ağırlıkların büyüklüğü)
```

- **λ (lambda, scikit-learn'de `alpha`):** Cezanın gücü
  - λ = 0 → sıradan doğrusal regresyon
  - λ çok büyük → tüm ağırlıklar ~0, model sabit bir değer tahmin eder (eksik öğrenme)
- Model artık iki şeyi dengelemek zorunda: veriye iyi uymak **ve** ağırlıkları küçük tutmak.

> Not: Kesişim terimi (b) genellikle cezalandırılmaz. Ayrıca ceza tüm ağırlıklara eşit uygulandığı için **öznitelikler mutlaka ölçeklenmelidir**.

### 5.2 Ridge regresyon (L2 düzenlileştirme)

```
L_Ridge = MSE + λ Σ wⱼ²
```

**Kapalı form çözüm:**

```
θ = (XᵀX + λI)⁻¹ Xᵀy
```

λI eklemek XᵀX'in köşegenini büyütür. Bu hem ağırlıkları küçültür hem de XᵀX tersinemez (ör. birbirine çok benzeyen öznitelikler) olduğunda bile çözümü mümkün kılar.

**Gradyan iniş güncellemesi:**

```
∂L/∂w = (2/n) Xᵀ(Xw − y) + 2λw
w ← w − α·[(2/n) Xᵀ(Xw − y) + 2λw]  =  w(1 − 2αλ) − α·(2/n)Xᵀ(Xw − y)
```

`w(1 − 2αλ)` terimi her adımda ağırlıkları biraz küçültür. Bu yüzden L2'ye **ağırlık sönümü (weight decay)** da denir.

**Sıfırdan Ridge:**

```python
def ridge_kapali_form(X, y, lam):
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    I = np.eye(d + 1)
    I[0, 0] = 0                       # kesişimi cezalandırma
    theta = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return theta[0], theta[1:]        # b, w
```

### 5.3 Lasso regresyon (L1 düzenlileştirme)

```
L_Lasso = MSE + λ Σ |wⱼ|
```

**Lasso'nun özel yeteneği:** Bazı ağırlıkları **tam olarak sıfıra** indirir. Yani otomatik **öznitelik seçimi** yapar. Ridge ise ağırlıkları küçültür ama nadiren tam sıfır yapar.

**Neden? Geometrik sezgi:**

Düzenlileştirme, "ağırlıklar belli bir bütçe içinde kalsın" kısıtı gibi düşünülebilir:
- L2: w₁² + w₂² ≤ t → **daire**
- L1: |w₁| + |w₂| ≤ t → **elmas (köşeli)**

MSE'nin eş-yükselti eğrileri (elipsler) bu bölgeye ilk nerede dokunursa çözüm odur. Elmasın **köşeleri eksenlerin üzerindedir**; elips genellikle önce bir köşeye dokunur → o köşede bir ağırlık sıfırdır.

```
        w₂                               w₂
        │   ╭──╮ MSE elipsleri            │   ╭──╮
        │  ╭┼──┼╮                         │  ╭┼──┼╮
       ╭┼─╮│ ● ││                         ◆  │ ● ││
      ╱ │ ╰╰──╯╯     ── w₁              ╱ ╲ ╰──╯╯   ── w₁
      ╲ │ ╱ ← dokunma noktası            ╲ ╱ ← köşeye dokunur: w₁ = 0
       ╰┼╯   (w₁, w₂ ≠ 0)                 ◆
     L2: daire                          L1: elmas
```

**Çalışma mantığı: Koordinat inişi ve yumuşak eşikleme**

|w| sıfırda türevlenemediği için Lasso genellikle **koordinat inişi** ile çözülür: Her seferinde bir ağırlık güncellenir, diğerleri sabit tutulur. Her ağırlık için çözüm **yumuşak eşikleme (soft thresholding)** fonksiyonudur:

```
               ┌ ρ − λ    ρ >  λ ise
S(ρ, λ) =      │ 0        |ρ| ≤ λ ise     ← küçük etkiler tam sıfıra çekilir
               └ ρ + λ    ρ < −λ ise
```

Burada ρ, j'nci özniteliğin "diğer özniteliklerin açıklayamadığı kısımla" korelasyonudur. Etkisi λ'dan küçükse ağırlık sıfır olur.

```python
def yumusak_esik(rho, lam):
    if rho > lam:  return rho - lam
    if rho < -lam: return rho + lam
    return 0.0

def lasso_koordinat_inisi(X, y, lam, iterasyon=100):
    """X'in standardize edildiği, y'nin ortalamasının çıkarıldığı varsayılır.
    Kayıp: (1/2n)·||y − Xw||² + lam·||w||₁  (scikit-learn ile aynı biçim)"""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iterasyon):
        for j in range(d):
            artik = y - X @ w + X[:, j] * w[j]       # j hariç tahminin artığı
            rho = X[:, j] @ artik / n
            z = (X[:, j] ** 2).sum() / n
            w[j] = yumusak_esik(rho, lam) / z
    return w
```

### 5.4 Ridge ve Lasso'yu karşılaştıralım

```python
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
ozellik = load_diabetes().feature_names

# Kendi Lasso'muzu scikit-learn ile karşılaştır
w_bizim = lasso_koordinat_inisi(X, y - y.mean(), lam=1.0)
w_sk = Lasso(alpha=1.0).fit(X, y).coef_
print(np.round(w_bizim, 2)); print(np.round(w_sk, 2))   # neredeyse aynı

# Katsayı yolları: λ arttıkça ağırlıklar nasıl değişiyor?
lambdalar = np.logspace(-2, 3, 60)
ridge_w = [Ridge(alpha=l).fit(X, y).coef_ for l in lambdalar]
lasso_w = [Lasso(alpha=l / 10, max_iter=10000).fit(X, y).coef_ for l in lambdalar]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(lambdalar, ridge_w); ax[0].set_xscale("log"); ax[0].set_title("Ridge katsayı yolu")
ax[1].plot(lambdalar / 10, lasso_w); ax[1].set_xscale("log"); ax[1].set_title("Lasso katsayı yolu")
for a in ax: a.set_xlabel("λ"); a.axhline(0, color="k", lw=0.5)
ax[1].legend(ozellik, fontsize=7); plt.show()
```

Grafikte göreceğiniz:
- **Ridge:** Tüm katsayılar λ arttıkça yumuşakça sıfıra **yaklaşır** ama sıfır olmaz.
- **Lasso:** Katsayılar tek tek **tam sıfıra düşer**; önemsiz öznitelikler önce elenir.

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Ceza | λΣw² | λΣ\|w\| |
| Ağırlıklar | Küçülür, sıfır olmaz | Bazıları tam sıfır olur |
| Öznitelik seçimi | Yapmaz | Yapar |
| Korelasyonlu öznitelikler | Ağırlığı aralarında paylaştırır | Birini seçer, diğerlerini sıfırlar |
| Kapalı form | Var | Yok (koordinat inişi) |

**Elastic Net** ikisini birleştirir: `MSE + λ₁Σ|w| + λ₂Σw²`.

### 5.5 λ'yı nasıl seçeriz? Çapraz doğrulama ile!

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X, y)
lasso_cv = LassoCV(cv=5, random_state=0).fit(X, y)
print("En iyi Ridge λ:", ridge_cv.alpha_)
print("En iyi Lasso λ:", lasso_cv.alpha_)
print("Lasso'nun sıfırladığı öznitelikler:",
      [o for o, w in zip(ozellik, lasso_cv.coef_) if w == 0])
```

---

## 6. Öğrenme Eğrileri ve Doğrulama Eğrileri

### 6.1 Öğrenme eğrisi (learning curve)

**x ekseni: eğitim verisi miktarı**, y ekseni: eğitim ve doğrulama skoru. Şu soruyu cevaplar: **"Daha fazla veri toplamak işe yarar mı?"**

```
 Yüksek yanlılık (eksik öğrenme)       Yüksek varyans (aşırı öğrenme)
 Hata                                   Hata
  │╲                                     │╲
  │ ╲____________ doğrulama              │ ╲
  │  ________________ eğitim             │  ╲______ doğrulama
  │ ╱    ↑ iki eğri yakın ama              │          ╲_____
  │╱       ikisi de yüksek                │     ↕ büyük boşluk
  │                                     │  ________ eğitim
  └──────────── veri miktarı            └──────────── veri miktarı
  → Daha fazla veri FAYDA ETMEZ         → Daha fazla veri FAYDA EDER
  → Daha karmaşık model gerekir          → Veya düzenlileştirme / daha basit model
```

```python
from sklearn.model_selection import learning_curve

def ogrenme_egrisi_ciz(model, X, y, baslik, ax):
    boyut, egitim, dogr = learning_curve(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=0)
    ax.plot(boyut, -egitim.mean(axis=1), "o-", label="Eğitim")
    ax.plot(boyut, -dogr.mean(axis=1), "s-", label="Doğrulama")
    ax.set_title(baslik); ax.set_xlabel("Eğitim örneği sayısı"); ax.set_ylabel("MSE")
    ax.legend()
```

### 6.2 Doğrulama eğrisi (validation curve)

**x ekseni: bir hiperparametre** (polinom derecesi, λ, ağaç derinliği...). En iyi değeri görsel olarak bulmaya yarar.

```python
from sklearn.model_selection import validation_curve

model = make_pipeline(PolynomialFeatures(10), StandardScaler(), Ridge())
lambdalar = np.logspace(-6, 2, 30)
egitim, dogr = validation_curve(
    model, x_train[:, None], y_train,
    param_name="ridge__alpha", param_range=lambdalar,
    cv=5, scoring="neg_mean_squared_error")

plt.semilogx(lambdalar, -egitim.mean(axis=1), label="Eğitim")
plt.semilogx(lambdalar, -dogr.mean(axis=1), label="Doğrulama")
plt.xlabel("λ"); plt.ylabel("MSE"); plt.legend(); plt.show()
```

Grafik soldan sağa: küçük λ (aşırı öğrenme: eğitim düşük, doğrulama yüksek) → uygun λ (doğrulama en düşük) → büyük λ (eksik öğrenme: ikisi de yüksek).

---

## 7. Laboratuvar: Öğrenme Eğrileri ve Düzenlileştirme Katsayısının Etkisi

**Veri:** Bölüm 2.1'deki sinüs verisini 20 yerine 40 noktayla üretin.

**Görev 1 — Öğrenme eğrileri:** Üç model için öğrenme eğrilerini yan yana çizin ve her birini teşhis edin:
- Derece 1 polinom (beklenen: yüksek yanlılık)
- Derece 4 polinom
- Derece 15 polinom (beklenen: yüksek varyans)

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, d in zip(axes, [1, 4, 15]):
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    ogrenme_egrisi_ciz(m, x_train[:, None], y_train, f"Derece {d}", ax)
axes[0].set_ylim(0, 1); plt.tight_layout(); plt.show()
```

**Görev 2 — Düzenlileştirmenin aşırı öğrenmeyi düzeltmesi:** Derece 15 polinomu λ = 0, 1e-4, 1e-2, 1, 100 değerleriyle Ridge kullanarak eğitin ve eğrileri aynı grafikte çizin.

```python
x_cizim = np.linspace(0, 4, 300)[:, None]
plt.scatter(x_train, y_train, color="k")
for lam in [1e-8, 1e-4, 1e-2, 1, 100]:
    m = make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=lam))
    m.fit(x_train[:, None], y_train)
    plt.plot(x_cizim, m.predict(x_cizim), label=f"λ={lam}")
plt.ylim(-2, 2); plt.legend(); plt.show()
```

**Görev 3 — Kendi CV fonksiyonunuzla λ seçimi:** `k_katli_cv` fonksiyonunu kullanarak 30 farklı λ için CV hatasını hesaplayın; en iyi λ'yı bulun ve `RidgeCV` sonucu ile karşılaştırın.

**Görev 4 — Lasso ile öznitelik seçimi:** Diabetes veri setinde λ'yı artırdıkça kaç özniteliğin sıfırlandığını tablo halinde gösterin. Hangi öznitelikler en son elenir? Bu ne anlama gelir?

**Rapor soruları:**
1. Derece 15 modelinde eğitim ve doğrulama hatası arasındaki boşluk veri arttıkça nasıl değişti?
2. Hangi λ değerinde model eksik öğrenmeye geçti? Grafikte nasıl anladınız?

---

## 8. Özet

- **Amaç genellemedir.** Eğitim hatası tek başına yanıltıcıdır.
- **Eksik öğrenme** = yüksek yanlılık (model çok basit). **Aşırı öğrenme** = yüksek varyans (model çok esnek).
- Test hatası = Yanlılık² + Varyans + Gürültü. Karmaşıklık arttıkça yanlılık azalır, varyans artar.
- **k-katlı CV** her örneği bir kez doğrulamada kullanarak daha güvenilir bir performans tahmini verir.
- **Ridge (L2)** ağırlıkları küçültür; **Lasso (L1)** bazılarını sıfırlayarak öznitelik seçer. λ çapraz doğrulamayla seçilir.
- **Öğrenme eğrisi** "daha çok veri işe yarar mı?", **doğrulama eğrisi** "hangi hiperparametre en iyi?" sorusunu yanıtlar.

## Kendinizi Sınayın

1. Eğitim doğruluğu %99, test doğruluğu %70 olan bir model için üç çözüm önerin.
2. Öğrenme eğrisinde eğitim ve doğrulama hataları birbirine çok yakın ama ikisi de yüksekse ne yaparsınız?
3. 1000 öznitelikten yalnızca 10'unun gerçekten önemli olduğunu düşünüyorsanız Ridge mi Lasso mu seçersiniz?
4. 10-katlı CV'de model kaç kez eğitilir? Her eğitimde verinin yüzde kaçı kullanılır?
5. Düzenlileştirme kullanmadan önce neden ölçekleme yapmak zorundayız?

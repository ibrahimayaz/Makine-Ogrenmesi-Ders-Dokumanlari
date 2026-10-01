# Hafta 3 — Doğrusal Regresyon

## Bu Haftanın Hedefleri

- Doğrusal regresyon modelinin ne öğrendiğini ve nasıl tahmin yaptığını anlamak
- **En küçük kareler** fikrini ve **kapalı form çözümünü** (normal denklem) türetmek
- **Gradyan iniş** algoritmasının adım adım nasıl çalıştığını görmek ve sıfırdan kodlamak
- Polinom regresyon ile doğrusal olmayan ilişkileri modellemek
- MSE, MAE, RMSE ve R² metriklerini hesaplamak ve yorumlamak

---

## 1. Problem: Bir Doğru Uydurmak

Bir kafede günlük sıcaklık ile satılan soğuk içecek sayısı arasındaki ilişkiyi inceleyelim:

| Sıcaklık (°C) x | Satış y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 4 |
| 5 | 5 |

(Sayıları küçük tutuyoruz ki elle hesaplayabilelim.)

Amacımız şu biçimde bir doğru bulmak:

```
ŷ = w·x + b
```

- **w** (eğim, ağırlık): x bir birim artınca ŷ ne kadar artar?
- **b** (kesişim, bias): x = 0 iken ŷ'nin değeri
- **ŷ** ("y şapka"): modelin tahmini; **y** ise gerçek değer

Birden çok öznitelik varsa model şöyle genelleşir:

```
ŷ = w₁x₁ + w₂x₂ + ... + w_d·x_d + b  =  wᵀx + b
```

Örneğin ev fiyatı = 30·metrekare + 150·oda_sayısı − 20·bina_yaşı + 500.

Peki "en iyi" doğru hangisi? Sonsuz sayıda doğru çizebiliriz. Bir **ölçüt** gerekiyor.

---

## 2. Kayıp Fonksiyonu: Hataların Karesi

Her nokta için **artık (residual)**, yani hata: `eᵢ = yᵢ − ŷᵢ`

```
  y
  │            ●              ● gerçek değer (yᵢ)
  │            ┆ eᵢ (hata)    ┆ dikey mesafe
  │      ──────○──────        ○ tahmin (ŷᵢ), doğrunun üzerinde
  │   ───
  └──────────────────── x
```

Hataları neden doğrudan toplamıyoruz? Çünkü +3 ve −3 birbirini götürür ve toplam 0 olur; kötü bir doğru "mükemmel" görünür. Bu yüzden **karelerini** alırız:

```
            1   n
MSE(w,b) = ─── Σ (yᵢ − ŷᵢ)²       (Ortalama Kare Hata)
            n  i=1
```

Karesini almanın avantajları:
1. Hatalar her zaman pozitif olur.
2. Büyük hatalar daha çok cezalandırılır (2 birimlik hata 4, 10 birimlik hata 100 ceza).
3. Fonksiyon türevlenebilir ve "kase" şeklindedir (konveks) → tek bir minimum noktası vardır.

**En küçük kareler yöntemi:** MSE'yi en küçük yapan w ve b'yi bul.

---

## 3. Kapalı Form Çözüm

### 3.1 Tek değişkenli durumda formül

MSE'nin w ve b'ye göre türevlerini alıp sıfıra eşitlersek şu formüller çıkar:

```
      Σ (xᵢ − x̄)(yᵢ − ȳ)
w = ──────────────────────          b = ȳ − w·x̄
         Σ (xᵢ − x̄)²
```

**Türetme (kısaca):** L = Σ(yᵢ − wxᵢ − b)²

- ∂L/∂b = −2 Σ(yᵢ − wxᵢ − b) = 0  →  Σyᵢ = wΣxᵢ + nb  →  **b = ȳ − w·x̄**
- ∂L/∂w = −2 Σ xᵢ(yᵢ − wxᵢ − b) = 0 → b'yi yerine koyup düzenlersek yukarıdaki w formülü elde edilir.

### 3.2 Elle hesaplama

x̄ = (1+2+3+4+5)/5 = **3**, ȳ = (2+4+5+4+5)/5 = **4**

| xᵢ | yᵢ | xᵢ − x̄ | yᵢ − ȳ | (xᵢ − x̄)(yᵢ − ȳ) | (xᵢ − x̄)² |
|---|---|---|---|---|---|
| 1 | 2 | −2 | −2 | 4 | 4 |
| 2 | 4 | −1 | 0 | 0 | 1 |
| 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 2 | 1 | 2 | 4 |
| | | | **Toplam** | **6** | **10** |

- w = 6 / 10 = **0.6**
- b = 4 − 0.6 × 3 = **2.2**

Model: **ŷ = 0.6x + 2.2**. Yorum: Sıcaklık 1 °C artınca satış ortalama 0.6 adet artıyor.

### 3.3 Çok değişkenli durum: Normal denklem

d öznitelik için veriyi matris olarak yazarız. b'yi de ağırlıkların içine almak için X'in başına 1'lerden oluşan bir sütun ekleriz:

```
     ┌ 1  x₁₁  x₁₂ ┐        ┌ b  ┐
X =  │ 1  x₂₁  x₂₂ │    θ = │ w₁ │      ŷ = Xθ
     └ 1  x₃₁  x₃₂ ┘        └ w₂ ┘
```

MSE'yi minimize eden θ:

```
θ = (XᵀX)⁻¹ Xᵀ y          ← Normal denklem
```

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

X = np.column_stack([np.ones_like(x), x])   # [1, x] sütunları
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(theta)   # [2.2 0.6]  -> b = 2.2, w = 0.6
```

Pratikte `inv` yerine sayısal olarak daha kararlı olan `np.linalg.lstsq` veya `np.linalg.solve` kullanılır:

```python
theta, *_ = np.linalg.lstsq(X, y, rcond=None)
```

**Kapalı formun dezavantajı:** XᵀX matrisi d×d boyutundadır ve tersini almak yaklaşık O(d³) işlem gerektirir. Öznitelik sayısı çok büyükse (ör. 100.000) veya veri belleğe sığmıyorsa pratik değildir. Bu durumda **gradyan iniş** kullanırız.

---

## 4. Gradyan İniş (Gradient Descent)

### 4.1 Sezgi: Sisli dağdan inmek

Sisli bir dağın tepesindesiniz ve vadiye inmek istiyorsunuz. Etrafı göremiyorsunuz ama ayağınızın altındaki eğimi hissedebiliyorsunuz. Strateji: **Her adımda en dik iniş yönüne küçük bir adım atın.** Zemin düzleşince (eğim ≈ 0) vadiye ulaşmışsınızdır.

- Dağın yüksekliği = **kayıp (MSE)**
- Konumunuz = **parametreler (w, b)**
- Eğim = **gradyan** (türev)
- Adım büyüklüğü = **öğrenme oranı (learning rate, α)**

```
 Kayıp
   │ ●  1. adım
   │  ╲
   │   ●  2. adım
   │    ╲
   │     ●
   │      ╲●
   │        ●●●  ← minimum (türev ≈ 0)
   └──────────────── w
```

### 4.2 Güncelleme kuralı

```
w  ←  w − α · ∂L/∂w
b  ←  b − α · ∂L/∂b
```

Neden **eksi**? Gradyan, fonksiyonun **arttığı** yönü gösterir. Azaltmak istediğimiz için ters yöne gideriz.

### 4.3 Gradyanları hesaplayalım

L = (1/n) Σ (ŷᵢ − yᵢ)²,  ŷᵢ = w·xᵢ + b

Zincir kuralıyla:

```
∂L/∂w = (2/n) Σ (ŷᵢ − yᵢ) · xᵢ
∂L/∂b = (2/n) Σ (ŷᵢ − yᵢ)
```

Yorum: Tahminler çok büyükse (ŷ > y), gradyan pozitif olur ve w, b azaltılır. Tahminler çok küçükse tersi olur. Model hatayı düzeltecek yönde kendini ayarlar.

### 4.4 Bir adımı elle hesaplayalım

Başlangıç: w = 0, b = 0, α = 0.01

| xᵢ | yᵢ | ŷᵢ = 0 | ŷᵢ − yᵢ | (ŷᵢ − yᵢ)·xᵢ |
|---|---|---|---|---|
| 1 | 2 | 0 | −2 | −2 |
| 2 | 4 | 0 | −4 | −8 |
| 3 | 5 | 0 | −5 | −15 |
| 4 | 4 | 0 | −4 | −16 |
| 5 | 5 | 0 | −5 | −25 |
| | | **Toplam** | **−20** | **−66** |

- ∂L/∂w = (2/5)(−66) = −26.4
- ∂L/∂b = (2/5)(−20) = −8.0
- w ← 0 − 0.01 × (−26.4) = **0.264**
- b ← 0 − 0.01 × (−8.0) = **0.080**

Başlangıçta MSE = (4+16+25+16+25)/5 = 17.2 idi. Yeni parametrelerle tahminler [0.344, 0.608, 0.872, 1.136, 1.4] olur ve MSE ≈ 10.49'a düşer. Bu adımlar yüzlerce kez tekrarlanınca w → 0.6 ve b → 2.2'ye yakınsar.

### 4.5 Sıfırdan kodlama

```python
import numpy as np
import matplotlib.pyplot as plt

def gradyan_inis(x, y, alpha=0.01, epoch=2000):
    w, b = 0.0, 0.0
    n = len(x)
    gecmis = []
    for i in range(epoch):
        y_hat = w * x + b                  # 1) tahmin
        hata = y_hat - y                   # 2) hata
        mse = (hata ** 2).mean()           # 3) kayıp
        dw = (2 / n) * (hata * x).sum()    # 4) gradyanlar
        db = (2 / n) * hata.sum()
        w -= alpha * dw                    # 5) güncelleme
        b -= alpha * db
        gecmis.append(mse)
    return w, b, gecmis

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

w, b, gecmis = gradyan_inis(x, y, alpha=0.05, epoch=2000)
print(f"w = {w:.4f}, b = {b:.4f}")   # w ≈ 0.6000, b ≈ 2.2000

plt.plot(gecmis)
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.yscale("log")
plt.title("Gradyan inişte kaybın azalması")
plt.show()
```

### 4.6 Vektörel (çok değişkenli) gradyan iniş

Matris notasyonuyla gradyan: `∇L = (2/n) Xᵀ(Xθ − y)`

```python
class GDRegresyon:
    def __init__(self, alpha=0.01, epoch=1000):
        self.alpha, self.epoch = alpha, epoch

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.kayiplar = []
        for _ in range(self.epoch):
            hata = X @ self.w + self.b - y
            self.w -= self.alpha * (2 / n) * (X.T @ hata)
            self.b -= self.alpha * (2 / n) * hata.sum()
            self.kayiplar.append((hata ** 2).mean())
        return self

    def predict(self, X):
        return X @ self.w + self.b
```

### 4.7 Öğrenme oranının etkisi

```python
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for i, a in enumerate([0.001, 0.05, 0.12]):
    _, _, g = gradyan_inis(x, y, alpha=a, epoch=60)
    ax[i].plot(g); ax[i].set_title(f"α = {a}")
    ax[i].set_xlabel("Epoch"); ax[i].set_ylabel("MSE")
plt.tight_layout(); plt.show()
```

| α | Davranış |
|---|---|
| Çok küçük (0.001) | Çok yavaş ilerler, binlerce adım gerekir |
| Uygun (0.05) | Hızlı ve düzgün biçimde minimuma iner |
| Çok büyük (0.12) | Minimumun üzerinden atlar, kayıp **artar** (ıraksar) |

```
 α çok büyük:          α uygun:            α çok küçük:
   ●       ●            ●                   ●
    ╲     ╱              ╲                   ●
     ╲   ╱                ●                   ●
      ╲ ╱  ← sağa sola     ╲                   ●
       ×     zıplar         ●●●                 ● (çok yavaş)
```

### 4.8 Ölçeklemenin gradyan inişe etkisi

Öznitelikler farklı ölçeklerdeyse (metrekare: 50–300, oda sayısı: 1–5) kayıp yüzeyi uzun, dar bir vadiye dönüşür. Gradyan iniş bu vadide zigzag çizerek çok yavaş ilerler. Standardizasyon yüzeyi daha "yuvarlak" hale getirir ve yakınsamayı hızlandırır.

```
 Ölçeklenmemiş:              Ölçeklenmiş:
  ╭──────────────────╮         ╭───────╮
  │ ╲╱╲╱╲╱╲╱╲──► ●   │         │   ↘   │
  ╰──────────────────╯         │    ●  │
   zigzag, yavaş               ╰───────╯  doğrudan, hızlı
```

### 4.9 Gradyan iniş türleri

| Tür | Her adımda kullanılan veri | Özellik |
|---|---|---|
| **Batch GD** | Tüm veri | Kararlı ama büyük veride yavaş |
| **Stokastik GD (SGD)** | Tek bir örnek | Hızlı, gürültülü; yerel minimumlardan kaçabilir |
| **Mini-batch GD** | Küçük gruplar (32, 64, 128...) | İkisinin dengesi; derin öğrenmede standart |

```python
def minibatch_gd(X, y, alpha=0.01, epoch=50, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epoch):
        sira = rng.permutation(n)                # her epoch'ta veriyi karıştır
        for bas in range(0, n, batch):
            idx = sira[bas:bas + batch]
            Xb, yb = X[idx], y[idx]
            hata = Xb @ w + b - yb
            w -= alpha * (2 / len(idx)) * (Xb.T @ hata)
            b -= alpha * (2 / len(idx)) * hata.sum()
    return w, b
```

---

## 5. Değerlendirme Metrikleri

Örneğimizdeki model ŷ = 0.6x + 2.2 için:

| x | y | ŷ | y − ŷ | (y − ŷ)² | \|y − ŷ\| | (y − ȳ)² |
|---|---|---|---|---|---|---|
| 1 | 2 | 2.8 | −0.8 | 0.64 | 0.8 | 4 |
| 2 | 4 | 3.4 | 0.6 | 0.36 | 0.6 | 0 |
| 3 | 5 | 4.0 | 1.0 | 1.00 | 1.0 | 1 |
| 4 | 4 | 4.6 | −0.6 | 0.36 | 0.6 | 0 |
| 5 | 5 | 5.2 | −0.2 | 0.04 | 0.2 | 1 |
| | | | **Toplam** | **2.40** | **3.2** | **6** |

**MSE (Ortalama Kare Hata)** = 2.40 / 5 = **0.48**
- Birimi, hedefin biriminin karesidir (adet²) → yorumlaması zor.

**RMSE (Kök Ortalama Kare Hata)** = √0.48 ≈ **0.69**
- Hedefle aynı birimde: "Tahminler ortalama ~0.69 adet sapıyor."
- Büyük hatalara duyarlı.

**MAE (Ortalama Mutlak Hata)** = 3.2 / 5 = **0.64**
- Hedefle aynı birimde, aykırı değerlere MSE/RMSE'den daha dayanıklı.

**R² (Belirlilik katsayısı)**

```
           Σ(yᵢ − ŷᵢ)²         SS_artık
R² = 1 − ───────────── = 1 − ──────────  = 1 − 2.4/6 = 0.60
           Σ(yᵢ − ȳ)²          SS_toplam
```

Yorum: "Model, satışlardaki değişkenliğin %60'ını açıklıyor."

- R² = 1: Mükemmel tahmin
- R² = 0: Model, her zaman ortalamayı (ȳ) tahmin etmekten daha iyi değil
- R² < 0: Model, ortalamadan bile kötü (test verisinde olabilir!)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = np.array([2, 4, 5, 4, 5])
y_pred = 0.6 * np.array([1, 2, 3, 4, 5]) + 2.2

print("MSE :", mean_squared_error(y_true, y_pred))          # 0.48
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred))) # 0.693
print("MAE :", mean_absolute_error(y_true, y_pred))         # 0.64
print("R²  :", r2_score(y_true, y_pred))                    # 0.60
```

**Hangi metrik ne zaman?**
- Büyük hatalar çok maliyetliyse (ör. köprü yük tahmini) → RMSE
- Tüm hatalar eşit önemdeyse ve aykırı değerler varsa → MAE
- Modelleri farklı veri setlerinde karşılaştırmak için ölçekten bağımsız → R²

---

## 6. Polinom Regresyon

### 6.1 Doğrusal olmayan ilişkiler

Arabanın hızı ile yakıt tüketimi arasındaki ilişki U şeklindedir: Çok yavaş ve çok hızlı giderken tüketim artar. Düz bir doğru bunu yakalayamaz.

**Fikir:** Yeni öznitelikler türetiriz: x, x², x³, ... Model hâlâ **parametrelerine göre doğrusaldır**, bu yüzden aynı yöntemler (kapalı form, gradyan iniş) çalışır:

```
ŷ = w₁x + w₂x² + w₃x³ + b
```

İki öznitelik (x₁, x₂) için 2. derece polinom: 1, x₁, x₂, x₁², x₁x₂, x₂²

### 6.2 Çalışma mantığını görelim

```python
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[1], [2], [3]])
poly = PolynomialFeatures(degree=3, include_bias=False)
print(poly.fit_transform(x))
# [[ 1.  1.  1.]
#  [ 2.  4.  8.]
#  [ 3.  9. 27.]]     -> x, x², x³ sütunları
```

### 6.3 Uygulama: Derecenin etkisi

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(0)
x = np.sort(np.random.uniform(-3, 3, 30))
y = 0.5 * x**2 - x + 2 + np.random.normal(0, 1, 30)   # gerçek ilişki: ikinci derece
X = x.reshape(-1, 1)
x_cizim = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.scatter(x, y, color="black", label="Veri")
for derece in [1, 2, 15]:
    model = make_pipeline(PolynomialFeatures(derece), LinearRegression())
    model.fit(X, y)
    plt.plot(x_cizim, model.predict(x_cizim), label=f"Derece {derece}")
plt.ylim(-3, 15); plt.legend(); plt.show()
```

- **Derece 1:** Eğriyi yakalayamaz → **eksik öğrenme (underfitting)**
- **Derece 2:** Gerçek ilişkiye uygun
- **Derece 15:** Her noktadan geçmeye çalışır, noktalar arasında çılgınca salınır → **aşırı öğrenme (overfitting)**

Bu durum Hafta 4'ün ana konusu olacak.

---

## 7. Laboratuvar: Gradyan İnişi Sıfırdan Kodlama ve scikit-learn ile Karşılaştırma

California konut verisini kullanacağız.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

veri = fetch_california_housing(as_frame=True)
X, y = veri.data.to_numpy(), veri.target.to_numpy()   # hedef: medyan ev değeri (100 bin $)
print(veri.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçekleme (yalnızca eğitimde fit!)
sc = StandardScaler().fit(X_train)
X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
```

**Görev 1 — Kendi gradyan iniş modeliniz** (Bölüm 4.6'daki `GDRegresyon` sınıfı)

```python
gd = GDRegresyon(alpha=0.1, epoch=1000).fit(X_train_s, y_train)
plt.plot(gd.kayiplar); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.show()
```

**Görev 2 — Normal denklem**

```python
Xb_train = np.column_stack([np.ones(len(X_train_s)), X_train_s])
theta = np.linalg.solve(Xb_train.T @ Xb_train, Xb_train.T @ y_train)
```

**Görev 3 — scikit-learn**

```python
lr = LinearRegression().fit(X_train_s, y_train)
sgd = SGDRegressor(max_iter=1000, random_state=42).fit(X_train_s, y_train)
```

**Görev 4 — Karşılaştırma tablosu**

```python
def degerlendir(ad, y_pred):
    return {"model": ad,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)}

Xb_test = np.column_stack([np.ones(len(X_test_s)), X_test_s])
sonuc = pd.DataFrame([
    degerlendir("Bizim GD", gd.predict(X_test_s)),
    degerlendir("Normal denklem", Xb_test @ theta),
    degerlendir("sklearn LinearRegression", lr.predict(X_test_s)),
    degerlendir("sklearn SGDRegressor", sgd.predict(X_test_s)),
])
print(sonuc.round(4))

# Katsayıları karşılaştırın
print(pd.DataFrame({"GD": gd.w, "Normal": theta[1:], "sklearn": lr.coef_},
                   index=veri.feature_names).round(4))
```

Dört yöntemin de çok yakın katsayılar ve R² ≈ 0.58 vermesi beklenir.

**Görev 5 — Deneyler**
1. Ölçekleme yapmadan kendi GD'nizi çalıştırın. Ne oluyor? (İpucu: α'yı çok küçültmeniz gerekecek veya kayıp `nan` olacak.)
2. α = 0.001, 0.01, 0.1, 0.5 için kayıp eğrilerini tek grafikte çizin.
3. Katsayılara bakarak ev fiyatını en çok etkileyen öznitelik hangisi? (Ölçeklenmiş verideki katsayılar karşılaştırılabilir.)
4. Gerçek vs. tahmin grafiği çizin (`plt.scatter(y_test, y_pred)`) ve köşegen çizgiyi ekleyin. Hangi bölgelerde model zorlanıyor?

---

## 8. Doğrusal Regresyonun Varsayımları (Kısa Not)

Klasik istatistikte doğrusal regresyonun güvenilir yorumlanması için:

1. **Doğrusallık:** X ile y arasındaki ilişki doğrusal
2. **Bağımsızlık:** Hatalar birbirinden bağımsız
3. **Eşit varyans (homoskedastisite):** Hataların yayılımı her x değerinde benzer
4. **Normallik:** Hatalar yaklaşık normal dağılımlı
5. **Çoklu doğrusallık yok:** Öznitelikler birbirleriyle çok yüksek korelasyonlu değil

Artık (residual) grafiği bu varsayımları kontrol etmenin pratik yoludur:

```python
y_pred = lr.predict(X_test_s)
plt.scatter(y_pred, y_test - y_pred, alpha=0.3)
plt.axhline(0, color="red"); plt.xlabel("Tahmin"); plt.ylabel("Artık")
plt.show()
```

İyi bir modelde artıklar 0 çizgisi etrafında rastgele dağılır. Belirgin bir desen (eğri, huni şekli) varsa model bir şeyi yakalayamıyordur.

---

## 9. Özet

- Doğrusal regresyon `ŷ = wᵀx + b` modelini, **MSE'yi minimize ederek** öğrenir.
- **Kapalı form:** `θ = (XᵀX)⁻¹Xᵀy` — tek adımda kesin çözüm; çok öznitelikte pahalı.
- **Gradyan iniş:** `θ ← θ − α∇L` — adım adım iyileştirme; büyük veride ölçeklenebilir, sinir ağlarının da temeli.
- Öğrenme oranı çok küçükse yavaş, çok büyükse ıraksayan bir eğitim olur. Ölçekleme yakınsamayı hızlandırır.
- **Metrikler:** MSE, RMSE, MAE (hata büyüklüğü) ve R² (açıklanan varyans oranı).
- **Polinom regresyon** öznitelikleri genişleterek eğrisel ilişkileri modeller; derece çok artarsa aşırı öğrenme olur.

## Kendinizi Sınayın

1. x = [0, 1, 2], y = [1, 3, 5] verisi için w ve b'yi kapalı form ile elle hesaplayın.
2. Gradyan iniş sırasında kayıp her epoch'ta artıyorsa ilk neyi değiştirirsiniz?
3. Bir modelin test verisinde R² = −0.3 çıkması ne anlama gelir?
4. Polinom regresyon neden hâlâ "doğrusal" regresyon olarak adlandırılır?
5. 1 milyon örnek ve 50.000 öznitelik varsa normal denklemi mi gradyan inişi mi tercih edersiniz? Neden?

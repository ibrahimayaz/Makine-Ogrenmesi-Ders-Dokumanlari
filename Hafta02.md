# Hafta 2 — Veri Hazırlama ve Keşifsel Veri Analizi

## Bu Haftanın Hedefleri

- Keşifsel veri analizi (EDA) ile bir veri setini tanımak
- Eksik verileri ve aykırı değerleri ele almak
- Ölçekleme ve kategorik kodlamanın nasıl çalıştığını anlamak
- Veriyi eğitim / doğrulama / test olarak ayırmak ve **veri sızıntısını** önlemek

> "Çöp girerse çöp çıkar." En iyi algoritma bile kötü hazırlanmış veriden iyi sonuç çıkaramaz.

---

## 1. Keşifsel Veri Analizi (EDA)

EDA, modele geçmeden önce veriyi **tanıma** sürecidir. Temel sorular: Kaç satır/sütun var? Eksik değer var mı? Dağılımlar nasıl? Aykırı değer var mı? Öznitelikler hedefle ilişkili mi?

Bu hafta **Titanic** veri setini kullanıyoruz: yolcuların bilgileri ve hayatta kalıp kalmadıkları.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")
print(df.shape)          # (891, 15)
df.info()                # sütun tipleri ve dolu değer sayıları
df.describe()            # sayısal sütunların istatistikleri
```

**Dağılımlar ve hedefle ilişki:**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["age"], bins=30, ax=axes[0]).set_title("Yaş dağılımı")
sns.boxplot(x=df["fare"], ax=axes[1]).set_title("Bilet ücreti")
sns.barplot(x="class", y="survived", hue="sex", data=df, ax=axes[2]).set_title("Hayatta kalma oranı")
plt.show()

df.groupby("sex")["survived"].mean()     # female 0.742, male 0.189
```

Bu basit analiz bile önemli bir bilgi verir: Kadınların ve 1. sınıf yolcuların hayatta kalma oranı çok daha yüksek.

**Kutu grafiği okuma:** Kutu Q1 (%25) ile Q3 (%75) arasını, içindeki çizgi medyanı gösterir. Bıyıkların dışındaki noktalar **potansiyel aykırı değerlerdir**.

**Korelasyon:** İki sayısal değişken arasındaki doğrusal ilişkiyi −1 ile +1 arasında ölçer.

```python
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
```

> **Dikkat:** Korelasyon nedensellik değildir. Dondurma satışları ile boğulma vakaları ilişkilidir; ama ortak nedenleri sıcak havadır.

---

## 2. Eksik Veriler

```python
eksik = df.isnull().sum()
eksik[eksik > 0]
# age 177 (%19.9), embarked 2 (%0.2), deck 688 (%77.2), embark_town 2 (%0.2)
```

| Strateji | Ne zaman? | Örnek |
|---|---|---|
| Satırı sil | Eksik oranı çok düşükse | `embarked` (%0.2) |
| Sütunu sil | Sütunun büyük kısmı eksikse | `deck` (%77) |
| Medyan / ortalama ile doldur | Sayısal değişkenler (aykırı varsa medyan) | `age` |
| Mod ile doldur | Kategorik değişkenler | `embarked` |

**Neden aykırı değer varsa medyan?** Maaşlar [30, 32, 35, 38, 500] bin TL olsun. Ortalama 127 bin, hiçbir çalışanı temsil etmiyor. Medyan ise 35 bin, tipik çalışanı temsil ediyor.

```python
# Daha akıllı yöntem: yaşı, aynı sınıf ve cinsiyetteki yolcuların medyanıyla doldur
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda s: s.fillna(s.median()))
```

scikit-learn'de `SimpleImputer` aynı işi yapar. `fit` adımı medyanı **öğrenir**, `transform` adımı onu **uygular**. Bu ayrım, sızıntıyı önlemek için önemlidir (Bölüm 6).

---

## 3. Aykırı Değerler

Aykırı değer bir **hata** olabilir (yaş = 250) ya da **gerçek ama nadir** bir durum olabilir (milyarderin geliri). Hatalılar düzeltilir veya silinir; gerçek olanlar çoğu zaman bilgi taşır.

### IQR yöntemi — çalışma mantığı

1. Q1 ve Q3'ü bul, IQR = Q3 − Q1 hesapla.
2. Alt sınır = Q1 − 1.5·IQR, üst sınır = Q3 + 1.5·IQR.
3. Sınırların dışındaki değerler aykırıdır.

**Elle örnek:** `[2, 4, 4, 5, 6, 7, 8, 9, 10, 40]`

- Q1 = 4.25, Q3 = 8.75 → IQR = 4.5
- Sınırlar: [4.25 − 6.75, 8.75 + 6.75] = [−2.5, 15.5]
- 40 > 15.5 → **40 aykırıdır**

```python
veri = pd.Series([2, 4, 4, 5, 6, 7, 8, 9, 10, 40])
q1, q3 = veri.quantile([0.25, 0.75])
iqr = q3 - q1
print(veri[(veri < q1 - 1.5 * iqr) | (veri > q3 + 1.5 * iqr)].tolist())   # [40]
```

**Ne yapılır?** Hatalıysa silinir. Uç değerler kırpılabilir (`clip`) veya sağa çarpık değişkenlere (gelir, fiyat) **log dönüşümü** (`np.log1p`) uygulanabilir. Ağaç tabanlı modeller aykırı değerlere zaten dayanıklıdır.

---

## 4. Ölçekleme

### Neden gerekli?

**Yaş** (20–60) ve **maaş** (20.000–200.000) özniteliklerini düşünün:

```
Kişi A: yaş=25, maaş=50.000      Kişi B: yaş=55, maaş=51.000
Uzaklık = √(30² + 1000²) ≈ 1000.4   → 30 yıllık yaş farkı neredeyse hiç etki etmiyor!
```

Mesafe tabanlı modeller (k-NN, k-means, SVM) ve gradyan inişi kullanan modeller ölçekleme ister. **Ağaç tabanlı modeller** ölçeklemeden etkilenmez.

| Yöntem | Formül | Sonuç | Not |
|---|---|---|---|
| **Standardizasyon** | (x − μ) / σ | Ortalama 0, std 1 | En yaygın seçim |
| **Min-Max** | (x − min) / (max − min) | [0, 1] aralığı | Aykırı değerlere hassas |
| **Robust** | (x − medyan) / IQR | — | Aykırı değer varsa |

**Elle örnek (standardizasyon):** Yaşlar = [20, 30, 40, 50, 60]
- μ = 40, σ = √200 ≈ 14.14
- x' = [−1.41, −0.71, 0, 0.71, 1.41]

**Sıfırdan bir StandardScaler:**

```python
class BenimStandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)            # istatistikleri ÖĞREN
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_  # öğrenilenleri UYGULA

X = np.array([[25, 50000], [55, 51000], [40, 120000], [30, 80000]], dtype=float)
print(BenimStandardScaler().fit(X).transform(X).round(2))

from sklearn.preprocessing import StandardScaler
print(StandardScaler().fit_transform(X).round(2))     # aynı sonuç
```

**Önemli:** Test verisi, **eğitim verisinden öğrenilen** μ ve σ ile ölçeklenir.

---

## 5. Kategorik Değişkenlerin Kodlanması

**Sıralı değişkenler → Ordinal kodlama:** Doğal bir sıra varsa (ilkokul < lise < lisans) kategoriler 0, 1, 2 gibi sayılara eşlenir.

**Sırasız değişkenler → One-hot kodlama:** Şehir veya renk gibi değişkenlerde Ankara=0, İstanbul=1, İzmir=2 dersek model "İzmir > İstanbul" gibi anlamsız ilişkiler öğrenebilir. Bu yüzden her kategori için ayrı bir 0/1 sütunu oluşturulur:

| sehir | sehir_Ankara | sehir_İstanbul | sehir_İzmir |
|---|---|---|---|
| Ankara | 1 | 0 | 0 |
| İzmir | 0 | 0 | 1 |

```python
egitim = pd.Series(["lise", "lisans", "ilkokul"])
print(egitim.map({"ilkokul": 0, "lise": 1, "lisans": 2}).tolist())   # ordinal: [1, 2, 0]

pd.get_dummies(df["embarked"], prefix="embarked")                    # one-hot
```

scikit-learn'de `OneHotEncoder(handle_unknown="ignore")` kullanılır. Bu ayarla, testte eğitimde görülmemiş bir kategori çıkarsa hata verilmez.

---

## 6. Eğitim / Doğrulama / Test Ayrımı ve Veri Sızıntısı

### 6.1 Neden ayırıyoruz?

Sınav sorularını önceden gören bir öğrenci 100 alır, ama konuyu öğrenip öğrenmediğini bilemeyiz. Model de eğitim verisinde başarılı olabilir, ama asıl önemli olan **yeni veride** başarısıdır.

| Set | Oran | Görevi |
|---|---|---|
| Eğitim | %60–70 | Model parametrelerini öğrenir |
| Doğrulama | %15–20 | Modelleri ve ayarları karşılaştırır |
| Test | %15–20 | Son performansı ölçer — **yalnızca en sonda, bir kez** |

```python
from sklearn.model_selection import train_test_split

X, y = df[["pclass", "age", "fare"]], df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

- `random_state`: Bölmenin her seferinde aynı olmasını sağlar.
- `stratify=y`: Sınıf oranlarını her parçada korur.
- **Zaman serilerinde** rastgele bölme yapılmaz. Geçmiş verilerle eğitilir, gelecek verilerle test edilir.

### 6.2 Veri sızıntısı

**Veri sızıntısı**, modelin eğitimde, gerçek kullanımda erişemeyeceği bir bilgiyi görmesidir. Böyle bir model değerlendirmede harika görünür ama gerçek dünyada başarısız olur.

**1. Ön işlemede sızıntı (en yaygın hata):**

```python
# YANLIŞ: Önce tüm veriyi ölçekle, sonra böl → test verisinin ortalaması da hesaba katıldı!
X_scaled = scaler.fit_transform(X_sayisal)
X_train, X_test = train_test_split(X_scaled, ...)

# DOĞRU: Önce böl, sonra sadece eğitim verisinde fit et
X_train, X_test = train_test_split(X_sayisal, ...)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

**2. Hedef sızıntısı:** Hedef oluştuktan **sonra** ortaya çıkan bir öznitelik kullanmak. Titanic'teki `alive` sütunu, `survived` sütununun metin halidir ve doğrudan cevabı verir.

### 6.3 Çözüm: Pipeline

`Pipeline` ve `ColumnTransformer` tüm ön işleme adımlarını modelle birlikte paketler. `fit` çağrıldığında her şey **yalnızca eğitim verisinden** öğrenilir.

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

sayisal = ["age", "fare", "sibsp", "parch"]
kategorik = ["sex", "embarked", "pclass"]

on_isleme = ColumnTransformer([
    ("say", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), sayisal),
    ("kat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), kategorik),
])
model = Pipeline([("on_isleme", on_isleme), ("model", LogisticRegression(max_iter=1000))])

X, y = df[sayisal + kategorik], df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
print("Test doğruluğu:", round(model.score(X_test, y_test), 3))
```

---

## 7. Laboratuvar: Titanic Üzerinde EDA ve Ön İşleme

1. Veriyi yükleyin. Sızıntı yaratan (`alive`), başka sütunların tekrarı olan (`class`, `who`, `adult_male`, `embark_town`) ve çoğu eksik olan (`deck`) sütunları çıkarın.
2. En az 4 grafikle veriyi inceleyin ve her grafiği 1–2 cümleyle yorumlayın.
3. Yeni bir öznitelik türetin: `aile_buyuklugu = sibsp + parch + 1`.
4. Bölüm 6.3'teki Pipeline ile modeli eğitip test doğruluğunu raporlayın.
5. **Sızıntı deneyi:** `alive` sütununu modele ekleyin. Doğruluk neden %100'e yaklaşıyor? Bu bir başarı mı?

```python
df = sns.load_dataset("titanic").drop(
    columns=["alive", "class", "who", "adult_male", "embark_town", "deck"])
df["aile_buyuklugu"] = df["sibsp"] + df["parch"] + 1
df.groupby("aile_buyuklugu")["survived"].mean().plot(kind="bar")
plt.show()
```

---

## 8. Özet

| Konu | Anahtar fikir |
|---|---|
| EDA | Modelden önce veriyi tanı |
| Eksik veri | Sil veya doldur (sayısal: medyan, kategorik: mod) |
| Aykırı değer | IQR ile tespit et; hata mı, gerçek mi karar ver |
| Ölçekleme | Mesafe ve gradyan tabanlı modeller için şart |
| Kategorik kodlama | Sıralıysa ordinal, değilse one-hot |
| Veri ayrımı ve sızıntı | Test seti en sonda kullanılır; `fit` sadece eğitim verisinde yapılır → Pipeline |

## Kendinizi Sınayın

1. Gelir sütununda eksik değerler ve birkaç çok yüksek gelir var. Ortalama mı, medyan mı kullanırsınız? Neden?
2. "Renk" sütununu 0, 1, 2 olarak kodlamak neden sorunludur?
3. Tüm veriyi ölçekledikten sonra eğitim/test ayrımı yapmak neden yanlıştır?
4. `[3, 5, 7, 8, 9, 11, 13, 60]` verisindeki aykırı değerleri IQR yöntemiyle elle bulun.

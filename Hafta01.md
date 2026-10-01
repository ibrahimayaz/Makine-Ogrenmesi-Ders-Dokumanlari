# Hafta 1 — Makine Öğrenmesine Giriş

## Bu Haftanın Hedefleri

Bu haftanın sonunda şunları yapabiliyor olacaksınız:

- Makine öğrenmesinin (ML) ne olduğunu ve klasik programlamadan farkını açıklamak
- Denetimli, denetimsiz ve pekiştirmeli öğrenmeyi birbirinden ayırt etmek
- Bir ML projesinin adımlarını (iş akışını) sıralamak
- Python ortamını kurmak ve NumPy ile pandas'ın temel işlemlerini kullanmak

---

## 1. Makine Öğrenmesi Nedir?

### 1.1 Klasik programlama ile karşılaştırma

Klasik programlamada **kuralları biz yazarız**, bilgisayar bu kuralları veriye uygular:

```
Veri + Kurallar  ──►  Program  ──►  Cevaplar
```

Makine öğrenmesinde ise **veriyi ve cevapları veririz**, bilgisayar kuralları kendisi çıkarır:

```
Veri + Cevaplar  ──►  Öğrenme algoritması  ──►  Kurallar (model)
```

**Örnek: Spam e-posta tespiti**

Klasik yaklaşımda şöyle kurallar yazardık:

```python
def spam_mi(eposta):
    if "bedava" in eposta or "kazandınız" in eposta or "tıklayın" in eposta:
        return True
    return False
```

Bu yaklaşımın sorunları:

- Spam gönderenler "b3dava" yazarak kuralı atlatır.
- Yüzlerce kuralı elle yazmak ve güncel tutmak çok zordur.
- Hangi kelimenin ne kadar önemli olduğunu bilemeyiz.

ML yaklaşımında ise binlerce e-postayı "spam" / "spam değil" etiketiyle algoritmaya veririz. Algoritma hangi kelimelerin, hangi kalıpların spam'e işaret ettiğini **kendisi öğrenir**. Yeni spam türleri çıktığında modeli yeni verilerle yeniden eğitmemiz yeterlidir.

### 1.2 Tanım

Tom Mitchell'in (1997) sık kullanılan tanımı:

> Bir bilgisayar programı, bir **T** görevinde, **P** performans ölçüsüne göre, **E** deneyiminden öğreniyorsa; yani T'deki performansı (P ile ölçüldüğünde) E deneyimiyle artıyorsa, o program öğreniyor denir.

Spam örneğinde:

| Bileşen | Spam örneği |
|---|---|
| **T** (görev) | E-postayı spam / spam değil diye sınıflandırmak |
| **P** (performans) | Doğru sınıflandırılan e-postaların oranı |
| **E** (deneyim) | Daha önce etiketlenmiş e-postalar |

### 1.3 Temel kavramlar

Ev fiyatı tahmini örneği üzerinden temel terimleri tanıyalım:

| Metrekare | Oda sayısı | Bina yaşı | Fiyat (bin TL) |
|---|---|---|---|
| 120 | 3 | 5 | 4500 |
| 85 | 2 | 20 | 2800 |
| 150 | 4 | 2 | 6100 |

- **Örnek (sample, gözlem):** Tablodaki her satır. Burada her satır bir ev.
- **Öznitelik (feature):** Girdi olarak kullanılan sütunlar: metrekare, oda sayısı, bina yaşı. Genelde `X` ile gösterilir.
- **Hedef / etiket (target / label):** Tahmin etmek istediğimiz sütun: fiyat. Genelde `y` ile gösterilir.
- **Model:** Özniteliklerden hedefi tahmin eden fonksiyon: `ŷ = f(X)`.
- **Eğitim (training):** Modelin parametrelerini veriye bakarak ayarlama süreci.
- **Çıkarım (inference / prediction):** Eğitilmiş modelle yeni veri için tahmin üretme.

Matematiksel olarak `X` bir matristir: `n` satır (örnek) ve `d` sütun (öznitelik). Yukarıdaki tabloda `n = 3`, `d = 3`.

---

## 2. Öğrenme Türleri

### 2.1 Denetimli öğrenme (Supervised Learning)

Veride **doğru cevaplar (etiketler) vardır**. Model, girdiden çıktıya giden eşlemeyi öğrenir.

İki ana türü vardır:

**Regresyon:** Hedef **sürekli bir sayıdır**.
- Ev fiyatı tahmini
- Yarın hava sıcaklığının kaç derece olacağı
- Bir ürünün gelecek ay kaç adet satacağı

**Sınıflandırma:** Hedef **bir kategoridir**.
- E-posta spam mi değil mi? (ikili sınıflandırma)
- Bir görüntüde kedi mi, köpek mi, kuş mu var? (çok sınıflı sınıflandırma)
- Bir müşteri kredisini ödeyebilir mi?

> **İpucu:** "Cevap bir sayı mı, bir kategori mi?" sorusunu sorun. Sayıysa regresyon, kategoriyse sınıflandırma. Dikkat: "1, 2, 3" diye kodlanmış ürün kategorileri hâlâ sınıflandırmadır; çünkü 3, 1'in "üç katı" değildir.

### 2.2 Denetimsiz öğrenme (Unsupervised Learning)

Veride **etiket yoktur**. Model, verinin kendi içindeki yapıyı keşfeder.

- **Kümeleme:** Benzer müşterileri gruplara ayırmak (müşteri segmentasyonu)
- **Boyut indirgeme:** 784 pikselli bir görüntüyü 2 boyuta indirip görselleştirmek
- **Anomali tespiti:** Kredi kartı işlemlerinde olağandışı olanları bulmak

Örnek: Bir e-ticaret sitesinin elinde 1 milyon müşterinin alışveriş geçmişi var ama kimse bu müşterileri gruplara ayırmamış. Kümeleme algoritması "hafta sonu indirim avcıları", "sadık premium müşteriler" gibi grupları kendisi bulabilir.

### 2.3 Pekiştirmeli öğrenme (Reinforcement Learning)

Bir **ajan (agent)**, bir **ortamda (environment)** **eylemler (action)** yapar ve karşılığında **ödül (reward)** veya ceza alır. Amaç toplam ödülü en büyük yapacak davranışı (politika) öğrenmektir.

```
        eylem
  Ajan ───────►  Ortam
   ▲               │
   └── durum, ödül ┘
```

- Satranç / Go oynayan programlar (AlphaGo)
- Robotun yürümeyi öğrenmesi
- Trafik ışıklarının optimize edilmesi

Pekiştirmeli öğrenmede "doğru cevap" önceden verilmez; ajan deneme-yanılma ile öğrenir. Bu ders denetimli ve denetimsiz öğrenmeye odaklanacak.

### 2.4 Özet tablo

| Tür | Veri | Amaç | Örnek algoritmalar (bu derste) |
|---|---|---|---|
| Denetimli – Regresyon | X, y (sayı) | Sayı tahmini | Doğrusal regresyon, Ridge, Lasso, Random Forest |
| Denetimli – Sınıflandırma | X, y (kategori) | Sınıf tahmini | Lojistik regresyon, k-NN, Naive Bayes, Karar ağacı, SVM |
| Denetimsiz | Sadece X | Yapı keşfi | k-means, DBSCAN, PCA, t-SNE |
| Pekiştirmeli | Ortam + ödül | Politika öğrenme | (Bu dersin kapsamı dışında) |

**Kısa alıştırma:** Aşağıdakilerin her biri hangi öğrenme türüdür?

1. Bir hastanın tümörünün iyi huylu mu kötü huylu mu olduğunu tahmin etmek
2. Haber makalelerini konu etiketi olmadan benzerliklerine göre gruplamak
3. Bir öğrencinin final notunu (0–100) tahmin etmek
4. Bir oyunda karakterin engellerden kaçmayı öğrenmesi

<details>
<summary>Cevaplar</summary>

1. Denetimli – sınıflandırma
2. Denetimsiz – kümeleme
3. Denetimli – regresyon
4. Pekiştirmeli öğrenme

</details>

---

## 3. Makine Öğrenmesi İş Akışı

Gerçek bir ML projesi yalnızca "model eğitmekten" ibaret değildir. Tipik adımlar:

```
1. Problemi tanımla
       │
2. Veri topla
       │
3. Keşifsel veri analizi (EDA)
       │
4. Veri hazırlama / ön işleme
       │
5. Model seç ve eğit
       │
6. Değerlendir  ──── (yetersizse 3, 4 veya 5'e dön)
       │
7. Dağıt (deploy) ve izle
```

1. **Problemi tanımla:** Ne tahmin edilecek? Başarı nasıl ölçülecek? Örnek: "Müşterinin önümüzdeki ay aboneliği iptal edip etmeyeceğini tahmin etmek; iptal edecek müşterilerin en az %80'ini yakalamak."
2. **Veri topla:** Veritabanları, API'ler, anketler, sensörler.
3. **EDA:** Veriyi tanı. Dağılımlar nasıl? Eksik değer var mı? Öznitelikler hedefle ilişkili mi?
4. **Ön işleme:** Eksik verileri doldur, kategorik değişkenleri sayıya çevir, ölçekle, veriyi eğitim/test olarak ayır (Hafta 2).
5. **Model eğitimi:** Basit bir modelle başla (baseline), sonra daha karmaşık modelleri dene.
6. **Değerlendirme:** Model daha önce görmediği test verisinde ne kadar başarılı? (Hafta 4–5)
7. **Dağıtım:** Modeli bir uygulamaya entegre et, performansını zaman içinde izle (Hafta 14).

> **Pratikte:** Projelerde zamanın büyük kısmı (çoğu zaman %60–80) veri toplama, temizleme ve hazırlamaya gider. Model eğitimi genellikle en kısa adımdır.

### 3.1 İlk ML programımız: iş akışını uçtan uca görelim

Henüz algoritmaların detaylarını bilmiyoruz ama tüm akışı bir kez görmek faydalı. Aşağıda iris çiçeği veri setiyle bir sınıflandırma yapıyoruz:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Veri topla
iris = load_iris()
X, y = iris.data, iris.target   # X: 150x4 matris, y: 150 etiket (0, 1, 2)
print("Öznitelikler:", iris.feature_names)
print("Sınıflar:", iris.target_names)

# 4. Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model seç ve eğit
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 6. Değerlendir
y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))

# Yeni bir çiçek için tahmin
yeni_cicek = [[5.1, 3.5, 1.4, 0.2]]  # çanak uzunluğu, çanak genişliği, taç uzunluğu, taç genişliği
print("Tahmin:", iris.target_names[model.predict(yeni_cicek)[0]])
```

scikit-learn'deki tüm modeller aynı arayüzü kullanır:

- `model.fit(X, y)` → eğit
- `model.predict(X)` → tahmin et
- `model.score(X, y)` → hızlı değerlendirme

Bu tutarlılık sayesinde modeli değiştirmek tek satırlık bir iştir.

---

## 4. Uygulama Alanları

| Alan | Örnek uygulama | Problem türü |
|---|---|---|
| Sağlık | Röntgen görüntüsünden zatürre tespiti | Sınıflandırma |
| Finans | Kredi riski skoru, sahtekarlık tespiti | Sınıflandırma, anomali tespiti |
| E-ticaret | Ürün önerisi, müşteri segmentasyonu | Öneri sistemleri, kümeleme |
| Ulaşım | Varış süresi tahmini, otonom sürüş | Regresyon, derin öğrenme |
| Dil | Makine çevirisi, sohbet botları | Dizi modelleri, Transformer |
| Üretim | Arıza öngörüsü (predictive maintenance) | Sınıflandırma, regresyon |
| Tarım | Hasat miktarı tahmini, bitki hastalığı tespiti | Regresyon, görüntü sınıflandırma |

### Ne zaman ML kullanmalı, ne zaman kullanmamalı?

ML **iyi bir seçimdir**:
- Kurallar çok karmaşıksa veya bilinmiyorsa (yüz tanıma)
- Ortam sürekli değişiyorsa (spam, sahtekarlık)
- Elde çok miktarda veri varsa

ML **gereksiz olabilir**:
- Basit ve net kurallar yeterliyse (KDV hesaplama)
- Yeterli veri yoksa
- Hatanın kabul edilemez olduğu ve kararın tam açıklanması gereken durumlarda (dikkatli olunmalı)

---

## 5. Laboratuvar: Ortam Kurulumu ve NumPy / pandas Tekrarı

### 5.1 Ortam kurulumu

**Seçenek A — Google Colab (önerilen başlangıç):**
1. https://colab.research.google.com adresine gidin.
2. "Yeni not defteri" oluşturun. NumPy, pandas, scikit-learn, matplotlib hazır kurulu gelir.

**Seçenek B — Yerel kurulum (Anaconda veya pip):**

```bash
# Sanal ortam oluşturma (önerilir)
python -m venv ml-ders
# Windows:
ml-ders\Scripts\activate
# macOS / Linux:
source ml-ders/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Kurulumu kontrol edin:

```python
import numpy as np, pandas as pd, sklearn, matplotlib
print(np.__version__, pd.__version__, sklearn.__version__, matplotlib.__version__)
```

### 5.2 NumPy tekrarı

NumPy, ML'nin temelini oluşturan **çok boyutlu dizi (array)** kütüphanesidir. Python listelerine göre çok daha hızlıdır çünkü işlemler C dilinde, vektörel olarak yapılır.

```python
import numpy as np

# Dizi oluşturma
a = np.array([1, 2, 3, 4])
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)      # (2, 3) -> 2 satır, 3 sütun
print(M.ndim)       # 2 boyutlu

# Hazır diziler
np.zeros((2, 3))         # 2x3 sıfır matrisi
np.ones(5)               # 5 elemanlı birler
np.arange(0, 10, 2)      # [0 2 4 6 8]
np.linspace(0, 1, 5)     # [0. 0.25 0.5 0.75 1.]
np.random.seed(0)
np.random.randn(3)       # standart normal dağılımdan 3 sayı
```

**Vektörel işlemler (döngüsüz hesaplama):**

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

print(x + y)     # [11 22 33]  eleman eleman toplama
print(x * y)     # [10 40 90]  eleman eleman çarpma
print(x @ y)     # 140         iç çarpım: 1*10 + 2*20 + 3*30
print(x ** 2)    # [1 4 9]
print(np.sqrt(x), np.exp(x), np.log(x))
```

Neden önemli? Döngü ile 1 milyon sayının karesini almak ile NumPy'ın vektörel işlemini karşılaştıralım:

```python
import time
liste = list(range(1_000_000))
dizi = np.arange(1_000_000)

t = time.time(); [v**2 for v in liste]; print("Döngü :", time.time() - t)
t = time.time(); dizi**2;               print("NumPy :", time.time() - t)
```

NumPy genellikle 10–100 kat daha hızlıdır.

**Yayınlama (broadcasting):** Farklı boyuttaki diziler uyumluysa NumPy küçük olanı otomatik "genişletir".

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])
ortalama = X.mean(axis=0)   # sütun ortalamaları: [2.5 3.5 4.5]
X_merkez = X - ortalama     # her satırdan ortalama çıkarılır
print(X_merkez)
# [[-1.5 -1.5 -1.5]
#  [ 1.5  1.5  1.5]]
```

Bu işlem ileride "ölçekleme" konusunda çok işimize yarayacak.

**İndeksleme ve filtreleme:**

```python
X = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
X[0]          # ilk satır
X[:, 1]       # ikinci sütun
X[1:, 2:]     # alt-sağ blok
X[X > 5]      # 5'ten büyük elemanlar: [ 6  7  8  9 10 11]
```

**`axis` kavramı:** `axis=0` satırlar boyunca (yani her sütun için), `axis=1` sütunlar boyunca (yani her satır için) işlem yapar.

```python
X.sum(axis=0)   # her sütunun toplamı: [12 15 18 21]
X.sum(axis=1)   # her satırın toplamı: [ 6 22 38]
```

**Matris işlemleri (doğrusal regresyonda kullanacağız):**

```python
A = np.array([[2, 1], [1, 3]])
A.T                   # transpoz
np.linalg.inv(A)      # ters matris
A @ np.linalg.inv(A)  # birim matris (yaklaşık)
```

### 5.3 pandas tekrarı

pandas, tablo şeklindeki verilerle çalışmak için kullanılır. Temel yapısı **DataFrame**'dir (satır ve sütunlardan oluşan tablo).

```python
import pandas as pd

df = pd.DataFrame({
    "isim":   ["Ali", "Ayşe", "Mehmet", "Zeynep", "Can"],
    "yas":    [23, 31, 27, None, 45],
    "sehir":  ["Ankara", "İstanbul", "İzmir", "Ankara", "İstanbul"],
    "maas":   [32000, 55000, 41000, 38000, 72000],
})

df.head()          # ilk 5 satır
df.info()          # sütun tipleri ve eksik değer sayıları
df.describe()      # sayısal sütunların istatistikleri
df.shape           # (5, 4)
```

**Seçme ve filtreleme:**

```python
df["maas"]                         # tek sütun (Series)
df[["isim", "maas"]]               # birden çok sütun
df[df["maas"] > 40000]             # koşullu filtreleme
df[(df["sehir"] == "Ankara") & (df["maas"] > 35000)]
df.loc[0, "isim"]                  # etikete göre erişim
df.iloc[0, 0]                      # konuma göre erişim
```

**Eksik değerler:**

```python
df.isnull().sum()                          # her sütundaki eksik sayısı
df["yas"] = df["yas"].fillna(df["yas"].median())   # medyanla doldur
```

**Gruplama ve özetleme:**

```python
df.groupby("sehir")["maas"].mean()
# sehir
# Ankara      35000.0
# İstanbul    63500.0
# İzmir       41000.0

df["sehir"].value_counts()
```

**Yeni sütun oluşturma:**

```python
df["maas_bin"] = df["maas"] / 1000
df["kidemli"] = df["yas"] > 30
```

**Dosya okuma / yazma:**

```python
df.to_csv("calisanlar.csv", index=False)
df2 = pd.read_csv("calisanlar.csv")
```

**pandas'tan NumPy'a geçiş (modellere veri verirken):**

```python
X = df[["yas", "maas"]].to_numpy()
```

### 5.4 Laboratuvar görevleri

1. Colab veya yerel ortamda yeni bir not defteri açın ve kütüphane sürümlerini yazdırın.
2. `np.random.seed(42)` ile 100 öğrencilik rastgele bir not tablosu oluşturun: `vize` (0–100), `final` (0–100). `ortalama = 0.4*vize + 0.6*final` sütununu hesaplayın.
3. Bu tabloyu pandas DataFrame'e çevirin; ortalaması 50'nin altında kalan öğrenci sayısını bulun.
4. `harf_notu` sütunu oluşturun (90+ AA, 80+ BA, 70+ BB, 60+ CB, 50+ CC, altı FF). İpucu: `pd.cut`.
5. Seaborn'un `titanic` veri setini yükleyin (`sns.load_dataset("titanic")`), sınıfa (`pclass`) göre hayatta kalma oranlarını hesaplayın.

```python
# Görev 2-4 için başlangıç kodu
import numpy as np, pandas as pd
np.random.seed(42)
df = pd.DataFrame({
    "vize": np.random.randint(0, 101, 100),
    "final": np.random.randint(0, 101, 100),
})
df["ortalama"] = 0.4 * df["vize"] + 0.6 * df["final"]
df["harf_notu"] = pd.cut(df["ortalama"],
                         bins=[-1, 49.99, 59.99, 69.99, 79.99, 89.99, 100],
                         labels=["FF", "CC", "CB", "BB", "BA", "AA"])
print(df["harf_notu"].value_counts().sort_index())
```

---

## 6. Özet

- ML, kuralları elle yazmak yerine **veriden öğrenen** sistemler kurmaktır.
- **Denetimli** öğrenmede etiket vardır (regresyon: sayı, sınıflandırma: kategori). **Denetimsiz** öğrenmede etiket yoktur (kümeleme, boyut indirgeme). **Pekiştirmeli** öğrenmede ajan ödülle öğrenir.
- ML iş akışı: problem → veri → EDA → ön işleme → model → değerlendirme → dağıtım.
- scikit-learn'de tüm modeller `fit` / `predict` arayüzünü kullanır.
- NumPy vektörel hesaplama, pandas tablo işlemleri için temel araçlarımızdır.

## Kendinizi Sınayın

1. Bir bankanın, geçmiş müşteri verilerine bakarak yeni başvuranın krediyi geri ödeyip ödemeyeceğini tahmin etmesi hangi öğrenme türüdür? T, P, E'yi tanımlayın.
2. `X.mean(axis=0)` ile `X.mean(axis=1)` arasındaki fark nedir?
3. Neden modelin başarısını eğitim verisinde değil de ayrı bir test verisinde ölçeriz? (Hafta 2 ve 4'te detaylı göreceğiz.)
4. Bir müşteri yorumunun olumlu mu olumsuz mu olduğunu tahmin eden bir sistem için ML iş akışının adımlarını kısaca yazın.

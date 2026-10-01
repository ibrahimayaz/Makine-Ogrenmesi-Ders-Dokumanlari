# Hafta 11 — Boyut İndirgeme ve Öznitelik Seçimi

## Bu Haftanın Hedefleri

- Boyut indirgemenin neden gerekli olduğunu anlamak
- **PCA**'nın çalışma mantığını kovaryans matrisi, özdeğer ve özvektörlerle adım adım görmek
- Açıklanan varyans oranını yorumlamak ve bileşen sayısını seçmek
- **t-SNE** ve **UMAP**'in nasıl çalıştığını ve grafiklerinin nasıl (ve nasıl **okunmaması** gerektiğini) öğrenmek
- **Öznitelik seçimi** için filtre, sarmalayıcı ve gömülü yöntemleri uygulamak
- MNIST rakam görüntülerini PCA ve t-SNE ile görselleştirmek

---

## 1. Neden Boyut İndirgeme?

Gerçek veri setlerinde yüzlerce, binlerce öznitelik olabilir: 28×28 bir görüntüde 784 piksel, bir metinde on binlerce kelime, bir gen ifade verisinde 20.000 gen.

**Sorunlar:**
1. **Boyutluluk laneti** (Hafta 6): Mesafeler anlamını yitirir, veri seyrekleşir.
2. **Aşırı öğrenme:** Öznitelik sayısı örnek sayısına yaklaştıkça model gürültüyü ezberler.
3. **Hesaplama maliyeti:** Eğitim süresi ve bellek kullanımı artar.
4. **Görselleştirme:** İnsanlar en fazla 3 boyutu görebilir.
5. **Gereksizlik (redundancy):** Birçok öznitelik birbirine çok benzer bilgi taşır (ör. metrekare ve oda sayısı).

**İki yaklaşım:**

| | **Öznitelik çıkarımı** (feature extraction) | **Öznitelik seçimi** (feature selection) |
|---|---|---|
| Ne yapar? | Mevcut özniteliklerin **birleşiminden yeni** öznitelikler üretir | Mevcut özniteliklerin bir **alt kümesini** seçer |
| Örnek | PCA, t-SNE, UMAP | Varyans eşiği, RFE, Lasso |
| Yorumlanabilirlik | Yeni eksenleri yorumlamak zor | Orijinal öznitelikler korunur |

---

# BÖLÜM A: Temel Bileşen Analizi (PCA)

## 2. PCA'nın Sezgisi

İki öznitelik düşünelim: öğrencilerin **matematik** ve **fizik** notları. Bu iki not birbiriyle güçlü biçimde ilişkilidir:

```
 Fizik
  │              ● ●
  │           ● ● ●        PC1 (ana eğilim: "sayısal yetenek")
  │        ● ●●  ╱
  │      ● ● ●╱ ●
  │    ● ● ╱● ●
  │   ●  ╱ ●            PC2 (PC1'e dik, küçük sapmalar)
  │    ╱ ●                ↖
  │  ╱                      ↖
  └───────────────────── Matematik
```

Veri neredeyse tek bir doğru boyunca uzanıyor. Her öğrenciyi iki sayı yerine **tek bir sayıyla** (doğru üzerindeki konumuyla) temsil etsek çok az bilgi kaybederiz.

**PCA'nın amacı:** Verideki **varyansın en büyük olduğu yönleri** bulmak ve veriyi bu yönlere izdüşürmek.

- **1. temel bileşen (PC1):** Verinin en çok yayıldığı (varyansı en büyük) yön
- **2. temel bileşen (PC2):** PC1'e **dik** yönler arasında varyansı en büyük olan
- ... ve böyle devam eder. d boyutlu veride d tane bileşen vardır.

**Neden varyans?** Varyans = bilgi. Tüm öğrencilerin aynı değere sahip olduğu bir yön, öğrencileri birbirinden ayırt etmemize hiç yardım etmez.

**Eşdeğer bir bakış:** PC1, noktalardan **dik uzaklıkların kareleri toplamını en küçük** yapan doğrudur (izdüşüm hatasını minimize eder). Varyansı maksimize etmek ile izdüşüm hatasını minimize etmek aynı şeydir.

> Not: Bu, doğrusal regresyondan farklıdır. Regresyon **dikey** (y yönündeki) hataları minimize eder ve bir değişkeni hedef olarak ayırır. PCA tüm değişkenlere eşit davranır ve **dik** uzaklıkları minimize eder.

---

## 3. PCA Algoritması

```
1. Veriyi merkezle: her sütundan ortalamasını çıkar (genelde standardize de edilir)
2. Kovaryans matrisini hesapla:  Σ = (1/(n−1)) · XᵀX        (d × d)
3. Σ'nın özdeğerlerini (λ) ve özvektörlerini (v) bul
4. Özvektörleri özdeğerlere göre büyükten küçüğe sırala
5. İlk k özvektörü seç → W (d × k)
6. İzdüşür:  Z = X · W                                       (n × k)
```

### 3.1 Kovaryans matrisi

Kovaryans, iki değişkenin birlikte nasıl değiştiğini ölçer:

```
cov(x, y) = (1/(n−1)) Σ (xᵢ − x̄)(yᵢ − ȳ)
```

Kovaryans matrisinde köşegen = her değişkenin **varyansı**, köşegen dışı = çiftlerin **kovaryansı**:

```
     ┌ var(x)     cov(x, y) ┐
Σ =  │                      │
     └ cov(y, x)  var(y)    ┘
```

### 3.2 Özdeğer ve özvektör: Ne anlama gelir?

Bir matris A ile çarpıldığında **yönü değişmeyen**, sadece uzayan veya kısalan vektörlere **özvektör** denir:

```
A · v = λ · v
```

λ, uzama miktarıdır (**özdeğer**).

**Kovaryans matrisi için yorum:**
- **Özvektörler** → verinin yayıldığı ana yönler (temel bileşenler)
- **Özdeğerler** → o yöndeki **varyans miktarı**

```
          v₁ (λ₁ büyük: veri bu yönde çok yayılmış)
         ╱
    ●  ●╱●  ●
   ● ●╱● ●
  ● ●╱ ●
    ╱  ╲
       ╲ v₂ (λ₂ küçük: bu yönde az yayılmış)
```

Kovaryans matrisi simetrik olduğu için özvektörleri **birbirine diktir** ve özdeğerleri **negatif değildir**.

### 3.3 Elle bir örnek

10 noktalık 2 boyutlu veri:

| x | 2.5 | 0.5 | 2.2 | 1.9 | 3.1 | 2.3 | 2.0 | 1.0 | 1.5 | 1.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| y | 2.4 | 0.7 | 2.9 | 2.2 | 3.0 | 2.7 | 1.6 | 1.1 | 1.6 | 0.9 |

**Adım 1 — Merkezleme:** x̄ = 1.81, ȳ = 1.91. Her değerden ortalamayı çıkarırız. Örneğin ilk nokta: (2.5 − 1.81, 2.4 − 1.91) = **(0.69, 0.49)**

**Adım 2 — Kovaryans matrisi:**

```
     ┌ 0.6166   0.6154 ┐
Σ =  │                 │
     └ 0.6154   0.7166 ┘
```

Kovaryans pozitif ve büyük → x ve y birlikte artıyor.

**Adım 3 — Özdeğerler:** 2×2 bir matris için karakteristik denklem det(Σ − λI) = 0:

```
λ² − (iz)·λ + det = 0
iz  = 0.6166 + 0.7166 = 1.3332
det = 0.6166 · 0.7166 − 0.6154² = 0.4419 − 0.3787 = 0.0631

λ = (1.3332 ± √(1.3332² − 4·0.0631)) / 2 = (1.3332 ± 1.2349) / 2

λ₁ = 1.2840      λ₂ = 0.0491
```

**Adım 4 — Özvektörler:** λ₁ için (Σ − λ₁I)v = 0:

```
(0.6166 − 1.2840)·v₁ + 0.6154·v₂ = 0   →   v₂ = 1.0845·v₁
Birim uzunluğa normalize:  v₁ = (0.6779, 0.7352)
```

Benzer şekilde λ₂ için: v₂ = (−0.7352, 0.6779). İki vektörün iç çarpımı: 0.6779·(−0.7352) + 0.7352·0.6779 = 0 → **diktirler** ✓

**Adım 5 — Açıklanan varyans oranı:**

```
PC1: 1.2840 / (1.2840 + 0.0491) = 0.963   →  %96.3
PC2: 0.0491 / 1.3331           = 0.037   →  %3.7
```

**Yorum:** Verinin %96.3'ü tek bir boyutla açıklanabiliyor. 2 boyutu 1'e indirirsek bilginin sadece %3.7'sini kaybederiz.

**Adım 6 — İzdüşüm:** İlk nokta için PC1 koordinatı:

```
z₁ = (0.69, 0.49) · (0.6779, 0.7352) = 0.468 + 0.360 = 0.828
```

### 3.4 Sıfırdan PCA

```python
import numpy as np
import matplotlib.pyplot as plt

class BenimPCA:
    def __init__(self, n_bilesen):
        self.n_bilesen = n_bilesen

    def fit(self, X):
        self.ortalama = X.mean(axis=0)
        Xc = X - self.ortalama                              # 1) merkezle
        kov = Xc.T @ Xc / (len(X) - 1)                      # 2) kovaryans
        ozdeger, ozvektor = np.linalg.eigh(kov)             # 3) simetrik matris için eigh
        sira = np.argsort(ozdeger)[::-1]                    # 4) büyükten küçüğe
        self.ozdegerler = ozdeger[sira]
        self.bilesenler = ozvektor[:, sira][:, :self.n_bilesen].T   # (k, d)
        self.aciklanan_varyans_orani = self.ozdegerler[:self.n_bilesen] / ozdeger.sum()
        return self

    def transform(self, X):
        return (X - self.ortalama) @ self.bilesenler.T      # 6) izdüşür

    def inverse_transform(self, Z):
        return Z @ self.bilesenler + self.ortalama           # geri dönüşüm (yaklaşık)

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
              [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])
pca = BenimPCA(n_bilesen=1).fit(X)
print("Özdeğerler       :", pca.ozdegerler.round(4))        # [1.284  0.0491]
print("PC1              :", pca.bilesenler.round(4))         # ±[0.6779 0.7352]
print("Açıklanan varyans:", pca.aciklanan_varyans_orani.round(3))
print("İlk nokta (PC1)  :", pca.transform(X[:1]).round(3))   # ±0.828

# scikit-learn ile karşılaştır
from sklearn.decomposition import PCA
sk = PCA(n_components=1).fit(X)
print(sk.components_, sk.explained_variance_ratio_)
```

> Özvektörün **işareti** keyfidir (v ve −v aynı yönü tanımlar). Bu yüzden kütüphaneler arasında işaret farkı görebilirsiniz.

**Görselleştirme: İzdüşüm ve geri dönüşüm**

```python
Z = pca.transform(X)
X_geri = pca.inverse_transform(Z)          # 1 boyuttan 2 boyuta geri dön

plt.scatter(X[:, 0], X[:, 1], label="Orijinal")
plt.scatter(X_geri[:, 0], X_geri[:, 1], marker="x", label="PC1 üzerine izdüşüm")
for a, b in zip(X, X_geri):
    plt.plot([a[0], b[0]], [a[1], b[1]], "k:", lw=0.8)    # dik izdüşüm çizgileri
m = pca.ortalama; v = pca.bilesenler[0]
plt.plot(*np.c_[m - 2 * v, m + 2 * v], "r-", label="PC1 yönü")
plt.axis("equal"); plt.legend(); plt.show()
```

Kesikli çizgiler, PCA'nın minimize ettiği **dik izdüşüm hatalarıdır**.

### 3.5 SVD ile PCA

Pratikte (scikit-learn dahil) PCA, kovaryans matrisi yerine merkezlenmiş verinin **tekil değer ayrışımı (SVD)** ile hesaplanır: X = U·S·Vᵀ. V'nin sütunları temel bileşenlerdir ve λᵢ = sᵢ² / (n − 1). Bu yöntem sayısal olarak daha kararlıdır ve d × d kovaryans matrisini hiç oluşturmaz.

```python
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print("Bileşenler:", Vt)
print("Özdeğerler:", S**2 / (len(X) - 1))
```

---

## 4. Bileşen Sayısını Seçmek

### 4.1 Kümülatif açıklanan varyans

En yaygın yöntem: Varyansın belirli bir oranını (ör. %90 veya %95) koruyan en küçük k'yı seçmek.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)            # 1797 görüntü, 8x8 = 64 piksel
pca = PCA().fit(X)
kumulatif = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, 65), kumulatif, "o-", ms=3)
plt.axhline(0.95, color="r", ls="--"); plt.axhline(0.90, color="orange", ls="--")
plt.xlabel("Bileşen sayısı"); plt.ylabel("Kümülatif açıklanan varyans")
plt.show()
print("%90 için:", np.argmax(kumulatif >= 0.90) + 1, "bileşen")
print("%95 için:", np.argmax(kumulatif >= 0.95) + 1, "bileşen")

# scikit-learn kısayolu: oran vererek
PCA(n_components=0.95).fit(X).n_components_
```

### 4.2 Scree grafiği

Özdeğerleri sırayla çizip "dirsek" aramak (Hafta 10'daki dirsek yöntemiyle aynı mantık).

### 4.3 Görev performansı

PCA bir ön işleme adımıysa, k'yı bir **hiperparametre** gibi ele alıp çapraz doğrulama ile seçmek en güvenilir yoldur:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("olcek", StandardScaler()), ("pca", PCA()),
                 ("lr", LogisticRegression(max_iter=2000))])
arama = GridSearchCV(pipe, {"pca__n_components": [5, 10, 20, 30, 40, 64]}, cv=5).fit(X, y)
print(arama.best_params_, arama.best_score_)
```

---

## 5. PCA Hakkında Önemli Notlar

1. **Ölçekleme:** PCA varyansa dayalıdır. Bir öznitelik "metre", diğeri "milimetre" cinsindeyse milimetre olan baskın çıkar. Farklı birimlerdeki özniteliklerde **önce standardize edin**. (Aynı birimdeki piksellerde gerekmeyebilir.)
2. **Doğrusal yöntemdir:** Sadece doğrusal ilişkileri yakalar. Veri bir eğri (ör. "İsviçre rulosu") üzerindeyse başarısız olur → Kernel PCA, t-SNE, UMAP.
3. **Etiketleri kullanmaz:** En büyük varyans yönü, sınıfları en iyi ayıran yön olmak zorunda değildir. (Sınıf ayrımı için denetimli bir alternatif: LDA.)
4. **Yorumlanabilirlik:** Her bileşen tüm orijinal özniteliklerin ağırlıklı toplamıdır. **Yüklemeler (loadings)** — `components_` — her özniteliğin bileşene katkısını gösterir.
5. **Sızıntı:** PCA da bir `fit` işlemidir; yalnızca eğitim verisinde fit edilmeli → Pipeline kullanın.

**Kullanım alanları:** Görselleştirme (2–3 bileşen), gürültü azaltma, sıkıştırma, model eğitimini hızlandırma, çoklu doğrusallığı giderme.

**Örnek — PCA ile görüntü sıkıştırma ve gürültü azaltma:**

```python
fig, ax = plt.subplots(2, 6, figsize=(13, 5))
for j, k in enumerate([2, 5, 10, 20, 40, 64]):
    p = PCA(n_components=k).fit(X)
    X_geri = p.inverse_transform(p.transform(X[:2]))
    for i in range(2):
        ax[i, j].imshow(X_geri[i].reshape(8, 8), cmap="gray_r")
        ax[i, j].axis("off")
    ax[0, j].set_title(f"k={k}\n%{100 * p.explained_variance_ratio_.sum():.0f} varyans")
plt.show()
```

**Özyüzler (eigenfaces) fikri:** Yüz görüntülerine PCA uygulanınca bileşenler "hayalet yüzler" gibi görünür. Her yüz, bu temel yüzlerin ağırlıklı toplamı olarak yazılabilir.

```python
fig, ax = plt.subplots(1, 8, figsize=(14, 2))
p = PCA(n_components=8).fit(X)
for i in range(8):
    ax[i].imshow(p.components_[i].reshape(8, 8), cmap="RdBu"); ax[i].axis("off")
    ax[i].set_title(f"PC{i+1}")
plt.show()
```

---

# BÖLÜM B: Doğrusal Olmayan Görselleştirme: t-SNE ve UMAP

## 6. t-SNE (t-distributed Stochastic Neighbor Embedding)

### 6.1 Amaç

Yüksek boyutlu veriyi 2–3 boyuta, **yerel komşulukları koruyarak** indirgemek: Yüksek boyutta komşu olan noktalar, düşük boyutta da komşu olsun.

PCA **küresel** yapıyı (büyük varyans yönlerini) korur; t-SNE **yerel** yapıya (kimin kimin komşusu olduğuna) odaklanır. Bu yüzden kümeleri görselleştirmede çok başarılıdır.

### 6.2 Çalışma mantığı

**Adım 1 — Yüksek boyutta benzerlikler (Gauss):**

Her i noktası için, j noktasının "komşusu olma olasılığı":

```
                exp(−‖xᵢ − xⱼ‖² / 2σᵢ²)
p(j|i) = ─────────────────────────────────
           Σₖ≠ᵢ exp(−‖xᵢ − xₖ‖² / 2σᵢ²)
```

- Yakın noktalar yüksek, uzak noktalar ~0 olasılık alır.
- σᵢ her nokta için ayrı seçilir: Yoğun bölgelerde dar, seyrek bölgelerde geniş. Bu seçim **perplexity** parametresiyle yapılır (kabaca "her noktanın etkili komşu sayısı", tipik 5–50).
- Simetrik hale getirilir: pᵢⱼ = (p(j|i) + p(i|j)) / 2n

**Adım 2 — Düşük boyutta benzerlikler (Student-t):**

Noktalar 2 boyutta rastgele konumlarla (yᵢ) başlar. Benzerlikler, **ağır kuyruklu** t-dağılımıyla (1 serbestlik dereceli) hesaplanır:

```
               (1 + ‖yᵢ − yⱼ‖²)⁻¹
qᵢⱼ = ──────────────────────────────
        Σₖ≠ₗ (1 + ‖yₖ − yₗ‖²)⁻¹
```

**Neden t-dağılımı?** "Kalabalık sorunu" (crowding problem): Yüksek boyutta bir noktanın çevresinde orta mesafede çok sayıda nokta olabilir; 2 boyutta onlara yer yoktur. Ağır kuyruklu t-dağılımı, orta mesafedeki noktaların düşük boyutta **daha uzağa** yerleşmesine izin verir → kümeler arasında belirgin boşluklar oluşur.

```
 benzerlik
   │╲
   │ ╲  Gauss (yüksek boyut): hızla düşer
   │  ╲___
   │ ╲    ‾‾‾───___ t-dağılımı (düşük boyut): yavaş düşer (ağır kuyruk)
   │   ‾‾──___
   └──────────────── mesafe
```

**Adım 3 — İki dağılımı eşleştir:**

P ve Q arasındaki farkı **KL ıraksaması** ile ölç ve yᵢ konumlarını **gradyan iniş** ile güncelle:

```
KL(P ‖ Q) = Σᵢⱼ pᵢⱼ · log(pᵢⱼ / qᵢⱼ)
```

KL asimetriktir: pᵢⱼ büyükken qᵢⱼ küçükse (yakın komşuları uzağa koymak) **çok büyük ceza**; pᵢⱼ küçükken qᵢⱼ büyükse (uzak noktaları yakına koymak) küçük ceza. Bu yüzden t-SNE yerel yapıyı korumaya öncelik verir.

**Gradyanın sezgisi:** Her nokta çifti arasında bir "yay" gibi kuvvet vardır: pᵢⱼ > qᵢⱼ ise **çekme** (birbirine yaklaştır), pᵢⱼ < qᵢⱼ ise **itme**.

### 6.3 t-SNE grafiklerini okurken dikkat!

| Grafikte gördüğünüz | Anlamlı mı? |
|---|---|
| Hangi noktaların aynı kümede olduğu | ✓ Genelde evet |
| Kümelerin **boyutu** (ne kadar geniş yayıldığı) | ✗ Hayır — t-SNE yoğunlukları eşitlemeye çalışır |
| Kümeler **arası mesafeler** | ✗ Güvenilmez — iki küme arası mesafe gerçek benzerliği yansıtmayabilir |
| Eksenler (x, y değerleri) | ✗ Anlamsız |
| Farklı perplexity/çalıştırmalarda farklı şekiller | Normal — stokastik bir yöntem |

**Diğer sınırlamalar:**
- Yavaş: Klasik hali O(n²); Barnes-Hut yaklaşımıyla O(n log n).
- `transform` yoktur: Yeni bir noktayı mevcut haritaya yerleştiremez. Bu yüzden **model eğitimi için ön işleme olarak kullanılmaz**; sadece görselleştirme içindir.
- Çok yüksek boyutta önce PCA ile ~50 boyuta indirmek hem hızlandırır hem gürültüyü azaltır.

```python
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
for a, perp in zip(ax, [2, 10, 30, 100]):
    Z = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(X)
    s = a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5)
    a.set_title(f"perplexity = {perp}")
plt.colorbar(s, ax=ax); plt.show()
```

## 7. UMAP (Uniform Manifold Approximation and Projection)

### 7.1 Çalışma mantığı (sezgisel)

1. **Yüksek boyutta graf kur:** Her nokta için k en yakın komşuyu bul (`n_neighbors`). Komşuluk ilişkilerini, her noktanın yerel yoğunluğuna göre ölçeklenen **bulanık (ağırlıklı) kenarlarla** bir graf olarak temsil et.
2. **Düşük boyutta benzer bir graf ara:** 2 boyutlu noktaları, benzer bir graf oluşturacak şekilde yerleştir.
3. **Optimize et:** İki graf arasındaki **çapraz entropiyi** stokastik gradyan iniş ile minimize et. Komşular birbirini **çeker**, rastgele örneklenen komşu olmayanlar birbirini **iter** (negatif örnekleme → hızlı).

### 7.2 Parametreler

| Parametre | Etki |
|---|---|
| `n_neighbors` (ör. 15) | Küçük → yerel detaylar; büyük → küresel yapı |
| `min_dist` (ör. 0.1) | Küçük → sıkı, yoğun kümeler; büyük → daha yayvan dağılım |
| `metric` | Öklid, kosinüs vb. |

### 7.3 t-SNE ile karşılaştırma

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Doğrusal mı? | Evet | Hayır | Hayır |
| Korunan yapı | Küresel (varyans) | Yerel | Yerel + küresel yapının bir kısmı |
| Hız | Çok hızlı | Yavaş | Hızlı |
| Yeni veriye uygulama (`transform`) | Var | Yok | Var |
| Deterministik mi? | Evet | Hayır | Hayır (seed ile tekrarlanabilir) |
| Model öncesi ön işleme | Uygun | Uygun değil | Dikkatle kullanılabilir |

```python
# pip install umap-learn
import umap
Z = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5); plt.title("UMAP"); plt.show()
```

---

# BÖLÜM C: Öznitelik Seçimi

## 8. Neden Seçim?

Boyut indirgemenin aksine öznitelik seçimi **orijinal öznitelikleri korur**. Bu sayede:
- Model **yorumlanabilir** kalır ("Karar en çok yaş ve gelire dayanıyor")
- Gelecekte daha **az veri toplamak** yeterli olur (maliyet tasarrufu)
- Gürültülü/alakasız öznitelikler elenerek aşırı öğrenme azalır

Üç ana yaklaşım:

```
FİLTRE:        Öznitelikler ──► [istatistiksel skor] ──► Seçilen ──► Model
               (modelden bağımsız, hızlı)

SARMALAYICI:   Öznitelikler ──► [alt küme seç ⇄ modeli eğit/değerlendir] ──► En iyi alt küme
               (model performansına göre, yavaş)

GÖMÜLÜ:        Öznitelikler ──► [Model eğitimi sırasında seçim] ──► Seçilen
               (Lasso, ağaç önemleri)
```

## 9. Filtre Yöntemleri

Her özniteliği **tek tek**, modelden bağımsız bir istatistikle skorlar.

### 9.1 Varyans eşiği

Neredeyse sabit öznitelikler (ör. %99.9'u aynı değer) bilgi taşımaz.

```python
from sklearn.feature_selection import VarianceThreshold
VarianceThreshold(threshold=0.01).fit_transform(X)
```

MNIST'te kenar pikselleri hep 0'dır → varyansları sıfırdır ve elenirler.

### 9.2 Korelasyon

- **Hedefle korelasyon:** |r| büyük olan öznitelikleri seç (yalnızca doğrusal ilişkileri yakalar).
- **Öznitelikler arası korelasyon:** |r| > 0.9 olan çiftlerden birini at (gereksizlik).

```python
import pandas as pd
def yuksek_korelasyonlulari_bul(df, esik=0.9):
    kor = df.corr().abs()
    ust = kor.where(np.triu(np.ones(kor.shape), k=1).astype(bool))   # üst üçgen
    return [s for s in ust.columns if (ust[s] > esik).any()]
```

### 9.3 İstatistiksel testler

| Test | Öznitelik → Hedef | Ölçtüğü |
|---|---|---|
| ANOVA F-testi (`f_classif`) | Sayısal → Kategorik | Sınıf ortalamaları farklı mı? |
| Ki-kare (`chi2`) | Negatif olmayan/sayım → Kategorik | Bağımsızlık |
| `f_regression` | Sayısal → Sayısal | Doğrusal ilişki |
| Karşılıklı bilgi (`mutual_info_*`) | Her tür | **Doğrusal olmayan** ilişkiler dahil her tür bağımlılık |

**Karşılıklı bilgi (mutual information):** "X'i bilmek Y hakkındaki belirsizliği ne kadar azaltır?" Hafta 7'deki **bilgi kazancı** ile aynı kavramdır: I(X; Y) = H(Y) − H(Y | X). X ve Y bağımsızsa 0'dır.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import load_breast_cancer

veri = load_breast_cancer()
Xb, yb = veri.data, veri.target
secici = SelectKBest(mutual_info_classif, k=10).fit(Xb, yb)
skor = pd.Series(secici.scores_, index=veri.feature_names).sort_values(ascending=False)
print(skor.head(10))
```

**Zayıf yönü:** Her özniteliğe tek başına bakar. Tek başına işe yaramaz ama başka bir öznitelikle birlikte çok faydalı olan öznitelikleri (etkileşimler, ör. XOR) kaçırır; birbirinin kopyası olan öznitelikleri ikisini birden seçebilir.

## 10. Sarmalayıcı (Wrapper) Yöntemler

Öznitelik alt kümelerini **modeli gerçekten eğitip değerlendirerek** seçer.

### 10.1 İleri seçim (Forward selection) — çalışma mantığı

```
Seçilen = {}
Tekrarla:
    Seçilmemiş her f özniteliği için:
        Seçilen ∪ {f} ile modeli CV ile değerlendir
    En çok iyileştiren f'yi Seçilen'e ekle
    (İyileşme yoksa veya hedef sayıya ulaşıldıysa dur)
```

**Örnek akış** (4 öznitelik: A, B, C, D):

```
Tur 1: {A}: 0.70   {B}: 0.82   {C}: 0.75   {D}: 0.60   → B seçildi
Tur 2: {B,A}: 0.85 {B,C}: 0.88 {B,D}: 0.83             → C eklendi
Tur 3: {B,C,A}: 0.88 {B,C,D}: 0.87                     → iyileşme yok, dur
Sonuç: {B, C}
```

**Geriye eleme (Backward elimination):** Tüm özniteliklerle başla, her turda çıkarılınca performansı en az düşüreni at.

**Sıfırdan ileri seçim:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def ileri_secim(model, X, y, isimler, max_oznitelik=10, cv=5):
    secilen, kalan = [], list(range(X.shape[1]))
    en_iyi_skor = -np.inf
    while kalan and len(secilen) < max_oznitelik:
        skorlar = {f: cross_val_score(model, X[:, secilen + [f]], y, cv=cv).mean()
                   for f in kalan}
        f_en_iyi = max(skorlar, key=skorlar.get)
        if skorlar[f_en_iyi] <= en_iyi_skor:
            break                                   # iyileşme yok
        en_iyi_skor = skorlar[f_en_iyi]
        secilen.append(f_en_iyi); kalan.remove(f_en_iyi)
        print(f"+ {isimler[f_en_iyi]:25s} CV doğruluk = {en_iyi_skor:.4f}")
    return secilen

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
secilen = ileri_secim(model, Xb, yb, veri.feature_names)
```

scikit-learn karşılığı: `SequentialFeatureSelector(model, n_features_to_select=..., direction="forward")`

### 10.2 RFE (Recursive Feature Elimination)

1. Tüm özniteliklerle modeli eğit.
2. Model içi öneme göre (katsayı büyüklüğü veya ağaç önemi) **en önemsiz** özniteliği at.
3. İstenen sayıya ulaşana kadar tekrarla.

`RFECV` öznitelik sayısını da çapraz doğrulama ile seçer.

```python
from sklearn.feature_selection import RFECV
rfe = RFECV(LogisticRegression(max_iter=5000), step=1, cv=5)
rfe.fit(StandardScaler().fit_transform(Xb), yb)
print("Seçilen öznitelik sayısı:", rfe.n_features_)
print(veri.feature_names[rfe.support_])
```

**Sarmalayıcıların zayıf yönü:** Hesaplama maliyeti yüksek (d öznitelik için ileri seçim O(d²) model eğitimi) ve doğrulama verisine aşırı uyum riski.

## 11. Gömülü (Embedded) Yöntemler

Seçim, modelin eğitim sürecinin doğal bir parçasıdır:

- **Lasso (L1):** Önemsiz özniteliklerin katsayılarını tam sıfır yapar (Hafta 4).
- **Ağaç tabanlı önem:** Random Forest / Gradient Boosting öznitelik önemleri (Hafta 7–8).

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

secici = SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver="liblinear"))
secici.fit(StandardScaler().fit_transform(Xb), yb)
print("L1 ile seçilen:", veri.feature_names[secici.get_support()])

secici_rf = SelectFromModel(RandomForestClassifier(300, random_state=0), threshold="median")
secici_rf.fit(Xb, yb)
print("RF ile seçilen:", secici_rf.get_support().sum(), "öznitelik")
```

## 12. Karşılaştırma

| | Filtre | Sarmalayıcı | Gömülü |
|---|---|---|---|
| Hız | Çok hızlı | Yavaş | Orta |
| Etkileşimleri görür mü? | Hayır | Evet | Kısmen |
| Modele özgü mü? | Hayır | Evet | Evet |
| Aşırı uyum riski | Düşük | Yüksek | Orta |
| Ne zaman? | Çok fazla öznitelik, ilk eleme | Az öznitelik, en iyi alt küme | Çoğu pratik durum |

> **Sızıntı uyarısı:** Öznitelik seçimi de veriden öğrenir. Tüm veriye göre seçip sonra CV yaparsanız sonuçlar iyimser olur. Seçimi **Pipeline içine** koyun: `make_pipeline(SelectKBest(...), Model())`.

---

## 13. Laboratuvar: MNIST Üzerinde PCA ve t-SNE

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data / 255.0, mnist.target.astype(int)     # pikselleri [0, 1] aralığına
print(X.shape)                                           # (70000, 784)

# Hız için 10.000 örneklik bir alt küme
rng = np.random.default_rng(0)
idx = rng.choice(len(X), 10000, replace=False)
X, y = X[idx], y[idx]
```

> İnternet bağlantısı yoksa veya veri çok büyük geliyorsa `load_digits()` (8×8, 1.797 görüntü) ile aynı adımları uygulayın.

**Görev 1 — Veriyi tanıyın:** İlk 20 rakamı 28×28 olarak çizin. Piksellerin kaçının varyansı sıfır?

```python
fig, ax = plt.subplots(2, 10, figsize=(12, 3))
for a, img, lab in zip(ax.ravel(), X[:20], y[:20]):
    a.imshow(img.reshape(28, 28), cmap="gray_r"); a.set_title(lab); a.axis("off")
plt.show()
print("Sıfır varyanslı piksel:", (X.var(axis=0) == 0).sum())
```

**Görev 2 — Kendi PCA'nız ile scikit-learn'ü karşılaştırın:** `BenimPCA` ile ilk 50 bileşeni hesaplayın; açıklanan varyans oranlarının `PCA(50)` ile aynı olduğunu doğrulayın.

**Görev 3 — Açıklanan varyans:** Kümülatif varyans grafiğini çizin. %90 ve %95 varyans için kaç bileşen gerekiyor? 784 boyuttan ne kadar sıkıştırma demek bu?

**Görev 4 — Yeniden oluşturma:** Bir rakamı k = 10, 30, 50, 100, 200 bileşenle yeniden oluşturup orijinaliyle yan yana gösterin.

```python
pca_full = PCA().fit(X)
fig, ax = plt.subplots(1, 6, figsize=(14, 3))
ax[0].imshow(X[0].reshape(28, 28), cmap="gray_r"); ax[0].set_title("Orijinal")
for a, k in zip(ax[1:], [10, 30, 50, 100, 200]):
    p = PCA(n_components=k).fit(X)
    a.imshow(p.inverse_transform(p.transform(X[:1])).reshape(28, 28), cmap="gray_r")
    a.set_title(f"k = {k}")
for a in ax: a.axis("off")
plt.show()
```

**Görev 5 — İlk bileşenleri görselleştirin:** İlk 10 temel bileşeni 28×28 görüntü olarak çizin. Ne tür desenler yakalıyorlar?

**Görev 6 — 2B görselleştirme: PCA vs. t-SNE (vs. UMAP)**

```python
n = 5000
t = time.time(); Z_pca = PCA(n_components=2).fit_transform(X[:n]); t_pca = time.time() - t

t = time.time()
X50 = PCA(n_components=50).fit_transform(X[:n])          # önce 50 boyuta indir
Z_tsne = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X50)
t_tsne = time.time() - t

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
for a, Z, ad, sure in [(ax[0], Z_pca, "PCA", t_pca), (ax[1], Z_tsne, "t-SNE", t_tsne)]:
    s = a.scatter(Z[:, 0], Z[:, 1], c=y[:n], cmap="tab10", s=3)
    a.set_title(f"{ad} ({sure:.1f} sn)")
    for rakam in range(10):                              # küme merkezlerine etiket yaz
        m = np.median(Z[y[:n] == rakam], axis=0)
        a.text(m[0], m[1], str(rakam), fontsize=18, weight="bold")
plt.colorbar(s, ax=ax); plt.show()
```

Sorular:
- PCA grafiğinde hangi rakamlar iç içe geçmiş? t-SNE'de durum ne?
- t-SNE'de birbirine yakın duran kümeler (ör. 4–9, 3–5–8) gerçekten görsel olarak benzer rakamlar mı?
- t-SNE'yi farklı `random_state` ve `perplexity` değerleriyle tekrar çalıştırın. Neler değişiyor, neler aynı kalıyor?

**Görev 7 — PCA'nın sınıflandırmaya etkisi:** k-NN ve lojistik regresyonu, PCA'sız ve farklı bileşen sayılarıyla (10, 30, 50, 100) eğitin. Doğruluğu ve **eğitim + tahmin süresini** tablo halinde karşılaştırın.

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
sonuclar = []
for k in [None, 10, 30, 50, 100]:
    for ad, mdl in [("k-NN", KNeighborsClassifier(5)),
                    ("LojReg", LogisticRegression(max_iter=1000))]:
        adimlar = ([PCA(n_components=k)] if k else []) + [mdl]
        pipe = make_pipeline(*adimlar)
        t = time.time(); pipe.fit(X_tr, y_tr); dogruluk = pipe.score(X_te, y_te)
        sonuclar.append({"model": ad, "PCA k": k or 784, "doğruluk": round(dogruluk, 4),
                         "süre (sn)": round(time.time() - t, 2)})
import pandas as pd
print(pd.DataFrame(sonuclar).pivot(index="PCA k", columns="model"))
```

Beklenen gözlem: k-NN, PCA ile hem **hızlanır** hem de genellikle **daha doğru** olur (gürültü ve boyutluluk laneti azalır).

**Görev 8 (bonus) — Öznitelik seçimi:** MNIST'te `VarianceThreshold` ve `SelectKBest(f_classif, k=100)` ile seçilen pikselleri 28×28 bir maske olarak çizin. Hangi pikseller en bilgilendirici?

---

## 14. Özet

- **PCA:** Merkezle → kovaryans → özvektörler (yönler) ve özdeğerler (varyans) → en büyük k özvektöre izdüşür. Doğrusal, hızlı, `transform` ve `inverse_transform` destekler.
- **Açıklanan varyans oranı** = λᵢ / Σλ. Kümülatif %90–95 eşiği veya CV ile k seçilir.
- **t-SNE:** Yüksek boyutta Gauss, düşük boyutta t-dağılımı benzerlikleri; KL ıraksamasını minimize eder. Yerel yapıyı korur. Küme boyutları ve kümeler arası mesafeler yorumlanmamalı; yalnızca görselleştirme için.
- **UMAP:** k-NN grafı tabanlı; t-SNE'den hızlı, küresel yapıyı daha iyi korur, yeni veriye uygulanabilir.
- **Öznitelik seçimi:** Filtre (hızlı, tek değişkenli), sarmalayıcı (model temelli, yavaş), gömülü (Lasso, ağaç önemi). Seçim de bir `fit` işlemidir → Pipeline içinde yapılmalı.

## Kendinizi Sınayın

1. Kovaryans matrisi [[4, 0], [0, 1]] olan bir veri için temel bileşenler ve açıklanan varyans oranları nedir?
2. PCA'dan önce neden standardizasyon yapılır? Hangi durumda yapılmasa da olur?
3. Bir t-SNE grafiğinde A kümesi B kümesinden iki kat büyük görünüyor. A'da daha fazla örnek olduğu sonucuna varabilir miyiz?
4. Neden t-SNE'yi bir sınıflandırma modelinden önce ön işleme adımı olarak kullanmayız?
5. 1.000 öznitelikli bir veride ileri seçim, 10 öznitelik seçene kadar kaç model eğitir (5-katlı CV ile)?

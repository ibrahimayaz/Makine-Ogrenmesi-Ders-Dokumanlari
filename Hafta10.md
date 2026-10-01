# Hafta 10 — Kümeleme (Clustering)

## Bu Haftanın Hedefleri

- Denetimsiz öğrenmenin ve kümelemenin amacını anlamak
- **k-means** algoritmasını elle birkaç adım çalıştırmak ve sıfırdan kodlamak
- k-means++ başlatma yöntemini ve k-means'in sınırlamalarını öğrenmek
- **Hiyerarşik kümelemeyi** bağlantı (linkage) türleriyle elle uygulamak ve dendrogram okumak
- **DBSCAN** ile yoğunluk tabanlı kümeleme yapmak; çekirdek, sınır ve gürültü noktalarını ayırt etmek
- Küme sayısını **dirsek yöntemi** ve **siluet skoru** ile seçmek
- Gerçekçi bir müşteri segmentasyonu projesi yapmak

---

## 1. Kümeleme Nedir?

Şimdiye kadarki tüm yöntemler **etiketli** veriyle çalıştı. Kümelemede **etiket yoktur**; amaç, veride doğal olarak bulunan **benzer gruplar** keşfetmektir.

**İyi bir kümeleme:**
- Aynı kümedeki örnekler birbirine **benzer** (küme içi mesafe küçük)
- Farklı kümelerdeki örnekler birbirinden **farklı** (kümeler arası mesafe büyük)

```
        Kümeleme öncesi                 Kümeleme sonrası
   ·  · ·                              ○  ○ ○
  · ·  ·        ·  ·                  ○ ○  ○        △  △
   · ·        · · ·  ·                 ○ ○        △ △ △  △
                ·  ·                                △  △
       · · ·                                 □ □ □
      · · · ·                               □ □ □ □
```

**Uygulama alanları:**
- **Müşteri segmentasyonu:** Pazarlama stratejisini gruplara göre özelleştirmek
- **Belge gruplama:** Haberleri konularına göre otomatik gruplamak
- **Görüntü sıkıştırma:** Benzer renkleri tek bir renkle temsil etmek
- **Anomali tespiti:** Hiçbir kümeye uymayan noktaları bulmak
- **Biyoloji:** Benzer gen ifade profillerini gruplamak

**Zorluk:** "Doğru cevap" yoktur. Değerlendirme, iç ölçütlere (siluet vb.) ve **alan bilgisine** dayanır.

---

## 2. k-means

### 2.1 Algoritma

```
Girdi: Veri X, küme sayısı k

1. Başlatma: k tane merkez (centroid) seç (ör. rastgele k veri noktası)
2. Tekrarla (merkezler değişmeyene kadar):
   a. ATAMA:      Her noktayı en yakın merkezin kümesine ata
   b. GÜNCELLEME: Her merkezi, kümesindeki noktaların ortalamasına taşı
```

**Amaç fonksiyonu:** Küme içi kareler toplamı (WCSS, inertia):

```
         k
WCSS =   Σ    Σ    ‖x − μⱼ‖²
        j=1  x∈Cⱼ
```

Her iki adım da WCSS'yi **azaltır veya değiştirmez**: Atama adımı her noktayı en yakın merkeze bağlar; güncelleme adımında ortalama, kareler toplamını minimize eden noktadır. Bu yüzden algoritma her zaman **yakınsar** — ama mutlaka en iyi (global) çözüme değil.

### 2.2 Elle bir örnek

6 nokta, k = 2:

| Nokta | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| (x, y) | (1, 1) | (2, 1) | (1, 2) | (5, 4) | (6, 5) | (5, 6) |

Kötü bir başlatma seçelim: **c₁ = P1 = (1, 1), c₂ = P2 = (2, 1)**

**İterasyon 1 — Atama** (Öklid mesafesi):

| Nokta | c₁(1,1)'e uzaklık | c₂(2,1)'e uzaklık | Küme |
|---|---|---|---|
| P1 (1,1) | 0 | 1 | 1 |
| P2 (2,1) | 1 | 0 | 2 |
| P3 (1,2) | 1 | √2 = 1.41 | 1 |
| P4 (5,4) | 5.00 | 4.24 | 2 |
| P5 (6,5) | 6.40 | 5.66 | 2 |
| P6 (5,6) | 6.40 | 5.83 | 2 |

**İterasyon 1 — Güncelleme:**
- c₁ = ortalama(P1, P3) = ((1+1)/2, (1+2)/2) = **(1, 1.5)**
- c₂ = ortalama(P2, P4, P5, P6) = ((2+5+6+5)/4, (1+4+5+6)/4) = **(4.5, 4)**

**İterasyon 2 — Atama:**

| Nokta | c₁(1,1.5)'e | c₂(4.5,4)'e | Küme |
|---|---|---|---|
| P1 | 0.50 | 4.61 | 1 |
| P2 | 1.12 | 3.91 | **1** ← değişti! |
| P3 | 0.50 | 4.03 | 1 |
| P4 | 4.72 | 0.50 | 2 |
| P5 | 6.10 | 1.80 | 2 |
| P6 | 6.02 | 2.06 | 2 |

**İterasyon 2 — Güncelleme:**
- c₁ = ortalama(P1, P2, P3) = **(1.33, 1.33)**
- c₂ = ortalama(P4, P5, P6) = **(5.33, 5.00)**

**İterasyon 3 — Atama:** Hiçbir nokta küme değiştirmez → **yakınsadı.**

```
 y
 6 │              ○P6
 5 │          ✚c₂  ○P5
 4 │              ○P4
 3 │
 2 │ ●P3
 1 │ ●P1 ●P2
   │   ✚c₁
   └────────────────────── x
     1   2   3   4   5   6
```

**Son WCSS:**
- Küme 1: (0.33² + 0.33²) + (0.67² + 0.33²) + (0.33² + 0.67²) = 0.22 + 0.56 + 0.56 = 1.33
- Küme 2: (0.33² + 1²) + (0.67² + 0²) + (0.33² + 1²) = 1.11 + 0.44 + 1.11 = 2.67
- **WCSS = 4.00**

### 2.3 Sıfırdan k-means

```python
import numpy as np
import matplotlib.pyplot as plt

class BenimKMeans:
    def __init__(self, k=3, max_iter=100, baslatma="kmeans++", seed=0):
        self.k, self.max_iter, self.baslatma, self.seed = k, max_iter, baslatma, seed

    def _baslat(self, X, rng):
        if self.baslatma == "rastgele":
            return X[rng.choice(len(X), self.k, replace=False)]
        # k-means++ (Bölüm 2.5)
        merkezler = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None, :] - np.array(merkezler)[None]) ** 2).sum(-1), axis=1)
            olasilik = d2 / d2.sum()
            merkezler.append(X[rng.choice(len(X), p=olasilik)])
        return np.array(merkezler)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.merkezler = self._baslat(X, rng)
        self.gecmis = [self.merkezler.copy()]
        for it in range(self.max_iter):
            # ATAMA: her nokta - her merkez mesafesi, (n, k) matris
            d = ((X[:, None, :] - self.merkezler[None, :, :]) ** 2).sum(axis=2)
            self.etiketler = d.argmin(axis=1)
            # GÜNCELLEME
            yeni = np.array([X[self.etiketler == j].mean(axis=0)
                             if np.any(self.etiketler == j) else self.merkezler[j]
                             for j in range(self.k)])
            self.gecmis.append(yeni.copy())
            if np.allclose(yeni, self.merkezler):
                break
            self.merkezler = yeni
        self.n_iter = it + 1
        self.inertia = d.min(axis=1).sum()
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.merkezler[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

# Elle yaptığımız örnek
X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [5, 6]], dtype=float)
km = BenimKMeans(k=2)
# Elle seçtiğimiz başlatmayı (P1, P2) kullanmak için _baslat'ı geçersiz kılalım:
km._baslat = lambda X, rng: X[[0, 1]].copy()
km.fit(X)
print("Merkezler:\n", km.merkezler.round(2))
print("Etiketler:", km.etiketler, " İterasyon:", km.n_iter, " WCSS:", round(km.inertia, 2))
```

**Merkezlerin hareketini görselleştirelim:**

```python
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=3)
km = BenimKMeans(k=4, baslatma="rastgele", seed=5).fit(X)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, adim in zip(ax, [0, 1, 2, len(km.gecmis) - 1]):
    M = km.gecmis[adim]
    et = ((X[:, None] - M[None]) ** 2).sum(-1).argmin(1)
    a.scatter(X[:, 0], X[:, 1], c=et, s=10, cmap="tab10")
    a.scatter(M[:, 0], M[:, 1], c="k", marker="X", s=200)
    a.set_title(f"Adım {adim}")
plt.show()
```

### 2.4 Başlatma sorunu: Yerel minimum

k-means sonucu **başlangıç merkezlerine bağlıdır**. Kötü bir başlatma, iki merkezin aynı kümeye düşmesine ve iki gerçek kümenin birleşmesine yol açabilir:

```
 Gerçek kümeler:   (A)    (B)         (C)
 Kötü başlatma:    ✚ ✚                         ← iki merkez A'da
 Sonuç:           A ikiye bölündü, B ve C birleşti  (yerel minimum)
```

**Çözümler:**
1. **Birden fazla başlatma:** Algoritmayı farklı başlangıçlarla n kez çalıştır, en düşük WCSS'yi seç (scikit-learn: `n_init`).
2. **k-means++** akıllı başlatma.

### 2.5 k-means++

Fikir: Merkezleri birbirinden **uzak** seç.

```
1. İlk merkezi rastgele seç.
2. Her x noktası için D(x) = en yakın mevcut merkeze uzaklık.
3. Bir sonraki merkezi, D(x)² ile orantılı olasılıkla seç
   (uzak noktaların seçilme şansı daha yüksek).
4. k merkez seçilene kadar 2–3'ü tekrarla.
```

Örnek: Mevcut merkez (0, 0); aday noktalar A (1, 0), B (5, 0), C (10, 0):
- D² = 1, 25, 100 → toplam 126
- Seçilme olasılıkları: A %0.8, B %19.8, **C %79.4**

scikit-learn'ün `KMeans` sınıfı varsayılan olarak k-means++ kullanır.

### 2.6 k-means'in varsayımları ve sınırlamaları

k-means, kümelerin **küresel (yuvarlak), benzer büyüklükte ve benzer yoğunlukta** olduğunu varsayar. Çünkü her nokta sadece en yakın merkeze atanır; bu da kümeler arasında **doğrusal (Voronoi) sınırlar** oluşturur.

| Durum | k-means'in davranışı |
|---|---|
| Uzun/eliptik kümeler | Kümeyi yanlış bölebilir |
| Ay/halka şeklinde kümeler | Başarısız |
| Çok farklı büyüklükte kümeler | Büyük kümeyi bölüp küçükleri birleştirebilir |
| Aykırı değerler | Merkezleri kendine çeker (ortalama kullanıldığı için) |
| Ölçeklenmemiş öznitelikler | Büyük ölçekli öznitelik baskın olur → **ölçekleme şart** |
| k bilinmiyor | Önceden belirtilmesi gerekir (Bölüm 5) |

**Varyantlar:** k-medoids (merkez gerçek bir veri noktası; aykırılara dayanıklı), Mini-Batch k-means (büyük veri için hızlı), Gaussian Mixture Models (eliptik kümeler, yumuşak atama).

---

## 3. Hiyerarşik Kümeleme

### 3.1 Fikir

Önceden k belirlemek yerine kümelerin **iç içe geçmiş bir hiyerarşisini** (ağaç) oluşturur. Bu ağaca **dendrogram** denir.

**Birleştirici (agglomerative, aşağıdan yukarı)** yaklaşım:

```
1. Her nokta kendi başına bir küme olsun (n küme).
2. Tekrarla (tek küme kalana kadar):
   a. Birbirine en yakın iki kümeyi bul
   b. Onları birleştir
   c. Kümeler arası mesafeleri güncelle
```

### 3.2 Kümeler arası mesafe: Bağlantı (linkage) türleri

İki **nokta** arasındaki mesafe belli; peki iki **küme** arasındaki mesafe?

| Bağlantı | Tanım | Özellik |
|---|---|---|
| **Tekli (single)** | En yakın iki nokta arası mesafe: min d(a, b) | Uzun, zincir şeklinde kümeler; gürültüye hassas |
| **Tam (complete)** | En uzak iki nokta arası mesafe: max d(a, b) | Kompakt, benzer çaplı kümeler |
| **Ortalama (average)** | Tüm çiftlerin ortalama mesafesi | İkisi arası denge |
| **Ward** | Birleşince WCSS'deki artış | k-means'e benzer küresel kümeler; scikit-learn varsayılanı |

```
  Küme A          Küme B
  ● ●               ○ ○
   ●  ●- - - - - - ○    ○      single: en yakın çift (kısa kesikli)
  ●  ●━━━━━━━━━━━━━━━━━━━━━○   complete: en uzak çift (kalın)
```

### 3.3 Elle bir örnek

Tek boyutlu 5 nokta: **A = 1, B = 2, C = 4, D = 7, E = 8**

**Mesafe matrisi:**

| | A | B | C | D | E |
|---|---|---|---|---|---|
| A | 0 | 1 | 3 | 6 | 7 |
| B | | 0 | 2 | 5 | 6 |
| C | | | 0 | 3 | 4 |
| D | | | | 0 | 1 |
| E | | | | | 0 |

**Tekli bağlantı (single linkage):**

| Adım | En yakın kümeler | Mesafe | Kümeler |
|---|---|---|---|
| 1 | A – B | 1 | {A,B}, C, D, E |
| 2 | D – E | 1 | {A,B}, C, {D,E} |
| 3 | {A,B} – C: min(3, 2) = 2 | 2 | {A,B,C}, {D,E} |
| 4 | {A,B,C} – {D,E}: min(C–D) = 3 | 3 | {A,B,C,D,E} |

**Tam bağlantı (complete linkage):**

| Adım | Aday mesafeler | Birleşme | Mesafe |
|---|---|---|---|
| 1 | | A – B | 1 |
| 2 | | D – E | 1 |
| 3 | {A,B}–C: max(3,2) = **3**; {A,B}–{D,E}: max = 7; C–{D,E}: max(3,4) = 4 | {A,B} – C | 3 |
| 4 | {A,B,C}–{D,E}: max(8−1) = 7 | Hepsi | 7 |

**Dendrogram (tam bağlantı):**

```
 mesafe
   7 │        ┌───────────┐
     │        │           │
   3 │    ┌───┴──┐        │      ← burada kesersek: 2 küme
     │    │      │        │
   1 │  ┌─┴─┐    │     ┌──┴─┐
     │  │   │    │     │    │
   0 │  A   B    C     D    E
```

**Dendrogram nasıl okunur?**
- Yatay çizgilerin yüksekliği = birleşme mesafesi
- Dendrogramı **yatay bir çizgiyle kesmek** küme sayısını belirler. Örneğin yükseklik 5'te kesersek 2 küme: {A, B, C} ve {D, E}
- Uzun dikey çizgiler (büyük mesafe sıçramaları) "doğal" küme ayrımlarına işaret eder. Burada 3 → 7 sıçraması 2 kümeyi öneriyor.

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

noktalar = np.array([[1], [2], [4], [7], [8]])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, yontem in zip(ax, ["single", "complete"]):
    Z = linkage(noktalar, method=yontem)
    print(yontem, "\n", Z)       # her satır: [küme1, küme2, mesafe, eleman sayısı]
    dendrogram(Z, labels=list("ABCDE"), ax=a)
    a.set_title(f"{yontem} bağlantı")
plt.show()

print(fcluster(linkage(noktalar, "complete"), t=2, criterion="maxclust"))   # [1 1 1 2 2]
```

### 3.4 Sıfırdan hiyerarşik kümeleme (basit, O(n³))

```python
def hiyerarsik(X, baglanti="single", hedef_k=1):
    kumeler = [[i] for i in range(len(X))]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))       # nokta-nokta mesafeleri
    birlesim_gecmisi = []

    def kume_mesafesi(a, b):
        alt = D[np.ix_(a, b)]
        return {"single": alt.min(), "complete": alt.max(), "average": alt.mean()}[baglanti]

    while len(kumeler) > hedef_k:
        en_iyi = (np.inf, None, None)
        for i in range(len(kumeler)):
            for j in range(i + 1, len(kumeler)):
                d = kume_mesafesi(kumeler[i], kumeler[j])
                if d < en_iyi[0]:
                    en_iyi = (d, i, j)
        d, i, j = en_iyi
        birlesim_gecmisi.append((kumeler[i], kumeler[j], d))
        kumeler[i] = kumeler[i] + kumeler[j]
        del kumeler[j]
    return kumeler, birlesim_gecmisi

kumeler, gecmis = hiyerarsik(noktalar.astype(float), "complete", hedef_k=1)
for a, b, d in gecmis:
    print(f"{[ 'ABCDE'[i] for i in a ]} + {[ 'ABCDE'[i] for i in b ]}  @ {d}")
```

### 3.5 Güçlü ve zayıf yönler

| Güçlü | Zayıf |
|---|---|
| k'yı önceden belirtmek gerekmez | O(n²) bellek, O(n²)–O(n³) zaman → büyük veride yavaş |
| Dendrogram veri yapısını görsel olarak açıklar | Bir birleştirme geri alınamaz (açgözlü) |
| Farklı bağlantılarla farklı küme şekilleri | Bağlantı seçimine duyarlı |

---

## 4. DBSCAN (Yoğunluk Tabanlı Kümeleme)

### 4.1 Fikir

Küme = **yoğun** bölge. Kümeler, seyrek bölgelerle birbirinden ayrılır. Hiçbir yoğun bölgeye ait olmayan noktalar **gürültüdür**.

İki parametre:
- **ε (eps):** Komşuluk yarıçapı
- **minPts (min_samples):** Bir noktanın "çekirdek" olması için ε-komşuluğunda bulunması gereken minimum nokta sayısı (noktanın kendisi dahil)

### 4.2 Nokta türleri

```
              ε
          ╭───────╮
          │ ●  ●  │
          │   ★   │ ●      ★ Çekirdek: ε içinde ≥ minPts nokta
          │ ●   ● │  ◐     ◐ Sınır: çekirdek değil ama bir çekirdeğin ε-komşuluğunda
          ╰───────╯        ✕ Gürültü: ne çekirdek ne sınır
                         ✕
```

- **Çekirdek nokta (core):** ε yarıçapında en az minPts nokta var.
- **Sınır nokta (border):** Kendisi çekirdek değil ama bir çekirdek noktanın komşusu.
- **Gürültü (noise):** Hiçbir çekirdeğin komşusu değil → etiket −1.

### 4.3 Algoritma

```
Her ziyaret edilmemiş p noktası için:
    p'yi ziyaret edildi olarak işaretle
    N = p'nin ε-komşuları
    eğer |N| < minPts: p'yi (şimdilik) gürültü olarak işaretle
    değilse:
        yeni küme C başlat, p'yi C'ye ekle
        N'deki her q için:                      ← küme genişletme
            q gürültü olarak işaretliyse → C'ye ekle (sınır nokta)
            q ziyaret edilmediyse:
                q'yu ziyaret et, C'ye ekle
                q'nun komşuları ≥ minPts ise (q çekirdekse) → komşularını N'ye ekle
```

Kümeler, çekirdek noktalardan çekirdek noktalara "yayılarak" büyür. Bu sayede **herhangi bir şekildeki** kümeler bulunabilir.

### 4.4 Elle bir örnek

Tek boyutlu noktalar: **1, 2, 3, 8, 9, 10, 20**. ε = 1.5, minPts = 3.

| Nokta | ε-komşuluğu (\|x − p\| ≤ 1.5) | Sayı | Tür |
|---|---|---|---|
| 1 | {1, 2} | 2 | Sınır (2'nin komşusu) |
| 2 | {1, 2, 3} | 3 | **Çekirdek** |
| 3 | {2, 3} | 2 | Sınır |
| 8 | {8, 9} | 2 | Sınır |
| 9 | {8, 9, 10} | 3 | **Çekirdek** |
| 10 | {9, 10} | 2 | Sınır |
| 20 | {20} | 1 | **Gürültü** |

**Sonuç:** Küme 0 = {1, 2, 3}, Küme 1 = {8, 9, 10}, Gürültü = {20}

k'yı hiç belirtmedik; algoritma 2 küme ve 1 aykırı nokta buldu.

```python
from sklearn.cluster import DBSCAN
x = np.array([1, 2, 3, 8, 9, 10, 20]).reshape(-1, 1)
db = DBSCAN(eps=1.5, min_samples=3).fit(x)
print("Etiketler:", db.labels_)                     # [ 0  0  0  1  1  1 -1]
print("Çekirdek indeksler:", db.core_sample_indices_)  # [1 4]
```

### 4.5 Sıfırdan DBSCAN

```python
def benim_dbscan(X, eps, min_pts):
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    komsular = [np.where(D[i] <= eps)[0] for i in range(n)]
    cekirdek = np.array([len(k) >= min_pts for k in komsular])

    etiket = np.full(n, -1)          # -1: gürültü / atanmamış
    kume = 0
    for i in range(n):
        if etiket[i] != -1 or not cekirdek[i]:
            continue
        # i bir çekirdek ve henüz kümesi yok: yeni küme başlat, genişlet (BFS)
        kuyruk = [i]
        etiket[i] = kume
        while kuyruk:
            p = kuyruk.pop()
            if not cekirdek[p]:
                continue                 # sınır noktalar kümeyi genişletmez
            for q in komsular[p]:
                if etiket[q] == -1:
                    etiket[q] = kume
                    kuyruk.append(q)
        kume += 1
    return etiket, cekirdek

print(benim_dbscan(x.astype(float), 1.5, 3))
```

### 4.6 Parametre seçimi: k-mesafe grafiği

- **minPts:** Kural olarak ≥ boyut + 1; genelde 2·boyut veya 4–10 arası.
- **ε:** Her noktanın **k'ıncı en yakın komşusuna** (k = minPts − 1) olan mesafesini hesapla, küçükten büyüğe sırala ve çiz. Eğrinin **dirsek** yaptığı yer iyi bir ε adayıdır: Yoğun bölgelerdeki noktalar dirseğin altında, gürültü noktaları üstünde kalır.

```python
from sklearn.neighbors import NearestNeighbors

def k_mesafe_grafigi(X, min_pts):
    nn = NearestNeighbors(n_neighbors=min_pts).fit(X)
    d, _ = nn.kneighbors(X)               # ilk komşu noktanın kendisi (mesafe 0)
    plt.plot(np.sort(d[:, -1]))
    plt.xlabel("Noktalar (sıralı)"); plt.ylabel(f"{min_pts - 1}. komşuya mesafe")
    plt.title("k-mesafe grafiği: dirsek ≈ ε"); plt.show()
```

### 4.7 Güçlü ve zayıf yönler

| Güçlü | Zayıf |
|---|---|
| Keyfi şekilli kümeleri bulur (ay, halka) | Farklı yoğunluktaki kümelerde tek ε yetmez |
| k'yı belirtmek gerekmez | ε ve minPts'ye duyarlı |
| Gürültüyü/aykırı değerleri otomatik ayırır | Yüksek boyutta mesafeler anlamını yitirir |

(Farklı yoğunluklar için **HDBSCAN** — scikit-learn ≥ 1.3'te `HDBSCAN` — daha esnek bir alternatiftir.)

---

## 5. Küme Sayısını Seçmek ve Kümelemeyi Değerlendirmek

### 5.1 Dirsek yöntemi (Elbow method)

k = 1, 2, 3, ... için k-means çalıştırıp WCSS'yi çiz. k arttıkça WCSS **hep azalır** (k = n'de sıfır olur!). Aranan, eğrinin **"dirsek" yaptığı**, yani ek bir kümenin artık çok az fayda sağladığı noktadır.

```
 WCSS
   │●
   │ ╲
   │  ●
   │   ╲
   │    ●  ← dirsek (k = 3)
   │     ╲___●___●___●___●
   └──────────────────────── k
     1  2  3  4  5  6  7
```

Zayıf yönü: Dirsek her zaman net değildir; yorum öznel olabilir.

### 5.2 Siluet skoru (Silhouette score)

Her nokta i için:

- **a(i):** i'nin **kendi kümesindeki** diğer noktalara ortalama uzaklığı (ne kadar "uyumlu"?)
- **b(i):** i'nin **en yakın komşu kümedeki** noktalara ortalama uzaklığı (ne kadar "ayrık"?)

```
            b(i) − a(i)
s(i) = ─────────────────────        −1 ≤ s(i) ≤ 1
         max(a(i), b(i))
```

| s(i) | Yorum |
|---|---|
| ≈ 1 | Kendi kümesine çok iyi uymuş, diğerlerinden uzak |
| ≈ 0 | İki kümenin sınırında |
| < 0 | Muhtemelen yanlış kümeye atanmış |

**Elle hesaplama:** k-means örneğimizdeki (Bölüm 2.2) son kümeleme için P1 = (1, 1):

- a(P1) = ortalama[d(P1, P2), d(P1, P3)] = (1 + 1) / 2 = **1.00**
- b(P1) = ortalama[d(P1, P4), d(P1, P5), d(P1, P6)] = (5.00 + 6.40 + 6.40) / 3 = **5.93**
- s(P1) = (5.93 − 1.00) / 5.93 = **0.83** → çok iyi kümelenmiş

Tüm noktaların ortalaması **siluet skorudur**. Farklı k değerleri için hesaplanıp en yüksek olan seçilir.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

X6 = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [5, 6]], dtype=float)
etiket = np.array([0, 0, 0, 1, 1, 1])
print("Nokta siluetleri:", silhouette_samples(X6, etiket).round(3))
print("Ortalama siluet :", silhouette_score(X6, etiket).round(3))
```

**Sıfırdan siluet:**

```python
def siluet(X, etiket):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        ayni = (etiket == etiket[i]); ayni[i] = False
        if ayni.sum() == 0:
            continue                          # tek elemanlı küme: s = 0
        a = D[i, ayni].mean()
        b = min(D[i, etiket == k].mean() for k in np.unique(etiket) if k != etiket[i])
        s[i] = (b - a) / max(a, b)
    return s
```

### 5.3 Diğer ölçütler

| Ölçüt | Etiket gerekir mi? | Yorum |
|---|---|---|
| Siluet | Hayır | Yüksek iyi (−1 … 1) |
| Davies-Bouldin | Hayır | **Düşük** iyi |
| Calinski-Harabasz | Hayır | Yüksek iyi |
| Adjusted Rand Index (ARI) | **Evet** (gerçek gruplar biliniyorsa) | 1 = mükemmel eşleşme, 0 = rastgele |

### 5.4 Yöntemlerin karşılaştırması

```python
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

veriler = {
    "Küresel": make_blobs(400, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=170)[0],
    "Ay": make_moons(400, noise=0.06, random_state=0)[0],
    "Halka": make_circles(400, noise=0.05, factor=0.5, random_state=0)[0],
}
yontemler = {
    "k-means": lambda k: KMeans(k, n_init=10, random_state=0),
    "Hiyerarşik (ward)": lambda k: AgglomerativeClustering(k, linkage="ward"),
    "Hiyerarşik (single)": lambda k: AgglomerativeClustering(k, linkage="single"),
    "DBSCAN": lambda k: DBSCAN(eps=0.3, min_samples=5),
}
fig, ax = plt.subplots(3, 4, figsize=(18, 12))
for i, (vad, X) in enumerate(veriler.items()):
    X = StandardScaler().fit_transform(X)
    k = 3 if vad == "Küresel" else 2
    for j, (yad, yap) in enumerate(yontemler.items()):
        et = yap(k).fit_predict(X)
        ax[i, j].scatter(X[:, 0], X[:, 1], c=et, s=8, cmap="tab10")
        ax[i, j].set_title(f"{vad} – {yad}")
plt.tight_layout(); plt.show()
```

Beklenen: k-means ve Ward küresel kümelerde iyi, ay/halkada başarısız; DBSCAN ve single linkage ay/halkada başarılı.

---

## 6. Laboratuvar: Müşteri Segmentasyonu

Bir alışveriş merkezinin müşteri verisini analiz edip pazarlama ekibi için anlamlı segmentler oluşturacağız.

### 6.1 Veri

Kaggle'daki "Mall Customer Segmentation" verisini (`Mall_Customers.csv`) indirebilirsiniz. İnternet erişimi yoksa benzer yapıda sentetik veri üretin:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(42)
def grup(n, gelir, harcama, yas):
    return pd.DataFrame({
        "Yas": rng.normal(yas[0], yas[1], n).clip(18, 70).round(),
        "YillikGelir": rng.normal(gelir[0], gelir[1], n).clip(15, 140).round(),   # bin $
        "HarcamaSkoru": rng.normal(harcama[0], harcama[1], n).clip(1, 99).round(),
    })
df = pd.concat([
    grup(40, (25, 6), (20, 8), (45, 10)),    # düşük gelir, düşük harcama
    grup(40, (25, 6), (80, 8), (25, 5)),     # düşük gelir, yüksek harcama
    grup(80, (55, 8), (50, 7), (42, 12)),    # orta gelir, orta harcama
    grup(40, (88, 12), (82, 8), (32, 4)),    # yüksek gelir, yüksek harcama
    grup(40, (88, 12), (17, 8), (41, 10)),   # yüksek gelir, düşük harcama
], ignore_index=True)
df.insert(0, "MusteriID", range(1, len(df) + 1))
df.head()
```

### 6.2 Görevler

**Görev 1 — EDA:** Öznitelik dağılımlarını ve `sns.pairplot(df.drop(columns="MusteriID"))` grafiğini inceleyin. Gözle kaç grup görüyorsunuz?

**Görev 2 — Ölçekleme:**

```python
from sklearn.preprocessing import StandardScaler
oznitelikler = ["YillikGelir", "HarcamaSkoru"]          # önce 2 boyutla başlayalım
X = StandardScaler().fit_transform(df[oznitelikler])
```

**Görev 3 — k seçimi (dirsek + siluet):**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ks = range(2, 11)
wcss, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, wcss, "o-"); ax[0].set_title("Dirsek yöntemi"); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "s-"); ax[1].set_title("Siluet skoru"); ax[1].set_xlabel("k")
plt.show()
```

**Görev 4 — Kendi k-means'inizle karşılaştırın:** `BenimKMeans` ile seçtiğiniz k için kümeleme yapın; etiketleri `adjusted_rand_score` ile scikit-learn sonucu ile karşılaştırın (etiket numaraları farklı olabilir, ARI bundan etkilenmez).

**Görev 5 — Segmentleri görselleştirin ve profilleyin:**

```python
k = 5
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
df["Segment"] = km.labels_

sns.scatterplot(data=df, x="YillikGelir", y="HarcamaSkoru", hue="Segment", palette="tab10")
plt.title("Müşteri segmentleri"); plt.show()

profil = df.groupby("Segment")[["Yas", "YillikGelir", "HarcamaSkoru"]].mean().round(1)
profil["MusteriSayisi"] = df["Segment"].value_counts().sort_index()
print(profil)
```

**Görev 6 — Segmentlere isim verin ve pazarlama önerisi yazın.** Örnek:

| Segment | Profil | Örnek isim | Öneri |
|---|---|---|---|
| ? | Yüksek gelir, yüksek harcama | "Premium müşteriler" | Sadakat programı, özel etkinlikler |
| ? | Yüksek gelir, düşük harcama | "Temkinli zenginler" | Kalite vurgusu, kişisel danışmanlık |
| ? | Düşük gelir, yüksek harcama | "Hevesli gençler" | Taksit seçenekleri, trend ürünler |
| ... | | | |

**Görev 7 — Hiyerarşik kümeleme ve DBSCAN ile karşılaştırma:**
- Ward bağlantısıyla dendrogram çizin; dendrogram hangi k'yı öneriyor?
- DBSCAN için k-mesafe grafiğiyle ε seçin. Kaç küme ve kaç gürültü noktası buldu? Gürültü noktaları kimler?

```python
from scipy.cluster.hierarchy import linkage, dendrogram
plt.figure(figsize=(14, 5))
dendrogram(linkage(X, "ward"), truncate_mode="lastp", p=30)
plt.title("Ward dendrogramı"); plt.show()
```

**Görev 8 — Üç öznitelik (Yaş dahil):** Kümelemeyi 3 öznitelikle tekrarlayın. Segmentler nasıl değişti? (Görselleştirme için Hafta 11'de PCA kullanacağız.)

---

## 7. Özet

| | k-means | Hiyerarşik | DBSCAN |
|---|---|---|---|
| k gerekli mi? | Evet | Hayır (dendrogramdan seçilir) | Hayır |
| Küme şekli | Küresel, benzer boyut | Bağlantıya bağlı | Keyfi şekil |
| Gürültü | Her noktayı bir kümeye atar | Her noktayı atar | Gürültüyü ayırır |
| Ölçeklenebilirlik | Çok iyi (O(n·k·iter)) | Zayıf (O(n²)) | İyi (indeksle O(n log n)) |
| Ana parametreler | k, başlatma | Bağlantı türü, kesme yüksekliği | ε, minPts |

- **k-means:** Ata → güncelle → tekrarla. WCSS'yi azaltır, yerel minimuma takılabilir → k-means++ ve `n_init`.
- **Hiyerarşik:** En yakın kümeleri birleştir; tekli/tam/ortalama/Ward bağlantı; dendrogram kesilerek k seçilir.
- **DBSCAN:** Çekirdek noktalardan yayılan yoğun bölgeler; gürültüyü ayırır.
- **k seçimi:** Dirsek (WCSS) ve siluet skoru; ama en önemlisi **segmentlerin iş açısından anlamlı olması**.
- Mesafe tabanlı tüm yöntemlerde **ölçekleme şarttır**.

## Kendinizi Sınayın

1. Bölüm 2.2'deki örneği başlangıç merkezleri c₁ = P1, c₂ = P4 ile tekrarlayın. Kaç iterasyonda yakınsadı?
2. Neden k-means'te WCSS'yi en küçük yapan k değerini seçemeyiz?
3. Bölüm 3.3'teki noktalar için **ortalama bağlantı** ile birleşme sırasını ve mesafeleri bulun.
4. DBSCAN örneğinde ε = 3 seçilseydi sonuç ne olurdu?
5. Bölüm 2.2'deki son kümelemede P4 = (5, 4) için siluet değerini hesaplayın.

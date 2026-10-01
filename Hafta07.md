# Hafta 7 — Karar Ağaçları

## Bu Haftanın Hedefleri

- Karar ağacının nasıl tahmin yaptığını ve nasıl inşa edildiğini anlamak
- Entropi, bilgi kazancı ve Gini safsızlığını **elle hesaplamak**
- Sayısal özniteliklerde en iyi bölme noktasının nasıl bulunduğunu görmek
- Karar ağacını sıfırdan kodlamak
- Aşırı öğrenmeyi önlemek için ön budama ve son budama yöntemlerini uygulamak
- Ağaçları görselleştirmek ve yorumlamak

---

## 1. Karar Ağacı Nedir?

Karar ağacı, bir dizi **evet/hayır sorusu** sorarak karar veren bir modeldir. Günlük hayatta hepimiz böyle düşünürüz:

```
                     [Hava durumu?]
                   ╱       │        ╲
            Güneşli     Bulutlu     Yağmurlu
               │           │           │
          [Nem?]        OYNA ✓     [Rüzgar?]
          ╱    ╲                    ╱     ╲
      Yüksek  Normal              Var     Yok
        │       │                  │       │
   OYNAMA ✗   OYNA ✓          OYNAMA ✗   OYNA ✓
```

Terimler:
- **Kök düğüm (root):** En üstteki ilk soru
- **İç düğüm (internal node):** Bir özniteliğe göre soru soran düğüm
- **Dal (branch):** Sorunun bir cevabı
- **Yaprak (leaf):** Son karar (sınıf veya sayısal değer)
- **Derinlik (depth):** Kökten en uzak yaprağa kadar olan soru sayısı

**Tahmin nasıl yapılır?** Yeni bir örnek kökten başlar, her düğümde sorunun cevabına göre ilgili dala ilerler ve bir yaprağa ulaşınca o yaprağın kararını alır.

Örnek: (Hava = Güneşli, Nem = Normal, Rüzgar = Var) → Güneşli dalı → Nem normal → **OYNA**

**Asıl soru:** Bu ağacı veriden nasıl öğreniriz? Hangi soruyu önce sormalıyız?

---

## 2. Temel Fikir: Saflık

İyi bir soru, veriyi **mümkün olduğunca saf (homojen)** gruplara ayırır.

```
Kötü bölme:                          İyi bölme:
   [● ● ● ○ ○ ○]                        [● ● ● ○ ○ ○]
     ╱       ╲                            ╱       ╲
 [● ● ○]   [● ○ ○]                    [● ● ●]   [○ ○ ○]
 hâlâ karışık                         tamamen saf!
```

Saflığı ölçmek için iki yaygın ölçü vardır: **Entropi** ve **Gini safsızlığı**.

### 2.1 Entropi

Bilgi teorisinden gelir; bir gruptaki **belirsizliği** ölçer:

```
H(S) = − Σ pₖ · log₂(pₖ)
```

pₖ: k sınıfının gruptaki oranı. (0·log 0 = 0 kabul edilir.)

İki sınıf için:

| Grup | p(●) | p(○) | Entropi | Yorum |
|---|---|---|---|---|
| [● ● ● ●] | 1.0 | 0.0 | 0 | Tamamen saf, belirsizlik yok |
| [● ● ● ○] | 0.75 | 0.25 | 0.811 | |
| [● ● ○ ○] | 0.5 | 0.5 | **1.0** | En belirsiz durum |

Örnek hesap: [● ● ● ○] → −(0.75·log₂0.75 + 0.25·log₂0.25) = −(0.75·(−0.415) + 0.25·(−2)) = 0.311 + 0.5 = **0.811**

### 2.2 Gini safsızlığı

```
Gini(S) = 1 − Σ pₖ²
```

Yorumu: Gruptan rastgele bir örnek çekip yine gruptaki dağılıma göre rastgele bir etiket verirsek **yanlış etiketleme olasılığı**.

| Grup | Gini |
|---|---|
| [● ● ● ●] | 1 − (1² + 0²) = 0 |
| [● ● ● ○] | 1 − (0.75² + 0.25²) = 0.375 |
| [● ● ○ ○] | 1 − (0.5² + 0.5²) = **0.5** (iki sınıf için maksimum) |

```
 Safsızlık
 1.0 │        ╭───╮         Entropi
     │      ╱       ╲
 0.5 │    ╱  ╭─────╮  ╲     Gini
     │  ╱  ╱         ╲  ╲
   0 │╱──────────────────╲── p(sınıf 1)
     0        0.5         1
```

İki ölçü çok benzer davranır; Gini logaritma içermediği için biraz daha hızlı hesaplanır ve scikit-learn'ün varsayılanıdır.

### 2.3 Bilgi kazancı (Information Gain)

Bir bölmenin ne kadar iyi olduğunu, **bölmeden önceki safsızlık ile bölmeden sonraki ağırlıklı ortalama safsızlık arasındaki fark** ile ölçeriz:

```
                         |Sᵥ|
IG(S, A) = H(S) −  Σ   ─────── · H(Sᵥ)
                   v∈A    |S|
```

Sᵥ: A özniteliğinin v değerini alan alt grup. Ağırlık |Sᵥ|/|S|, büyük grupların daha çok önemsenmesini sağlar.

**Algoritma her adımda en yüksek bilgi kazancını sağlayan soruyu seçer.**

---

## 3. Elle Ağaç İnşası: "Tenis Oynayalım mı?"

| # | Hava | Sıcaklık | Nem | Rüzgar | Oyna? |
|---|---|---|---|---|---|
| 1 | Güneşli | Sıcak | Yüksek | Yok | Hayır |
| 2 | Güneşli | Sıcak | Yüksek | Var | Hayır |
| 3 | Bulutlu | Sıcak | Yüksek | Yok | Evet |
| 4 | Yağmurlu | Ilık | Yüksek | Yok | Evet |
| 5 | Yağmurlu | Serin | Normal | Yok | Evet |
| 6 | Yağmurlu | Serin | Normal | Var | Hayır |
| 7 | Bulutlu | Serin | Normal | Var | Evet |
| 8 | Güneşli | Ilık | Yüksek | Yok | Hayır |
| 9 | Güneşli | Serin | Normal | Yok | Evet |
| 10 | Yağmurlu | Ilık | Normal | Yok | Evet |
| 11 | Güneşli | Ilık | Normal | Var | Evet |
| 12 | Bulutlu | Ilık | Yüksek | Var | Evet |
| 13 | Bulutlu | Sıcak | Normal | Yok | Evet |
| 14 | Yağmurlu | Ilık | Yüksek | Var | Hayır |

### Adım 1: Kök düğümün entropisi

9 Evet, 5 Hayır:

```
H(S) = −(9/14)·log₂(9/14) − (5/14)·log₂(5/14)
     = −0.643·(−0.637) − 0.357·(−1.485)
     = 0.410 + 0.530 = 0.940
```

### Adım 2: Her öznitelik için bilgi kazancı

**Hava:**

| Değer | Örnekler | Evet / Hayır | Entropi |
|---|---|---|---|
| Güneşli | 1, 2, 8, 9, 11 | 2 / 3 | 0.971 |
| Bulutlu | 3, 7, 12, 13 | 4 / 0 | 0.000 |
| Yağmurlu | 4, 5, 6, 10, 14 | 3 / 2 | 0.971 |

```
Ağırlıklı entropi = (5/14)·0.971 + (4/14)·0 + (5/14)·0.971 = 0.694
IG(Hava) = 0.940 − 0.694 = 0.246
```

**Nem:**

| Değer | Evet / Hayır | Entropi |
|---|---|---|
| Yüksek (7) | 3 / 4 | 0.985 |
| Normal (7) | 6 / 1 | 0.592 |

```
IG(Nem) = 0.940 − [(7/14)·0.985 + (7/14)·0.592] = 0.940 − 0.789 = 0.151
```

**Rüzgar:**

| Değer | Evet / Hayır | Entropi |
|---|---|---|
| Yok (8) | 6 / 2 | 0.811 |
| Var (6) | 3 / 3 | 1.000 |

```
IG(Rüzgar) = 0.940 − [(8/14)·0.811 + (6/14)·1.0] = 0.940 − 0.892 = 0.048
```

**Sıcaklık:** Sıcak (2/2) → 1.0, Ilık (4/2) → 0.918, Serin (3/1) → 0.811

```
IG(Sıcaklık) = 0.940 − [(4/14)·1.0 + (6/14)·0.918 + (4/14)·0.811] = 0.940 − 0.911 = 0.029
```

| Öznitelik | Bilgi kazancı |
|---|---|
| **Hava** | **0.246** ← en yüksek |
| Nem | 0.151 |
| Rüzgar | 0.048 |
| Sıcaklık | 0.029 |

**Kök düğüm: Hava.** "Bulutlu" dalı zaten saf (hep Evet) → yaprak olur.

### Adım 3: Alt dallarda tekrarla (özyineleme)

**Güneşli dalı** (5 örnek: 2 Evet, 3 Hayır, H = 0.971):

| Nem | Evet / Hayır |
|---|---|
| Yüksek (1, 2, 8) | 0 / 3 → saf |
| Normal (9, 11) | 2 / 0 → saf |

IG(Nem) = 0.971 − 0 = 0.971 → Mükemmel bölme. (Rüzgar ve Sıcaklık için daha düşük çıktığını kendiniz doğrulayın.)

**Yağmurlu dalı** (5 örnek: 3 Evet, 2 Hayır):

| Rüzgar | Evet / Hayır |
|---|---|
| Yok (4, 5, 10) | 3 / 0 → saf |
| Var (6, 14) | 0 / 2 → saf |

IG(Rüzgar) = 0.971 → Mükemmel bölme.

**Sonuç:** Bölüm 1'de gösterdiğimiz ağacın aynısını elde ettik! Tüm yapraklar saf olduğu için algoritma durur.

### Durma koşulları

Algoritma bir düğümde şu durumlardan birinde durur ve yaprak oluşturur:
1. Düğümdeki tüm örnekler aynı sınıftaysa
2. Bölünecek öznitelik kalmadıysa
3. Belirlenen bir sınıra ulaşıldıysa (maksimum derinlik, minimum örnek sayısı vb. — Bölüm 6)

Yaprağın tahmini: **çoğunluk sınıfı** (sınıflandırma) veya **ortalama** (regresyon).

---

## 4. Sayısal Öznitelikler ve CART Algoritması

### 4.1 Eşik değeri nasıl bulunur?

Sayısal öznitelikler (yaş, gelir) için soru "x ≤ t?" biçimindedir. t için olası tüm değerleri denemek yerine:

1. Özniteliğe göre örnekleri **sırala**.
2. Ardışık iki farklı değerin **orta noktalarını** aday eşik olarak al.
3. Her aday için ağırlıklı safsızlığı hesapla, en iyisini seç.

**Örnek:** Yaşa göre kredi onayı

| Yaş | 22 | 25 | 30 | 35 | 40 | 52 |
|---|---|---|---|---|---|---|
| Onay | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |

Aday eşikler: 23.5, 27.5, 32.5, 37.5, 46. Gini ile hesaplayalım (toplam 3✓, 3✗; kök Gini = 0.5):

| Eşik | Sol (≤) | Sağ (>) | Gini sol | Gini sağ | Ağırlıklı Gini |
|---|---|---|---|---|---|
| 23.5 | ✗ | ✗✓✓✓✗ | 0 | 1 − (0.6² + 0.4²) = 0.48 | (1/6)·0 + (5/6)·0.48 = 0.400 |
| **27.5** | ✗✗ | ✓✓✓✗ | 0 | 1 − (0.75² + 0.25²) = 0.375 | (2/6)·0 + (4/6)·0.375 = **0.250** |
| 32.5 | ✗✗✓ | ✓✓✗ | 0.444 | 0.444 | 0.444 |
| 37.5 | ✗✗✓✓ | ✓✗ | 0.5 | 0.5 | 0.500 |
| 46 | ✗✗✓✓✓ | ✗ | 0.48 | 0 | 0.400 |

En iyi eşik: **Yaş ≤ 27.5** (Gini azalması = 0.5 − 0.25 = 0.25).

Not: Aynı sayısal öznitelik ağacın farklı seviyelerinde tekrar kullanılabilir (ör. önce Yaş ≤ 27.5, sonra sağ dalda Yaş ≤ 46).

### 4.2 CART'ın özellikleri

scikit-learn'ün kullandığı **CART (Classification and Regression Trees)** algoritması:
- Her zaman **ikili bölme** yapar (her düğümden 2 dal)
- Kategorik öznitelikleri de sayısal kodlanmış haliyle "x ≤ t" ile böler
- Sınıflandırmada Gini/entropi, regresyonda MSE kullanır
- Açgözlüdür (greedy): Her adımda o an en iyi bölmeyi seçer, geleceği düşünmez. Bu, global olarak en iyi ağacı garanti etmez ama hızlıdır.

### 4.3 Regresyon ağaçları

Hedef sayısal ise:
- **Bölme ölçütü:** Varyansı / MSE'yi en çok azaltan bölme
- **Yaprak tahmini:** Yapraktaki örneklerin hedef ortalaması

```
          [Metrekare ≤ 100?]
           ╱             ╲
   [Yaş ≤ 10?]        [Oda ≤ 3?]
    ╱      ╲           ╱      ╲
 2.8M     2.1M       4.5M     6.2M    ← yaprak = ortalama fiyat
```

Sonuç, **basamaklı (parçalı sabit)** bir fonksiyondur:

```
 y
  │        ┌────┐
  │   ┌────┘    │    ┌───
  │───┘         └────┘
  └────────────────────── x
```

---

## 5. Sıfırdan Karar Ağacı (CART, Gini)

```python
import numpy as np

class Dugum:
    def __init__(self, oznitelik=None, esik=None, sol=None, sag=None, deger=None):
        self.oznitelik, self.esik = oznitelik, esik
        self.sol, self.sag = sol, sag
        self.deger = deger                  # sadece yapraklarda dolu

    def yaprak_mi(self):
        return self.deger is not None

def gini(y):
    _, sayilar = np.unique(y, return_counts=True)
    p = sayilar / len(y)
    return 1 - np.sum(p ** 2)

def entropi(y):
    _, sayilar = np.unique(y, return_counts=True)
    p = sayilar / len(y)
    return -np.sum(p * np.log2(p))

class BenimKararAgaci:
    def __init__(self, max_derinlik=None, min_ornek_bolme=2, olcut="gini"):
        self.max_derinlik = max_derinlik
        self.min_ornek_bolme = min_ornek_bolme
        self.safsizlik = gini if olcut == "gini" else entropi

    def fit(self, X, y):
        self.kok = self._buyut(np.asarray(X), np.asarray(y), derinlik=0)
        return self

    def _en_iyi_bolme(self, X, y):
        en_iyi = {"kazanc": 0.0}
        ebeveyn = self.safsizlik(y)
        n = len(y)
        for j in range(X.shape[1]):
            degerler = np.unique(X[:, j])                   # sıralı benzersiz değerler
            esikler = (degerler[:-1] + degerler[1:]) / 2    # orta noktalar
            for t in esikler:
                sol = X[:, j] <= t
                n_sol = sol.sum()
                agirlikli = (n_sol / n) * self.safsizlik(y[sol]) + \
                            ((n - n_sol) / n) * self.safsizlik(y[~sol])
                kazanc = ebeveyn - agirlikli
                if kazanc > en_iyi["kazanc"]:
                    en_iyi = {"kazanc": kazanc, "oznitelik": j, "esik": t}
        return en_iyi

    def _buyut(self, X, y, derinlik):
        # Durma koşulları
        if (len(np.unique(y)) == 1 or len(y) < self.min_ornek_bolme or
                (self.max_derinlik is not None and derinlik >= self.max_derinlik)):
            return Dugum(deger=self._cogunluk(y))

        b = self._en_iyi_bolme(X, y)
        if b["kazanc"] == 0:                                # faydalı bölme yok
            return Dugum(deger=self._cogunluk(y))

        sol = X[:, b["oznitelik"]] <= b["esik"]
        return Dugum(b["oznitelik"], b["esik"],
                     self._buyut(X[sol], y[sol], derinlik + 1),
                     self._buyut(X[~sol], y[~sol], derinlik + 1))

    @staticmethod
    def _cogunluk(y):
        degerler, sayilar = np.unique(y, return_counts=True)
        return degerler[np.argmax(sayilar)]

    def _tahmin_tek(self, x, dugum):
        while not dugum.yaprak_mi():
            dugum = dugum.sol if x[dugum.oznitelik] <= dugum.esik else dugum.sag
        return dugum.deger

    def predict(self, X):
        return np.array([self._tahmin_tek(x, self.kok) for x in np.asarray(X)])

    def yazdir(self, dugum=None, girinti="", isimler=None):
        dugum = dugum or self.kok
        if dugum.yaprak_mi():
            print(girinti + f"→ {dugum.deger}")
            return
        ad = isimler[dugum.oznitelik] if isimler else f"x[{dugum.oznitelik}]"
        print(girinti + f"{ad} <= {dugum.esik:.2f}?")
        self.yazdir(dugum.sol, girinti + "  │ evet ", isimler)
        self.yazdir(dugum.sag, girinti + "  │ hayır ", isimler)
```

**Yaş örneğiyle test:**

```python
X = np.array([[22], [25], [30], [35], [40], [52]])
y = np.array([0, 0, 1, 1, 1, 0])
agac = BenimKararAgaci().fit(X, y)
agac.yazdir(isimler=["Yaş"])
# Yaş <= 27.50?
#   │ evet → 0
#   │ hayır Yaş <= 46.00?
#   │ hayır   │ evet → 1
#   │ hayır   │ hayır → 0
```

**Iris verisinde scikit-learn ile karşılaştırma:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

bizim = BenimKararAgaci(max_derinlik=3).fit(X_tr, y_tr)
sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
print("Bizim  :", (bizim.predict(X_te) == y_te).mean())
print("sklearn:", sk.score(X_te, y_te))
bizim.yazdir(isimler=load_iris().feature_names)
```

**Karmaşıklık:** Her düğümde d öznitelik × n eşik × n safsızlık hesabı yapılır. Yukarıdaki naif kod O(d·n²) çalışır; scikit-learn sıralama ve kümülatif sayımlarla bunu O(d·n·log n)'e indirir.

---

## 6. Aşırı Öğrenme ve Budama

### 6.1 Sorun

Hiç sınırlanmayan bir ağaç, **her yaprak tek bir örnek içerene kadar** büyüyebilir. Bu durumda eğitim doğruluğu %100 olur ama ağaç gürültüyü ezberlemiştir.

```
Derinlik:     1        3          10         sınırsız
Eğitim:     %70      %88        %98          %100
Test:       %68      %85        %80          %76      ← aşırı öğrenme
```

Karar ağaçları **yüksek varyanslı** modellerdir: Veride küçük bir değişiklik tamamen farklı bir ağaç üretebilir (Hafta 8'de topluluk yöntemleri bu sorunu çözecek).

### 6.2 Ön budama (Pre-pruning) — erken durdurma

Ağaç büyürken sınırlar koyarız:

| Parametre | Anlamı | Etkisi |
|---|---|---|
| `max_depth` | Maksimum derinlik | En etkili ve en basit kontrol |
| `min_samples_split` | Bir düğümü bölmek için gereken min. örnek | Küçük grupların bölünmesini engeller |
| `min_samples_leaf` | Bir yapraktaki min. örnek | Tek örneklik yaprakları önler |
| `max_leaf_nodes` | Maksimum yaprak sayısı | Toplam karmaşıklığı sınırlar |
| `min_impurity_decrease` | Bölme için gereken min. kazanç | Faydasız bölmeleri engeller |

Sorun: Çok erken durursak, şu an küçük kazanç sağlayan ama altında çok faydalı bölmeler olan bir dalı kaçırabiliriz (XOR problemi gibi).

### 6.3 Son budama (Post-pruning) — maliyet-karmaşıklık budaması

Önce ağacı tamamen büyüt, sonra faydasız dalları kes. scikit-learn **maliyet-karmaşıklık budaması** kullanır:

```
R_α(T) = R(T) + α · |T|
```

- R(T): Ağacın eğitim hatası (safsızlık)
- |T|: Yaprak sayısı
- α (`ccp_alpha`): Karmaşıklık cezası

Lasso'daki mantığın aynısı: Hata + karmaşıklık cezası. α = 0 → tam ağaç. α arttıkça, hatayı en az artıran "en zayıf dallar" sırayla kesilir ve ağaç küçülür. En iyi α çapraz doğrulama ile seçilir.

```python
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

yol = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alfalar = yol.ccp_alphas[:-1]               # son değer sadece kökü bırakır

egitim, test, yaprak = [], [], []
for a in alfalar:
    m = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
    egitim.append(m.score(X_tr, y_tr)); test.append(m.score(X_te, y_te))
    yaprak.append(m.get_n_leaves())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(alfalar, egitim, "o-", label="Eğitim"); ax[0].plot(alfalar, test, "s-", label="Test")
ax[0].set_xlabel("ccp_alpha"); ax[0].set_ylabel("Doğruluk"); ax[0].legend()
ax[1].plot(alfalar, yaprak, "o-"); ax[1].set_xlabel("ccp_alpha"); ax[1].set_ylabel("Yaprak sayısı")
plt.show()
```

> Not: Burada test setini görselleştirme amaçlı kullandık. Gerçek bir projede α'yı **çapraz doğrulama** ile seçmelisiniz (`GridSearchCV`, Hafta 9).

---

## 7. Yorumlanabilirlik

Karar ağaçlarının en büyük avantajı **beyaz kutu** model olmalarıdır: Her kararın nedenini kural olarak okuyabiliriz.

```
EĞER Hava = Güneşli VE Nem = Yüksek İSE → Oynama
```

Bu, kredi, sağlık, hukuk gibi kararların açıklanması gereken alanlarda çok değerlidir.

### 7.1 Öznitelik önemi (feature importance)

Bir özniteliğin önemi = O özniteliğin kullanıldığı tüm bölmelerde sağlanan **toplam (örnek sayısıyla ağırlıklı) safsızlık azalması**, normalize edilmiş hali.

```python
import pandas as pd
veri = load_breast_cancer()
m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
onem = pd.Series(m.feature_importances_, index=veri.feature_names).sort_values(ascending=False)
print(onem.head(10))
```

Dikkat: Bu ölçü çok sayıda farklı değere sahip (yüksek kardinaliteli) öznitelikleri kayırma eğilimindedir. Daha güvenilir alternatif: permütasyon önemi (Hafta 14).

### 7.2 Karar ağaçlarının güçlü ve zayıf yönleri

| Güçlü yönler | Zayıf yönler |
|---|---|
| Yorumlanması kolay, görselleştirilebilir | Yüksek varyans: veri biraz değişince ağaç çok değişir |
| Ölçekleme gerektirmez | Kolayca aşırı öğrenir |
| Sayısal ve kategorik veriyle çalışır | Eksene paralel sınırlar: çapraz sınırları merdiven gibi yaklaşık modeller |
| Doğrusal olmayan ilişkileri ve etkileşimleri yakalar | Regresyonda basamaklı tahmin, eğitim aralığı dışına çıkamaz |
| Aykırı değerlere dayanıklı | Açgözlü algoritma global en iyiyi garanti etmez |

**Eksene paralel sınır sorunu:**

```
 x₂        gerçek sınır: çapraz        ağacın yaklaşımı: merdiven
  │ ○ ○ ○ ╱ ● ●                         │ ○ ○ ○ ┌─● ●
  │ ○ ○  ╱ ● ● ●                        │ ○ ○ ┌─┘● ● ●
  │ ○   ╱ ● ● ●                         │ ○ ┌─┘ ● ● ●
  │    ╱ ● ●                            │ ┌─┘ ● ●
  └────────────── x₁                    └────────────── x₁
```

---

## 8. Laboratuvar: Ağaç Görselleştirme

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
```

**Görev 1 — Tenis verisi:** Bölüm 3'teki tabloyu DataFrame olarak oluşturun, one-hot kodlayın ve `DecisionTreeClassifier(criterion="entropy")` ile eğitin. Elde ettiğiniz ağaç elle bulduğumuzla aynı mı?

```python
tenis = pd.DataFrame({
    "Hava": ["Güneşli","Güneşli","Bulutlu","Yağmurlu","Yağmurlu","Yağmurlu","Bulutlu",
             "Güneşli","Güneşli","Yağmurlu","Güneşli","Bulutlu","Bulutlu","Yağmurlu"],
    "Sıcaklık": ["Sıcak","Sıcak","Sıcak","Ilık","Serin","Serin","Serin",
                 "Ilık","Serin","Ilık","Ilık","Ilık","Sıcak","Ilık"],
    "Nem": ["Yüksek","Yüksek","Yüksek","Yüksek","Normal","Normal","Normal",
            "Yüksek","Normal","Normal","Normal","Yüksek","Normal","Yüksek"],
    "Rüzgar": ["Yok","Var","Yok","Yok","Yok","Var","Var",
               "Yok","Yok","Yok","Var","Var","Yok","Var"],
    "Oyna": ["Hayır","Hayır","Evet","Evet","Evet","Hayır","Evet",
             "Hayır","Evet","Evet","Evet","Evet","Evet","Hayır"],
})
X = pd.get_dummies(tenis.drop(columns="Oyna"))
y = tenis["Oyna"]
agac = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
print(export_text(agac, feature_names=list(X.columns)))
```

Elle hesaplamalarınızı kontrol etmek için bilgi kazancını hesaplayan bir fonksiyon yazın:

```python
def bilgi_kazanci(df, oznitelik, hedef="Oyna"):
    def H(s):
        p = s.value_counts(normalize=True)
        return -(p * np.log2(p)).sum()
    agirlikli = sum(len(g) / len(df) * H(g[hedef]) for _, g in df.groupby(oznitelik))
    return H(df[hedef]) - agirlikli

for o in ["Hava", "Sıcaklık", "Nem", "Rüzgar"]:
    print(f"{o:10s} IG = {bilgi_kazanci(tenis, o):.3f}")
```

**Görev 2 — Ağacı görselleştirme**

```python
iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3,
                                          random_state=0, stratify=iris.target)
m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)

plt.figure(figsize=(14, 7))
plot_tree(m, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.show()
```

Her kutuda şunlar yazar: bölme koşulu, gini değeri, örnek sayısı (`samples`), sınıf dağılımı (`value`) ve çoğunluk sınıfı. Kök düğümdeki Gini değerini elle doğrulayın.

**Görev 3 — Karar sınırları:** Sadece iki öznitelik (taç yaprak uzunluğu ve genişliği) kullanarak `max_depth` = 1, 2, 3, 10 için karar sınırlarını çizin. Eksene paralel dikdörtgen bölgeleri gözlemleyin.

```python
X2 = iris.data[:, 2:4]
xx, yy = np.meshgrid(np.linspace(0.5, 7.5, 300), np.linspace(0, 2.8, 300))
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, d in zip(ax, [1, 2, 3, 10]):
    mm = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X2, iris.target)
    Z = mm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    a.scatter(X2[:, 0], X2[:, 1], c=iris.target, edgecolors="k", s=20)
    a.set_title(f"max_depth = {d}")
plt.show()
```

**Görev 4 — Derinliğin etkisi:** Meme kanseri verisinde `max_depth` = 1...15 için 5-katlı CV doğruluğunu ve eğitim doğruluğunu çizin. Aşırı öğrenme hangi derinlikte başlıyor?

**Görev 5 — Kararsızlık deneyi:** Eğitim verisinden rastgele %10'unu çıkarıp ağacı 5 kez yeniden eğitin ve kök düğümleri karşılaştırın. Karar ağaçlarının varyansı hakkında ne söyleyebilirsiniz?

---

## 9. Proje Önerisi Teslimi

Bu hafta dönem projesi önerilerinizi teslim ediyorsunuz. Öneri (1–2 sayfa) şunları içermelidir:

1. **Problem tanımı:** Ne tahmin edilecek? Neden önemli?
2. **Veri seti:** Kaynak (Kaggle, UCI, kamu verisi, kendi topladığınız), boyut, öznitelikler, hedef değişken
3. **Problem türü:** Regresyon / sınıflandırma / kümeleme
4. **Değerlendirme metriği:** Hangi metrik ve neden? (ör. dengesiz veri → F1 veya PR-AUC)
5. **Baseline:** Karşılaştırma için basit bir model veya kural
6. **Planlanan yöntemler:** Derste gördüğünüz ve göreceğiniz yöntemlerden hangileri
7. **Olası riskler:** Veri sızıntısı, eksik veri, etik konular

---

## 10. Özet

- Karar ağacı, veriyi ardışık evet/hayır sorularıyla saf gruplara böler.
- **Entropi** (−Σp log₂p) ve **Gini** (1 − Σp²) safsızlığı ölçer. Her adımda **bilgi kazancı** (veya Gini azalması) en yüksek bölme seçilir.
- Sayısal özniteliklerde aday eşikler sıralı değerlerin **orta noktalarıdır**.
- CART ikili bölme yapar; regresyonda MSE azalmasını kullanır ve yapraklarda ortalama tahmin eder.
- Sınırsız ağaçlar **aşırı öğrenir**. Ön budama (`max_depth`, `min_samples_leaf`) ve son budama (`ccp_alpha`) ile kontrol edilir.
- Ağaçlar yorumlanabilir ve ölçekleme gerektirmez, ama yüksek varyanslıdır → Hafta 8: Topluluk yöntemleri.

## Kendinizi Sınayın

1. [● ● ● ● ○ ○] grubunun entropisini ve Gini değerini hesaplayın.
2. Tenis verisinde Güneşli dalı için Rüzgar özniteliğinin bilgi kazancını hesaplayın ve Nem ile karşılaştırın.
3. Yaş örneğinde (Bölüm 4.1) entropi kullanılsaydı en iyi eşik değişir miydi? Hesaplayın.
4. Neden karar ağaçları özniteliklerin ölçeklenmesinden etkilenmez?
5. "Müşteri ID" gibi her satırda farklı olan bir öznitelik bilgi kazancı ölçütüne göre neden çok yüksek skor alır? Bu neden bir sorundur?

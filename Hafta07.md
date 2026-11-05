# Hafta 7 — Karar Ağaçları

## Bu haftanın sonunda

Bu notu bitirdiğinizde:

- Entropi, Gini safsızlığı ve bilgi kazancını hesaplayabilirsiniz.
- Küçük bir veri setinde karar ağacını elle kurabilirsiniz.
- Sayısal özniteliklerde en iyi eşiğin nasıl seçildiğini açıklayabilirsiniz.
- Basit bir karar ağacını NumPy ile sıfırdan yazabilirsiniz.
- Ön budama ve maliyet-karmaşıklık budamasıyla aşırı öğrenmeyi azaltabilirsiniz.
- Bir ağacı görselleştirip yorumlayabilir, öznitelik önemlerini okuyabilirsiniz.

## Sezgi: 20 soru oyunu

Karar ağacı, "evet / hayır" soruları sorarak karara ulaşır. Bir doktorun düşünme biçimine benzer:

```
Ateş 38°C'nin üstünde mi?
├── Evet → Öksürük var mı?
│          ├── Evet → Grip olabilir
│          └── Hayır → Başka testler
└── Hayır → Muhtemelen sağlıklı
```

Terimler:

- **Kök düğüm:** İlk soru (en üstteki düğüm)
- **İç düğüm:** Ara sorular
- **Dal:** Bir sorunun cevabı
- **Yaprak:** Son karar (tahmin)
- **Derinlik:** Kökten en uzak yaprağa kadar olan soru sayısı

Ağacı kurarken asıl soru şudur: **Hangi soruyu önce sormalıyız?** Cevap: Veriyi en "saf" gruplara ayıran soruyu.

## Safsızlık ölçüleri

Bir düğümdeki örneklerin hepsi aynı sınıftaysa düğüm **saftır**. Sınıflar karışıksa düğüm **safsızdır**.

### Entropi

`Entropi(S) = − Σ pᵢ · log₂(pᵢ)`

`pᵢ`, düğümdeki `i` sınıfının oranıdır.

| Düğüm | Hesap | Entropi |
|---|---|---|
| 5 Evet, 5 Hayır | −0.5·log₂0.5 − 0.5·log₂0.5 | **1.0** (en karışık) |
| 9 Evet, 5 Hayır | −(9/14)·log₂(9/14) − (5/14)·log₂(5/14) | **0.940** |
| 4 Evet, 0 Hayır | −1·log₂1 | **0** (saf) |

### Neden entropi kullanırız?

Entropi, bilgi kuramından gelen bir kavramdır ve **belirsizliği** ölçer. Bir düğümden rastgele bir örnek çektiğimizi ve sınıfını tahmin etmeye çalıştığımızı düşünün:

- Düğümdeki örneklerin hepsi "Evet" ise tahmin etmek kolaydır. Belirsizlik yoktur, entropi **0** olur.
- Yarısı "Evet", yarısı "Hayır" ise yazı-tura atmaktan farkımız kalmaz. Belirsizlik en yüksektir, entropi **1** olur.

Karar ağacının amacı, her soruyla bu belirsizliği azaltmaktır. İyi bir soru, cevabı öğrendiğimizde sınıf hakkında daha emin olmamızı sağlar. Bilgi kazancı da tam olarak bunu ölçer: *Bu soruyu sorduktan sonra belirsizlik ne kadar azaldı?*

**Formüldeki logaritma ne işe yarar?** `−log₂(p)`, bir olayı gördüğümüzde ne kadar "şaşırdığımızı" ölçer. Sık görülen bir olay (`p` büyük) az şaşırtır, nadir bir olay (`p` küçük) çok şaşırtır. Entropi, bu şaşkınlığın ortalamasıdır. `log₂` kullanıldığı için sonuç **bit** cinsindendir. Entropisi 1 olan bir düğüm, sınıfı öğrenmek için ortalama 1 evet/hayır sorusu gerektiği anlamına gelir.

**Neden doğrudan hata oranını kullanmıyoruz?** Hata oranı, bölmeler arasındaki farkı her zaman yakalayamaz. 40 Evet ve 40 Hayır içeren bir düğümü iki farklı şekilde bölelim:

| Bölme | Sol düğüm | Sağ düğüm | Hata oranı | Ağırlıklı entropi | Bilgi kazancı |
|---|---|---|---|---|---|
| A | 30E, 10H | 10E, 30H | 20/80 = 0.25 | 0.811 | 0.189 |
| B | 20E, 40H | 20E, 0H | 20/80 = 0.25 | (60/80) · 0.918 = 0.689 | **0.311** |

İki bölmenin hata oranı aynıdır, ama B bölmesi **tamamen saf** bir düğüm (20E, 0H) oluşturur. Entropi bu farkı görür ve B'yi seçer. Saf düğümler ağacın daha az soruyla doğru karara ulaşmasını sağlar.

Özetle entropi:

- Saf düğümde 0, en karışık düğümde en yüksek değeri alır.
- Saflığa doğru her adımı ödüllendirir; hata oranından daha hassastır.
- Ağacın "hangi soruyu sormalıyım?" kararını sayısal bir ölçüte bağlar.

### Gini safsızlığı

`Gini(S) = 1 − Σ pᵢ²`

- 5 Evet, 5 Hayır: `1 − (0.5² + 0.5²) = 0.5`
- 9 Evet, 5 Hayır: `1 − ((9/14)² + (5/14)²) = 1 − (0.413 + 0.128) = 0.459`
- Saf düğüm: `1 − 1² = 0`

Gini'nin hesaplanması logaritma gerektirmediği için biraz daha hızlıdır. scikit-learn varsayılan olarak Gini kullanır. Pratikte iki ölçü çoğu zaman benzer ağaçlar üretir.

### Bilgi kazancı

Bir öznitelikle bölme yapınca safsızlıktaki azalma **bilgi kazancıdır**:

`Kazanç(S, A) = Entropi(S) − Σ (|Sᵥ| / |S|) · Entropi(Sᵥ)`

Yani: *bölmeden önceki entropi − bölmeden sonraki ağırlıklı ortalama entropi*. En yüksek kazancı veren öznitelik seçilir.

## Tenis veri setinde ağacı elle kurma

Hava durumuna göre tenis oynanıp oynanmayacağını tahmin edelim.

| Gün | Hava | Sıcaklık | Nem | Rüzgâr | Oyna |
|---|---|---|---|---|---|
| 1 | Güneşli | Sıcak | Yüksek | Zayıf | Hayır |
| 2 | Güneşli | Sıcak | Yüksek | Güçlü | Hayır |
| 3 | Bulutlu | Sıcak | Yüksek | Zayıf | Evet |
| 4 | Yağmurlu | Ilık | Yüksek | Zayıf | Evet |
| 5 | Yağmurlu | Serin | Normal | Zayıf | Evet |
| 6 | Yağmurlu | Serin | Normal | Güçlü | Hayır |
| 7 | Bulutlu | Serin | Normal | Güçlü | Evet |
| 8 | Güneşli | Ilık | Yüksek | Zayıf | Hayır |
| 9 | Güneşli | Serin | Normal | Zayıf | Evet |
| 10 | Yağmurlu | Ilık | Normal | Zayıf | Evet |
| 11 | Güneşli | Ilık | Normal | Güçlü | Evet |
| 12 | Bulutlu | Ilık | Yüksek | Güçlü | Evet |
| 13 | Bulutlu | Sıcak | Normal | Zayıf | Evet |
| 14 | Yağmurlu | Ilık | Yüksek | Güçlü | Hayır |

Toplam: 9 Evet, 5 Hayır → `Entropi(S) = 0.940`

### Adım 1 — "Hava" özniteliğinin kazancı

| Hava | Evet | Hayır | Entropi |
|---|---|---|---|
| Güneşli (5) | 2 | 3 | 0.971 |
| Bulutlu (4) | 4 | 0 | 0 |
| Yağmurlu (5) | 3 | 2 | 0.971 |

Ağırlıklı entropi: `(5/14)·0.971 + (4/14)·0 + (5/14)·0.971 = 0.694`

`Kazanç(Hava) = 0.940 − 0.694 = 0.246`

### Adım 2 — "Rüzgâr" özniteliğinin kazancı

| Rüzgâr | Evet | Hayır | Entropi |
|---|---|---|---|
| Zayıf (8) | 6 | 2 | 0.811 |
| Güçlü (6) | 3 | 3 | 1.0 |

Ağırlıklı entropi: `(8/14)·0.811 + (6/14)·1.0 = 0.892`

`Kazanç(Rüzgâr) = 0.940 − 0.892 = 0.048`

### Adım 3 — Tüm öznitelikleri karşılaştırma

Aynı hesabı Nem ve Sıcaklık için de yaparsak:

| Öznitelik | Bilgi kazancı |
|---|---|
| **Hava** | **0.246** |
| Nem | 0.151 |
| Rüzgâr | 0.048 |
| Sıcaklık | 0.029 |

En yüksek kazanç **Hava**'dadır → kök düğüm "Hava" olur.

### Adım 4 — Alt dalları kurma

- **Bulutlu:** 4 Evet, 0 Hayır → saf. Yaprak: **Evet**.
- **Güneşli** (gün 1, 2, 8, 9, 11): Nem'e bakarsak Yüksek → hepsi Hayır, Normal → hepsi Evet. Nem bu dalı mükemmel ayırır.
- **Yağmurlu** (gün 4, 5, 6, 10, 14): Rüzgâr'a bakarsak Zayıf → hepsi Evet, Güçlü → hepsi Hayır. Rüzgâr bu dalı mükemmel ayırır.

Son ağaç:

```
Hava?
├── Güneşli  → Nem?
│              ├── Yüksek → Hayır
│              └── Normal → Evet
├── Bulutlu  → Evet
└── Yağmurlu → Rüzgâr?
               ├── Zayıf  → Evet
               └── Güçlü  → Hayır
```

Dikkat: "Sıcaklık" özniteliği ağaçta hiç kullanılmadı. Ağaç, işe yaramayan öznitelikleri kendiliğinden dışarıda bırakır.

## Sayısal özniteliklerde eşik seçimi

Sayısal bir öznitelikte soru "`x ≤ eşik` mi?" biçimindedir. Eşik şöyle seçilir:

1. Değerleri sırala.
2. Ardışık değerlerin orta noktalarını aday eşik olarak al.
3. Her aday için ağırlıklı safsızlığı hesapla, en düşüğü seç.

Örnek:

| Yaş | 22 | 25 | 30 | 35 | 40 | 50 |
|---|---|---|---|---|---|---|
| Satın aldı mı? | Hayır | Hayır | Evet | Evet | Evet | Hayır |

Bölmeden önce Gini: 3 Evet, 3 Hayır → `0.5`

| Eşik | Sol | Sağ | Ağırlıklı Gini |
|---|---|---|---|
| 23.5 | 1H (0) | 3E, 2H (0.48) | 5/6 · 0.48 = 0.400 |
| **27.5** | **2H (0)** | **3E, 1H (0.375)** | **4/6 · 0.375 = 0.250** |
| 32.5 | 1E, 2H (0.444) | 2E, 1H (0.444) | 0.444 |
| 37.5 | 2E, 2H (0.5) | 1E, 1H (0.5) | 0.500 |
| 45 | 3E, 2H (0.48) | 1H (0) | 5/6 · 0.48 = 0.400 |

En iyi eşik **Yaş ≤ 27.5** olur.

## CART ve regresyon ağaçları

scikit-learn'ün kullandığı algoritma **CART**'tır (Classification and Regression Trees):

- Her düğümde her zaman **ikiye** böler (ikili ağaç).
- Sınıflandırmada Gini veya entropi kullanır.
- Regresyonda da kullanılabilir.

**Regresyon ağacında:**

- Safsızlık yerine **MSE (varyans)** kullanılır: Bölme, alt gruplardaki değerlerin birbirine yakın olmasını hedefler.
- Yapraktaki tahmin, o yapraktaki örneklerin **ortalamasıdır**.
- Bu nedenle tahmin fonksiyonu "basamaklı" görünür.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([5, 6, 5, 20, 21, 22])

model = DecisionTreeRegressor(max_depth=1)
model.fit(X, y)
print(model.predict([[2.5], [5.5]]))   # [ 5.333 21. ]
```

Ağaç `x ≤ 3.5` eşiğinde böler. Sol yaprak `(5 + 6 + 5) / 3 ≈ 5.33`, sağ yaprak `(20 + 21 + 22) / 3 = 21` tahmin eder.

## NumPy ile sıfırdan karar ağacı

Sayısal öznitelikler için Gini kullanan basit bir sınıflandırma ağacı:

```python
import numpy as np

def gini(y):
    _, sayilar = np.unique(y, return_counts=True)
    p = sayilar / len(y)
    return 1 - np.sum(p ** 2)

def en_iyi_bolme(X, y):
    en_iyi = (None, None, gini(y))   # (öznitelik, eşik, safsızlık)
    for j in range(X.shape[1]):
        degerler = np.unique(X[:, j])
        esikler = (degerler[:-1] + degerler[1:]) / 2
        for t in esikler:
            sol, sag = y[X[:, j] <= t], y[X[:, j] > t]
            g = (len(sol) * gini(sol) + len(sag) * gini(sag)) / len(y)
            if g < en_iyi[2]:
                en_iyi = (j, t, g)
    return en_iyi

def agac_kur(X, y, derinlik=0, max_derinlik=3):
    j, t, _ = en_iyi_bolme(X, y)
    # Durma koşulu: derinlik sınırı veya iyileştiren bölme yok
    if derinlik == max_derinlik or j is None:
        degerler, sayilar = np.unique(y, return_counts=True)
        return {"yaprak": degerler[np.argmax(sayilar)]}
    maske = X[:, j] <= t
    return {
        "oznitelik": j,
        "esik": t,
        "sol": agac_kur(X[maske], y[maske], derinlik + 1, max_derinlik),
        "sag": agac_kur(X[~maske], y[~maske], derinlik + 1, max_derinlik),
    }

def tahmin_et(dugum, x):
    while "yaprak" not in dugum:
        if x[dugum["oznitelik"]] <= dugum["esik"]:
            dugum = dugum["sol"]
        else:
            dugum = dugum["sag"]
    return dugum["yaprak"]
```

Iris verisinde scikit-learn ile karşılaştırma:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

agac = agac_kur(X_egitim, y_egitim, max_derinlik=3)
tahminler = np.array([tahmin_et(agac, x) for x in X_test])
print("Bizim ağacımız :", np.mean(tahminler == y_test))

sk_agac = DecisionTreeClassifier(max_depth=3, random_state=42)
sk_agac.fit(X_egitim, y_egitim)
print("scikit-learn   :", sk_agac.score(X_test, y_test))
```

İki doğruluk değerinin aynı ya da birbirine çok yakın olması beklenir. Küçük farklar, eşitlik durumlarında farklı bölmelerin seçilmesinden kaynaklanabilir.

## Aşırı öğrenme ve budama

Sınırsız bırakılan bir ağaç, her yaprakta tek bir örnek kalana kadar bölmeye devam eder. Eğitim doğruluğu %100 olur, ama ağaç gürültüyü ezberlemiştir ve test verisinde kötü sonuç verir. Bu yüzden ağacı **budarız**.

### Ön budama (erken durdurma)

Ağaç büyürken sınır koyarız:

| Parametre | Anlamı |
|---|---|
| `max_depth` | En fazla derinlik |
| `min_samples_split` | Bir düğümün bölünmesi için gereken en az örnek sayısı |
| `min_samples_leaf` | Bir yaprakta bulunması gereken en az örnek sayısı |
| `max_leaf_nodes` | En fazla yaprak sayısı |

### Maliyet-karmaşıklık budaması (son budama)

Önce ağaç tam büyütülür, sonra fazla dallar kesilir. Her ağaç şu ölçüyle değerlendirilir:

`Rα(T) = R(T) + α · |T|`

- `R(T)`: Ağacın eğitim hatası (safsızlığı)
- `|T|`: Yaprak sayısı
- `α`: Karmaşıklık cezası

`α = 0` iken ağaç tam halindedir. `α` büyüdükçe daha basit ağaçlar tercih edilir. En iyi `α` çapraz doğrulamayla seçilir.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Budamasız ağaç
tam = DecisionTreeClassifier(random_state=42).fit(X_egitim, y_egitim)
print("Budamasız → eğitim:", tam.score(X_egitim, y_egitim),
      "test:", round(tam.score(X_test, y_test), 3),
      "yaprak:", tam.get_n_leaves())

# Aday alfa değerleri
yol = tam.cost_complexity_pruning_path(X_egitim, y_egitim)
alfalar = yol.ccp_alphas[:-1]   # sonuncusu yalnızca kökten oluşan ağaç

# Çapraz doğrulama ile en iyi alfa
skorlar = [
    cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=42),
                    X_egitim, y_egitim, cv=5).mean()
    for a in alfalar
]
en_iyi_alfa = alfalar[np.argmax(skorlar)]

budanmis = DecisionTreeClassifier(ccp_alpha=en_iyi_alfa, random_state=42)
budanmis.fit(X_egitim, y_egitim)
print("Budanmış  → eğitim:", round(budanmis.score(X_egitim, y_egitim), 3),
      "test:", round(budanmis.score(X_test, y_test), 3),
      "yaprak:", budanmis.get_n_leaves())
```

Budanmış ağaç genellikle çok daha az yaprakla benzer ya da daha iyi test doğruluğu verir. Test verisi `α` seçiminde kullanılmaz; yalnızca son değerlendirme içindir.

## Yorumlanabilirlik ve öznitelik önemi

Karar ağaçlarının en büyük avantajı **anlaşılabilir** olmalarıdır. Her tahmin, kökten yaprağa giden bir kural zinciri olarak okunabilir.

### Ağacı metin olarak yazdırma

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(iris.data, iris.target)

print(export_text(model, feature_names=iris.feature_names))
```

### Öznitelik önemi

`feature_importances_`, her özniteliğin ağaçtaki toplam safsızlık azalmasına katkısını gösterir. Değerlerin toplamı 1'dir.

```python
import pandas as pd

onem = pd.Series(model.feature_importances_, index=iris.feature_names)
print(onem.sort_values(ascending=False))
```

Dikkat edilmesi gerekenler:

- Bu önem, **eğitim verisi** üzerinden hesaplanır.
- Çok farklı değer alan özniteliklere (ör. kimlik numarası) yanlı olarak yüksek önem verebilir.
- Daha güvenilir bir alternatif **permütasyon önemidir** (Hafta 14).

### Artılar ve eksiler

| Artıları | Eksileri |
|---|---|
| Kolay yorumlanır, görselleştirilebilir | Kolayca aşırı öğrenir |
| Ölçekleme gerektirmez | Veride küçük bir değişiklik ağacı tamamen değiştirebilir (kararsız) |
| Sayısal ve kategorik veriyle çalışır | Karar sınırları eksenlere paralel basamaklardır |
| Doğrusal olmayan ilişkileri yakalar | Tek başına genellikle en yüksek doğruluğu vermez |

Kararsızlık ve aşırı öğrenme sorunları, gelecek hafta göreceğimiz **topluluk yöntemleriyle** (Random Forest, Boosting) büyük ölçüde çözülür.

## Laboratuvar: Ağaç görselleştirme ve karar sınırları

### 1. Ağacı çizme

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(iris.data, iris.target)

plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True)
plt.show()
```

Her kutuda şunları görürsünüz: bölme sorusu, Gini değeri, örnek sayısı (`samples`), sınıflara göre dağılım (`value`) ve çoğunluk sınıfı (`class`). Renk tonu düğümün ne kadar saf olduğunu gösterir.

### 2. Derinliğin karar sınırına etkisi

```python
from sklearn.inspection import DecisionBoundaryDisplay

X = iris.data[:, 2:4]   # petal length, petal width
y = iris.target

fig, eksenler = plt.subplots(1, 3, figsize=(15, 4))
for eksen, derinlik in zip(eksenler, [1, 3, None]):
    m = DecisionTreeClassifier(max_depth=derinlik, random_state=42).fit(X, y)
    DecisionBoundaryDisplay.from_estimator(m, X, ax=eksen, alpha=0.4,
                                           response_method="predict")
    eksen.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20)
    eksen.set_title(f"max_depth = {derinlik}")
    eksen.set_xlabel("petal length (cm)")
    eksen.set_ylabel("petal width (cm)")
plt.tight_layout()
plt.show()
```

Gözlem: Derinlik arttıkça karar sınırları daha çok **dikdörtgen parçaya** bölünür. Sınırsız ağaç, tek tek noktalara özel küçük bölgeler oluşturur. Bu, aşırı öğrenmenin görsel işaretidir.

### 3. Proje önerisi teslimi

Bu hafta dönem projesi önerisini teslim ediyorsunuz. Öneriniz en fazla 1–2 sayfa olmalı ve şunları içermelidir:

1. **Proje başlığı ve ekip üyeleri**
2. **Problem tanımı:** Hangi soruyu çözmek istiyorsunuz? Neden önemli?
3. **Problem türü:** Regresyon mu, sınıflandırma mı, kümeleme mi?
4. **Veri seti:** Kaynağı, büyüklüğü (satır/sütun), hedef değişken
5. **Değerlendirme metriği:** Hangi metriği neden seçtiniz?
6. **Temel model (baseline):** Karşılaştırma yapacağınız basit model
7. **Planlanan yöntemler:** Denemeyi düşündüğünüz algoritmalar
8. **Olası riskler:** Eksik veri, dengesizlik, sızıntı riski vb.

## Uygulama

1. Tenis veri setinde "Nem" özniteliğinin bilgi kazancını elle hesaplayıp `0.151` sonucunu doğrulayın.
2. Aynı veri setinde kök düğüm için Gini'ye göre en iyi özniteliği bulun. Entropiyle aynı sonucu veriyor mu?
3. Yaş örneğinde son kişinin etiketini "Evet" yapın. En iyi eşik değişiyor mu?
4. Sıfırdan yazdığımız ağaca `min_ornek` parametresi ekleyin: Düğümde bu sayıdan az örnek varsa bölmeyi durdursun.
5. Meme kanseri verisinde `max_depth` değerini 1'den 15'e kadar değiştirip eğitim ve doğrulama doğruluklarını çizin.
6. `random_state` değerini değiştirerek farklı eğitim/test bölmelerinde ağacın yapısının nasıl değiştiğini gözlemleyin.

## Kısa özet

- Karar ağacı, veriyi art arda sorularla daha saf gruplara böler.
- Entropi ve Gini düğüm safsızlığını ölçer; bilgi kazancı bölmenin ne kadar iyi olduğunu gösterir.
- Sayısal özniteliklerde aday eşikler ardışık değerlerin orta noktalarıdır.
- CART ikili bölme yapar; regresyon ağaçları MSE kullanır ve yapraklarda ortalamayı tahmin eder.
- Sınırsız ağaç aşırı öğrenir; ön budama ve maliyet-karmaşıklık budaması bunu azaltır.
- Ağaçlar yorumlanabilir ve ölçekleme gerektirmez, ama kararsızdır.

## Kendinizi sınayın

1. Entropisi 0 olan bir düğüm ne anlama gelir? Entropi ne zaman en büyüktür?
2. Bilgi kazancı nasıl hesaplanır? Hangi öznitelik kök düğüm olarak seçilir?
3. Tenis örneğinde "Bulutlu" dalı neden hemen yaprak oldu?
4. Sayısal bir öznitelikte aday eşikler nasıl belirlenir?
5. Regresyon ağacı bir yaprakta nasıl tahmin üretir?
6. Ön budama ile maliyet-karmaşıklık budaması arasındaki fark nedir?
7. Karar ağaçları neden ölçekleme gerektirmez?
8. Ağaçlardaki öznitelik önemi neden yanıltıcı olabilir?

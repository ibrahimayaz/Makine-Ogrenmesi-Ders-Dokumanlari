# Hafta 6 — Örnek Tabanlı ve Olasılıksal Modeller

## Bu haftanın sonunda

Bu notu bitirdiğinizde:

- k-NN algoritmasının bir tahmini nasıl ürettiğini adım adım açıklayabilirsiniz.
- Farklı mesafe ölçülerini hesaplayabilir ve ne zaman kullanılacaklarını söyleyebilirsiniz.
- Bayes teoremini kullanarak koşullu olasılık hesaplayabilirsiniz.
- Naive Bayes ile basit bir spam sınıflandırmasını elle yapabilirsiniz.
- SMS spam verisinde Naive Bayes, k-NN ve lojistik regresyonu karşılaştırabilirsiniz.

## İki farklı yaklaşım

Bu hafta iki farklı model ailesini tanıyacağız:

- **Örnek tabanlı modeller (k-NN):** Eğitim sırasında bir formül öğrenmez, eğitim verisini saklar. Yeni bir örnek gelince ona en çok benzeyen örneklere bakar. Sezgisi şudur: *"Bana arkadaşını söyle, sana kim olduğunu söyleyeyim."*
- **Olasılıksal modeller (Naive Bayes):** Veriden olasılıklar öğrenir ve yeni örnek için "hangi sınıf daha olası?" sorusunu Bayes teoremiyle yanıtlar.

## k-En Yakın Komşu (k-NN)

### Nasıl çalışır?

Yeni bir örneği sınıflandırmak için:

1. Yeni örnek ile tüm eğitim örnekleri arasındaki mesafeyi hesapla.
2. Mesafeleri küçükten büyüğe sırala.
3. En yakın `k` örneği seç.
4. Sınıflandırmada bu komşuların **çoğunluk oyunu**, regresyonda **ortalamasını** tahmin olarak ver.

### Elle örnek

Eğitim verisi:

| Nokta | x₁ | x₂ | Sınıf |
|---|---|---|---|
| A | 1 | 1 | Kırmızı |
| B | 2 | 1 | Kırmızı |
| C | 4 | 4 | Mavi |
| D | 5 | 4 | Mavi |
| E | 4 | 5 | Mavi |

Yeni nokta: `q = (2, 2)`, `k = 3`.

Öklid mesafeleri:

| Nokta | Hesap | Mesafe |
|---|---|---|
| A | √((2−1)² + (2−1)²) = √2 | 1.41 |
| B | √((2−2)² + (2−1)²) = √1 | 1.00 |
| C | √((2−4)² + (2−4)²) = √8 | 2.83 |
| D | √((2−5)² + (2−4)²) = √13 | 3.61 |
| E | √((2−4)² + (2−5)²) = √13 | 3.61 |

En yakın 3 komşu: B (Kırmızı), A (Kırmızı), C (Mavi). Oylama 2'ye 1 olduğu için tahmin **Kırmızı** olur.

### k seçimi

- **Küçük k (ör. 1):** Model gürültüye çok duyarlıdır → aşırı öğrenme riski.
- **Büyük k:** Karar sınırı çok düzleşir → eksik öğrenme riski.
- İkili sınıflandırmada beraberliği önlemek için tek sayılı `k` seçmek pratiktir.
- En iyi `k` değeri **çapraz doğrulama** ile seçilir.

### Ölçekleme şarttır

k-NN mesafeye dayandığı için öznitelikler aynı ölçekte olmalıdır. Örneğin "yaş" (20–70) ve "maaş" (10.000–100.000) birlikte kullanılırsa mesafeyi neredeyse yalnızca maaş belirler. Bu yüzden k-NN'den önce `StandardScaler` kullanılır (Hafta 2).

### NumPy ile sıfırdan k-NN

```python
import numpy as np
from collections import Counter

def knn_tahmin(X_egitim, y_egitim, x_yeni, k=3):
    mesafeler = np.sqrt(np.sum((X_egitim - x_yeni) ** 2, axis=1))
    en_yakin = np.argsort(mesafeler)[:k]
    oylar = Counter(y_egitim[en_yakin])
    return oylar.most_common(1)[0][0]

X = np.array([[1, 1], [2, 1], [4, 4], [5, 4], [4, 5]])
y = np.array(["Kırmızı", "Kırmızı", "Mavi", "Mavi", "Mavi"])

print(knn_tahmin(X, y, np.array([2, 2]), k=3))   # Kırmızı
```

scikit-learn ile doğrulama:

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print(model.predict([[2, 2]]))   # ['Kırmızı']
```

## Mesafe ölçüleri

İki nokta: `x = (1, 2)` ve `y = (4, 6)`. Farklar: `(3, 4)`.

| Ölçü | Formül | Sonuç |
|---|---|---|
| Öklid | √(Σ (xᵢ − yᵢ)²) | √(9 + 16) = **5** |
| Manhattan | Σ \|xᵢ − yᵢ\| | 3 + 4 = **7** |
| Chebyshev | max \|xᵢ − yᵢ\| | max(3, 4) = **4** |
| Minkowski | (Σ \|xᵢ − yᵢ\|ᵖ)^(1/p) | p=1 → Manhattan, p=2 → Öklid |

- **Öklid:** "Kuş uçuşu" mesafe; en yaygın seçim.
- **Manhattan:** Şehir blokları arasında yürümek gibi; aykırı değerlere biraz daha dayanıklıdır.
- **Chebyshev:** En büyük farkı alır (satrançta şahın hamle sayısı).

### Kosinüs benzerliği

Kosinüs benzerliği iki vektör arasındaki **açıya** bakar, uzunluğu önemsemez. Metin verisinde çok kullanılır: Uzun ve kısa iki belge aynı konudaysa yine benzer çıkar.

`cos(x, y) = (x · y) / (‖x‖ ‖y‖)`

Örnek: `x · y = 1·4 + 2·6 = 16`, `‖x‖ = √5 ≈ 2.24`, `‖y‖ = √52 ≈ 7.21`

`cos = 16 / (2.24 × 7.21) ≈ 0.99` → vektörler neredeyse aynı yönü gösteriyor.

Kosinüs mesafesi: `1 − cos`.

## k-NN'in artıları ve eksileri

| Artıları | Eksileri |
|---|---|
| Anlaşılması çok kolay | Tahmin aşaması yavaş (tüm veriye mesafe hesaplanır) |
| Eğitim süresi yok | Ölçeklemeye çok duyarlı |
| Karmaşık karar sınırları oluşturabilir | Yüksek boyutta zayıflar |

## Bayes teoremi

Bayes teoremi, yeni bir kanıt gördüğümüzde bir olasılığı nasıl güncelleyeceğimizi söyler:

`P(A | B) = P(B | A) · P(A) / P(B)`

- `P(A)`: **Önsel (prior)** — kanıtı görmeden önceki inancımız
- `P(B | A)`: **Olabilirlik (likelihood)** — A doğruysa B'yi görme olasılığı
- `P(A | B)`: **Sonsal (posterior)** — kanıtı gördükten sonraki inancımız
- `P(B)`: Kanıtın toplam olasılığı

### Tıbbi test paradoksu

Bir hastalık toplumda **%1** oranında görülüyor. Test:

- Hasta olanların **%99**'unda pozitif çıkıyor.
- Sağlıklı olanların **%5**'inde yanlışlıkla pozitif çıkıyor.

Testiniz pozitif çıktı. Gerçekten hasta olma olasılığınız nedir?

`P(+) = 0.99 × 0.01 + 0.05 × 0.99 = 0.0099 + 0.0495 = 0.0594`

`P(Hasta | +) = 0.0099 / 0.0594 ≈ 0.167`

Yani yalnızca **yaklaşık %17**! Sezgimiz %99 der; ama hastalık çok nadir olduğu için pozitif sonuçların çoğu sağlıklı insanlardan gelir. Önsel olasılığı göz ardı etmemek gerekir.

Sayılarla düşünmek daha kolaydır: 10.000 kişinin 100'ü hastadır ve bunların 99'u pozitif çıkar. Kalan 9.900 sağlıklı kişinin 495'i de pozitif çıkar. Toplam 594 pozitifin yalnızca 99'u hastadır: 99 / 594 ≈ %17.

## Naive Bayes

Naive Bayes, bir örneğin her sınıfa ait olma olasılığını Bayes teoremiyle hesaplar ve en yüksek olasılıklı sınıfı seçer:

`P(sınıf | x₁, ..., xₙ) ∝ P(sınıf) · P(x₁ | sınıf) · P(x₂ | sınıf) · ... · P(xₙ | sınıf)`

**"Naive" (saf)** denmesinin nedeni, sınıf bilindiğinde özniteliklerin **birbirinden bağımsız** olduğunu varsaymasıdır. Bu varsayım gerçekte nadiren doğrudur (ör. "bedava" ve "hediye" kelimeleri spam'de birlikte geçer), ama model pratikte yine de çok iyi çalışır.

### Elle spam sınıflandırma

Eğitim mesajları:

| Mesaj | Sınıf |
|---|---|
| bedava hediye kazan | Spam |
| bedava para | Spam |
| toplantı yarın | Normal |
| yarın para gönder | Normal |

**Adım 1 — Önsel olasılıklar:** `P(Spam) = 2/4 = 0.5`, `P(Normal) = 2/4 = 0.5`

**Adım 2 — Kelime sayıları:**

| Kelime | Spam | Normal |
|---|---|---|
| bedava | 2 | 0 |
| hediye | 1 | 0 |
| kazan | 1 | 0 |
| para | 1 | 1 |
| toplantı | 0 | 1 |
| yarın | 0 | 2 |
| gönder | 0 | 1 |
| **Toplam** | **5** | **5** |

Sözlükte 7 farklı kelime var (`V = 7`).

**Yeni mesaj:** "bedava para"

### Sıfır olasılık sorunu

Düzeltme yapmazsak `P(bedava | Normal) = 0/5 = 0` olur. Çarpımın içinde tek bir sıfır bile tüm olasılığı sıfırlar. Model, "bedava" kelimesini içeren bir mesajın asla normal olamayacağına karar verir. Bu çok katı bir sonuçtur.

### Laplace düzeltmesi

Her kelime sayısına 1 ekleriz:

`P(kelime | sınıf) = (sayı + 1) / (sınıftaki toplam kelime + V)`

Spam için payda: `5 + 7 = 12`, Normal için payda: `5 + 7 = 12`.

| | Spam | Normal |
|---|---|---|
| P(bedava \| sınıf) | (2+1)/12 = 3/12 | (0+1)/12 = 1/12 |
| P(para \| sınıf) | (1+1)/12 = 2/12 | (1+1)/12 = 2/12 |
| Önsel × çarpım | 0.5 × 3/12 × 2/12 ≈ **0.0208** | 0.5 × 1/12 × 2/12 ≈ **0.0069** |

Normalleştirme: `P(Spam | mesaj) = 0.0208 / (0.0208 + 0.0069) = 0.75`

Mesaj **%75 olasılıkla Spam** olarak sınıflandırılır.

### Log olasılıklar

Gerçek mesajlarda yüzlerce kelime vardır. Çok sayıda küçük olasılığı çarpmak bilgisayarda sıfıra yuvarlanır (**alttan taşma**). Bu yüzden logaritma alıp toplarız: `log(a · b) = log a + log b`.

- Spam: `ln 0.5 + ln(3/12) + ln(2/12) = −0.693 − 1.386 − 1.792 = −3.871`
- Normal: `ln 0.5 + ln(1/12) + ln(2/12) = −0.693 − 2.485 − 1.792 = −4.970`

Daha büyük (sıfıra daha yakın) olan Spam'dir. Sonuç aynıdır, ama hesap sayısal olarak güvenlidir.

### scikit-learn ile doğrulama

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

mesajlar = ["bedava hediye kazan", "bedava para",
            "toplantı yarın", "yarın para gönder"]
etiketler = ["spam", "spam", "normal", "normal"]

vektorlestirici = CountVectorizer()
X = vektorlestirici.fit_transform(mesajlar)

model = MultinomialNB(alpha=1.0)   # alpha=1 → Laplace düzeltmesi
model.fit(X, etiketler)

yeni = vektorlestirici.transform(["bedava para"])
print(model.classes_)              # ['normal' 'spam']
print(model.predict_proba(yeni))   # [[0.25 0.75]]
```

Elle bulduğumuz %75 sonucu kütüphaneyle aynıdır.

### Naive Bayes türleri

| Tür | Veri tipi | Örnek |
|---|---|---|
| `MultinomialNB` | Sayım verisi | Kelime sayıları (metin) |
| `BernoulliNB` | İkili (var/yok) | Kelime geçiyor mu? |
| `GaussianNB` | Sürekli sayısal | Boy, kilo, ölçümler |

## Laboratuvar: SMS spam tespiti

Bu laboratuvarda SMS Spam Collection veri setiyle üç modeli karşılaştıracağız.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["etiket", "mesaj"])

print(df.shape)
print(df["etiket"].value_counts(normalize=True))

X_egitim, X_test, y_egitim, y_test = train_test_split(
    df["mesaj"], df["etiket"], test_size=0.2,
    stratify=df["etiket"], random_state=42
)

modeller = {
    "Naive Bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
    "k-NN": make_pipeline(TfidfVectorizer(),
                          KNeighborsClassifier(n_neighbors=5, metric="cosine")),
    "Lojistik Regresyon": make_pipeline(TfidfVectorizer(),
                                        LogisticRegression(max_iter=1000)),
}

for ad, model in modeller.items():
    model.fit(X_egitim, y_egitim)
    tahmin = model.predict(X_test)
    print(f"===== {ad} =====")
    print(classification_report(y_test, tahmin, digits=3))
```

Çıktıyı yorumlarken dikkat edin:

- Veri **dengesizdir** (mesajların çoğu normal). Bu yüzden yalnızca accuracy'ye değil, **spam sınıfının precision ve recall** değerlerine bakın (Hafta 5).
- Vektörleştirici `Pipeline` içinde olduğu için sözlük yalnızca eğitim verisinden öğrenilir; **veri sızıntısı** oluşmaz (Hafta 2).
- k-NN metinde kosinüs mesafesiyle kullanılır; çünkü metin vektörleri çok boyutlu ve seyrektir.
- Naive Bayes genellikle çok hızlıdır ve metin verisinde güçlü bir başlangıç modelidir.

## Uygulama

1. Elle k-NN örneğinde `k = 1` ve `k = 5` için tahmini bulun. Sonuç değişiyor mu?
2. `x = (2, 3)` ve `y = (5, 7)` için Öklid, Manhattan ve Chebyshev mesafelerini hesaplayın.
3. Tıbbi test örneğinde hastalık oranı %10 olsaydı pozitif testten sonra hasta olma olasılığı ne olurdu?
4. Elle spam örneğinde "yarın para" mesajını Laplace düzeltmesiyle sınıflandırın.
5. Laboratuvarda `MultinomialNB(alpha=...)` için `0.1`, `1` ve `10` değerlerini deneyin.
6. k-NN modelinde `n_neighbors` değerini değiştirip spam recall'ının nasıl değiştiğini gözlemleyin.
7. Modelin yanlış sınıflandırdığı birkaç mesajı yazdırın ve neden hata yapmış olabileceğini tartışın.

## Kısa özet

- k-NN eğitim verisini saklar; yeni örneği en yakın `k` komşunun oyuyla sınıflandırır.
- Küçük `k` aşırı öğrenmeye, büyük `k` eksik öğrenmeye yol açabilir; `k` çapraz doğrulamayla seçilir.
- Mesafe tabanlı modellerde ölçekleme zorunludur.
- Bayes teoremi önsel inancı kanıtla günceller; nadir olaylarda önsel olasılık çok önemlidir.
- Naive Bayes öznitelikleri bağımsız varsayar; Laplace düzeltmesi sıfır olasılığı, log olasılıklar alttan taşmayı önler.

## Kendinizi sınayın

1. k-NN neden "tembel öğrenici" (lazy learner) olarak adlandırılır?
2. k-NN'de öznitelikleri ölçeklemezsek ne olur? Bir örnekle açıklayın.
3. Kosinüs benzerliği metin verisi için neden Öklid mesafesinden daha uygun olabilir?
4. Tıbbi test paradoksunda sonucun %99 değil %17 çıkmasının nedeni nedir?
5. Naive Bayes'in "naive" varsayımı nedir? Bu varsayım bozulsa da model neden iyi çalışabilir?
6. Laplace düzeltmesi hangi sorunu çözer?

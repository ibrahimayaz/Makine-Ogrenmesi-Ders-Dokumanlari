# Hafta 5 — Lojistik Regresyon ve Sınıflandırma Metrikleri

## Bu Haftanın Hedefleri

- Sınıflandırma için neden doğrusal regresyonun yetersiz kaldığını anlamak
- Sigmoid fonksiyonu ve lojistik regresyonun olasılık üretme mantığını kavramak
- Çapraz entropi (log-loss) kaybını ve gradyanını türetmek, modeli sıfırdan kodlamak
- Karmaşıklık matrisi, accuracy, precision, recall, F1, ROC-AUC metriklerini elle hesaplamak
- Dengesiz verilerle çalışma tekniklerini öğrenmek ve eşik değerini bilinçli seçmek

---

## 1. Sınıflandırma Problemi

Hedef değişken artık bir **kategori**: Bu hafta **ikili sınıflandırmaya** (iki sınıf: 0 ve 1) odaklanıyoruz.

- E-posta spam mi? (1 = spam, 0 = değil)
- Tümör kötü huylu mu? (1 = kötü huylu)
- Müşteri aboneliğini iptal edecek mi? (1 = iptal)

Genellikle ilgilendiğimiz, nadir veya "aranan" sınıfa **pozitif sınıf (1)** deriz.

### 1.1 Neden doğrusal regresyon kullanmıyoruz?

Ders çalışma saatine göre sınavı geçip geçmemeyi tahmin edelim:

```
 y (geçti=1)
  1 │            ● ● ● ●  ●
    │                 ╱  doğrusal regresyon
0.5 │ ─ ─ ─ ─ ─ ─ ─ ╱─ ─ ─
    │           ╱
  0 │ ● ● ● ● ╱
    │      ╱  ← negatif "olasılık"?
    └─────────────────────── çalışma saati
```

Sorunlar:
1. Çıktı 0–1 aralığında değil: 1.4 veya −0.3 gibi anlamsız "olasılıklar" üretir.
2. Aykırı bir nokta (ör. 50 saat çalışan öğrenci) doğrunun eğimini değiştirir ve karar sınırını kaydırır.

İhtiyacımız: Her girdiyi **0 ile 1 arasına** sıkıştıran ve **olasılık** olarak yorumlanabilen bir çıktı.

---

## 2. Lojistik Regresyon

### 2.1 Sigmoid fonksiyonu

```
           1
σ(z) = ─────────
        1 + e⁻ᶻ
```

```
 σ(z)
  1 │                    ________
    │                 ╱
0.5 │ - - - - - - - ●
    │            ╱
  0 │________╱
    └───────────────────────── z
        -6     -3    0    3     6
```

| z | −∞ | −4 | −2 | 0 | 2 | 4 | +∞ |
|---|---|---|---|---|---|---|---|
| σ(z) | 0 | 0.018 | 0.119 | 0.5 | 0.881 | 0.982 | 1 |

Özellikleri:
- Çıktı her zaman (0, 1) aralığındadır.
- z = 0 → 0.5; z büyük pozitif → 1'e yakın; z büyük negatif → 0'a yakın.
- Türevi çok kullanışlıdır: **σ'(z) = σ(z)·(1 − σ(z))**

### 2.2 Model

Lojistik regresyon iki adımda çalışır:

```
1) Doğrusal skor:   z = wᵀx + b           (doğrusal regresyonla aynı)
2) Olasılık:        p = σ(z) = P(y=1 | x)
3) Karar:           ŷ = 1  eğer p ≥ 0.5   (yani z ≥ 0)
                    ŷ = 0  aksi halde
```

**Örnek:** Model w = 1.5, b = −6 öğrenmiş olsun (x = çalışma saati).

| Çalışma saati | z = 1.5x − 6 | p = σ(z) | Tahmin |
|---|---|---|---|
| 2 | −3.0 | 0.047 | 0 (kaldı) |
| 4 | 0.0 | 0.500 | sınırda |
| 5 | 1.5 | 0.818 | 1 (geçti) |
| 7 | 4.5 | 0.989 | 1 (geçti) |

### 2.3 Karar sınırı

p = 0.5 ⇔ z = 0 ⇔ **wᵀx + b = 0**. Bu bir doğrunun (2 boyutta), düzlemin (3 boyutta) ya da hiperdüzlemin denklemidir. Yani lojistik regresyon **doğrusal bir sınıflandırıcıdır**: Sınıfları düz bir sınırla ayırır.

```
 x₂
  │ ○ ○   ╲
  │  ○  ○  ╲  ●
  │ ○   ○   ╲   ● ●
  │    ○     ╲ ●   ●
  │           ╲   ●
  └────────────╲─────── x₁
         wᵀx + b = 0  (karar sınırı)
```

Doğrusal olmayan sınırlar için polinom öznitelikler eklenebilir (x₁², x₁x₂, ...).

### 2.4 Odds ve katsayıların yorumu

Sigmoid'i tersine çevirirsek:

```
   ⎛  p  ⎞
ln ⎜─────⎟ = wᵀx + b      ← "log-odds" (logit) doğrusal!
   ⎝1 − p⎠
```

**Odds** = p / (1 − p). Örneğin p = 0.8 → odds = 4 ("4'e 1 şans").

Yorum: xⱼ bir birim artınca log-odds wⱼ kadar artar, yani **odds e^wⱼ katına çıkar**. Örnekte w = 1.5 → e^1.5 ≈ 4.48: Her ek çalışma saati geçme odds'unu yaklaşık 4.5 katına çıkarır.

---

## 3. Kayıp Fonksiyonu: İkili Çapraz Entropi (Log-Loss)

### 3.1 Neden MSE değil?

Sigmoid ile MSE birleşince kayıp yüzeyi **konveks olmaz** (birçok yerel minimum ve düz bölgeler oluşur). Ayrıca olasılık tahminleri için MSE yeterince sert ceza vermez.

### 3.2 Çapraz entropi

Bir örnek için:

```
ℓ(y, p) = −[ y·ln(p) + (1 − y)·ln(1 − p) ]
```

İki durumu ayrı yazarsak:
- y = 1 ise: ℓ = −ln(p) → p'nin 1'e yakın olmasını ister
- y = 0 ise: ℓ = −ln(1 − p) → p'nin 0'a yakın olmasını ister

**Sayısal örnek (gerçek sınıf y = 1):**

| Model tahmini p | Kayıp −ln(p) | Yorum |
|---|---|---|
| 0.99 | 0.010 | Doğru ve emin → neredeyse ceza yok |
| 0.90 | 0.105 | Doğru → küçük ceza |
| 0.50 | 0.693 | Kararsız |
| 0.10 | 2.303 | Yanlış ve emin → büyük ceza |
| 0.01 | 4.605 | Çok yanlış ve çok emin → çok büyük ceza |

**Kendinden emin ama yanlış** tahminler çok sert cezalandırılır.

Tüm veri için ortalama alınır:

```
L(w, b) = −(1/n) Σ [ yᵢ ln(pᵢ) + (1 − yᵢ) ln(1 − pᵢ) ]
```

**Nereden geliyor?** Bu kayıp, **en çok olabilirlik (maximum likelihood)** ilkesinden türer. Her örneğin olasılığı P(yᵢ|xᵢ) = pᵢ^yᵢ (1−pᵢ)^(1−yᵢ)'dir. Bütün verinin olabilirliği bunların çarpımıdır; logaritmasını alıp eksiyle çarparsak çapraz entropiyi elde ederiz. Yani çapraz entropiyi minimize etmek = gözlenen etiketleri en olası kılan parametreleri bulmak.

### 3.3 Gradyan

Zincir kuralı ve σ'(z) = σ(z)(1 − σ(z)) özelliği kullanılınca çok sade bir sonuç çıkar:

```
∂L/∂w = (1/n) Σ (pᵢ − yᵢ) xᵢ  =  (1/n) Xᵀ(p − y)
∂L/∂b = (1/n) Σ (pᵢ − yᵢ)
```

**Türetme (tek örnek için):**
- ∂ℓ/∂p = −y/p + (1−y)/(1−p)
- ∂p/∂z = p(1−p)
- ∂z/∂w = x
- Çarpım: [−y/p + (1−y)/(1−p)] · p(1−p) · x = [−y(1−p) + (1−y)p] · x = **(p − y)·x**

Doğrusal regresyondaki gradyanla aynı biçimdedir: "(tahmin − gerçek) × girdi". Fark, tahminin artık σ(wᵀx + b) olmasıdır. Kapalı form çözüm **yoktur**; gradyan iniş (veya Newton yöntemi gibi optimizasyon algoritmaları) kullanılır.

### 3.4 Bir adımı elle hesaplayalım

Veri: (x=1, y=0), (x=3, y=1). Başlangıç w = 0, b = 0, α = 0.5.

| x | y | z | p = σ(z) | p − y | (p − y)·x |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.5 | 0.5 | 0.5 |
| 3 | 1 | 0 | 0.5 | −0.5 | −1.5 |
| | | | **Ortalama** | **0.0** | **−0.5** |

- w ← 0 − 0.5 × (−0.5) = **0.25**
- b ← 0 − 0.5 × 0.0 = **0**

Yeni olasılıklar: x=1 → σ(0.25) = 0.562, x=3 → σ(0.75) = 0.679. İkinci örneğin (y=1) olasılığı arttı. Adımlar tekrarlandıkça b negatife, w daha pozitife giderek iki sınıfı ayırır.

---

## 4. Sıfırdan Lojistik Regresyon

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # taşmayı önlemek için clip

class LojistikRegresyon:
    def __init__(self, alpha=0.1, epoch=2000, lam=0.0):
        self.alpha, self.epoch, self.lam = alpha, epoch, lam

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        self.kayiplar = []
        eps = 1e-12
        for _ in range(self.epoch):
            p = sigmoid(X @ self.w + self.b)
            kayip = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.kayiplar.append(kayip)
            dw = X.T @ (p - y) / n + self.lam * self.w      # L2 düzenlileştirme (isteğe bağlı)
            db = np.mean(p - y)
            self.w -= self.alpha * dw
            self.b -= self.alpha * db
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, esik=0.5):
        return (self.predict_proba(X) >= esik).astype(int)
```

**2 boyutlu veride deneyelim ve karar sınırını çizelim:**

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.2, random_state=1)
model = LojistikRegresyon(alpha=0.5, epoch=1000).fit(X, y)
print("w =", model.w, " b =", model.b)
print("Eğitim doğruluğu:", (model.predict(X) == y).mean())

# Olasılık haritası ve karar sınırı
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
P = model.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, P, levels=20, cmap="RdBu_r", alpha=0.6)
plt.colorbar(label="P(y=1)")
plt.contour(xx, yy, P, levels=[0.5], colors="k", linewidths=2)   # karar sınırı
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k")
plt.title("Lojistik regresyon: olasılıklar ve karar sınırı"); plt.show()
```

Grafikte renk geçişi sigmoid'in yumuşak olasılık geçişini, siyah çizgi ise p = 0.5 karar sınırını gösterir.

**scikit-learn ile karşılaştırma:**

```python
from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(penalty=None).fit(X, y)
print("sklearn  w =", sk.coef_, " b =", sk.intercept_)
```

> scikit-learn'deki `LogisticRegression` varsayılan olarak L2 düzenlileştirme kullanır; `C` parametresi **λ'nın tersidir** (C küçükse düzenlileştirme güçlüdür).

### 4.1 Çok sınıflı sınıflandırma

- **One-vs-Rest (OvR):** K sınıf için K ayrı ikili model ("sınıf k mı, değil mi?"). En yüksek olasılığı veren sınıf seçilir.
- **Softmax (çok terimli lojistik regresyon):** Her sınıf için bir skor zₖ hesaplanır ve olasılıklara dönüştürülür:

```
              e^zₖ
P(y=k|x) = ─────────
            Σⱼ e^zⱼ
```

Örnek: z = [2.0, 1.0, 0.1] → e^z = [7.39, 2.72, 1.11] → toplam 11.21 → P = [0.66, 0.24, 0.10]

```python
z = np.array([2.0, 1.0, 0.1])
print(np.exp(z) / np.exp(z).sum())   # [0.659 0.242 0.099]
```

scikit-learn çok sınıflı veride otomatik olarak softmax kullanır.

---

## 5. Sınıflandırma Metrikleri

### 5.1 Karmaşıklık matrisi (confusion matrix)

```
                       TAHMİN
                  Pozitif (1)    Negatif (0)
GERÇEK  Pozitif │     TP       │     FN       │
        Negatif │     FP       │     TN       │
```

- **TP (True Positive):** Hasta olana "hasta" dedik ✓
- **TN (True Negative):** Sağlıklıya "sağlıklı" dedik ✓
- **FP (False Positive, Tip I hata):** Sağlıklıya "hasta" dedik ✗ (yanlış alarm)
- **FN (False Negative, Tip II hata):** Hastaya "sağlıklı" dedik ✗ (kaçırılan vaka)

**Örnek:** 100 kişilik bir tarama testi: 50 hasta, 50 sağlıklı.

```
                   Tahmin: Hasta   Tahmin: Sağlıklı
Gerçek: Hasta          40 (TP)          10 (FN)
Gerçek: Sağlıklı        5 (FP)          45 (TN)
```

### 5.2 Metrikler

| Metrik | Formül | Örnek | Sorduğu soru |
|---|---|---|---|
| **Accuracy** (doğruluk) | (TP+TN)/Toplam | 85/100 = **0.85** | Tahminlerin ne kadarı doğru? |
| **Precision** (kesinlik) | TP/(TP+FP) | 40/45 = **0.889** | "Hasta" dediklerimizin ne kadarı gerçekten hasta? |
| **Recall** (duyarlılık, TPR) | TP/(TP+FN) | 40/50 = **0.80** | Gerçek hastaların ne kadarını yakaladık? |
| **Specificity** (özgüllük) | TN/(TN+FP) | 45/50 = **0.90** | Sağlıklıların ne kadarını doğru tanıdık? |
| **F1 skoru** | 2·P·R/(P+R) | 2·0.889·0.8/1.689 = **0.842** | Precision ve recall'un dengesi |

**F1 neden harmonik ortalama?** Aritmetik ortalama, biri çok düşük olsa bile yüksek çıkabilir. P = 1.0, R = 0.01 için aritmetik ortalama 0.505, F1 ise 0.02'dir. F1, iki metrikten biri bile kötüyse düşük kalır.

### 5.3 Precision mı, recall mı?

| Senaryo | Hangi hata daha kötü? | Öncelik |
|---|---|---|
| Kanser taraması | FN: Hastayı kaçırmak hayati | **Recall** |
| Spam filtresi | FP: Önemli bir e-postayı spam'e atmak | **Precision** |
| Sahtekarlık tespiti | FN genelde pahalı, ama çok FP müşteriyi bıktırır | Denge (F1 veya maliyet analizi) |
| Mahkeme kararı | FP: Masum birini mahkum etmek | **Precision** |

### 5.4 Sıfırdan metrik hesaplama

```python
def karmasiklik_matrisi(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def metrikler(y_true, y_pred):
    tp, tn, fp, fn = karmasiklik_matrisi(y_true, y_pred)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
```

scikit-learn karşılıkları:

```python
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             ConfusionMatrixDisplay)

print(classification_report(y_true, y_pred))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred); plt.show()
```

> **Dikkat:** scikit-learn `confusion_matrix` çıktısında satırlar gerçek, sütunlar tahmindir ve **sınıf 0 önce gelir**: `[[TN, FP], [FN, TP]]`.

---

## 6. Eşik Değeri ve ROC Eğrisi

### 6.1 Eşik değeri bir tercihtir

Model bize olasılık verir; 0.5 eşiği yalnızca bir varsayılandır.

- **Eşiği düşürürsek (ör. 0.2):** Daha çok örneğe "pozitif" deriz → recall ↑, precision ↓
- **Eşiği yükseltirsek (ör. 0.8):** Sadece çok emin olduklarımıza "pozitif" deriz → precision ↑, recall ↓

### 6.2 ROC eğrisi — çalışma mantığı

ROC eğrisi, **tüm olası eşik değerleri** için iki oranı çizer:

- **TPR (True Positive Rate)** = Recall = TP / (TP + FN)  → y ekseni
- **FPR (False Positive Rate)** = FP / (FP + TN) = 1 − Specificity  → x ekseni

**Elle örnek:** 6 örnek, modelin olasılıkları büyükten küçüğe sıralı:

| Örnek | Olasılık | Gerçek |
|---|---|---|
| A | 0.9 | 1 |
| B | 0.8 | 1 |
| C | 0.7 | 0 |
| D | 0.6 | 1 |
| E | 0.4 | 0 |
| F | 0.2 | 0 |

Pozitif sayısı P = 3, negatif sayısı N = 3. Eşiği her olasılık değerine sırayla indirelim ("≥ eşik" ise pozitif):

| Eşik | Pozitif denenler | TP | FP | TPR = TP/3 | FPR = FP/3 |
|---|---|---|---|---|---|
| > 0.9 | — | 0 | 0 | 0.00 | 0.00 |
| 0.9 | A | 1 | 0 | 0.33 | 0.00 |
| 0.8 | A, B | 2 | 0 | 0.67 | 0.00 |
| 0.7 | A, B, C | 2 | 1 | 0.67 | 0.33 |
| 0.6 | A–D | 3 | 1 | 1.00 | 0.33 |
| 0.4 | A–E | 3 | 2 | 1.00 | 0.67 |
| 0.2 | A–F | 3 | 3 | 1.00 | 1.00 |

```
 TPR
 1.0 │    ●──────●──────● (1,1)
     │    │
0.67 ●────●
     │
0.33 ●
     │                   ╱ rastgele tahmin (AUC=0.5)
 0.0 ●───────────────────── FPR
     0   0.33   0.67    1.0
```

**AUC (Area Under Curve):** Eğrinin altındaki alan.
- İlk dikdörtgen: 0.33 genişlik × 0.67 yükseklik = 0.222
- İkinci dikdörtgen: 0.67 genişlik × 1.0 yükseklik = 0.667
- **AUC = 0.889**

**AUC'nin olasılıksal anlamı:** Rastgele bir pozitif ve rastgele bir negatif örnek seçersek, modelin pozitife daha yüksek olasılık verme ihtimali. Kontrol edelim: 3 × 3 = 9 (pozitif, negatif) çifti var. Sadece (D=0.6, C=0.7) çiftinde negatif önde. 8/9 = **0.889** ✓

| AUC | Yorum |
|---|---|
| 1.0 | Mükemmel ayırma |
| 0.9+ | Çok iyi |
| 0.7–0.9 | Kabul edilebilir – iyi |
| 0.5 | Yazı-tura kadar iyi |
| < 0.5 | Rastgeleden kötü (etiketler ters mi?) |

**Sıfırdan ROC ve AUC:**

```python
def roc_egrisi(y_true, skor):
    sira = np.argsort(-skor)                # büyükten küçüğe
    y_sirali = y_true[sira]
    P, N = y_true.sum(), len(y_true) - y_true.sum()
    tpr = np.concatenate([[0], np.cumsum(y_sirali) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sirali) / N])
    return fpr, tpr

def auc(fpr, tpr):
    return np.trapz(tpr, fpr)               # yamuk kuralı ile alan

y_true = np.array([1, 1, 0, 1, 0, 0])
skor = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
fpr, tpr = roc_egrisi(y_true, skor)
print("AUC:", auc(fpr, tpr))               # 0.889

from sklearn.metrics import roc_auc_score
print("sklearn AUC:", roc_auc_score(y_true, skor))
```

(Yeni NumPy sürümlerinde `np.trapz` yerine `np.trapezoid` kullanılır.)

**ROC-AUC'nin avantajı:** Eşikten bağımsızdır. Modelin genel **sıralama** yeteneğini ölçer.

### 6.3 Precision-Recall eğrisi

Pozitif sınıf çok nadirse (%1) FPR'nin paydası (N) çok büyük olduğu için ROC eğrisi iyimser görünebilir. Bu durumda **Precision-Recall eğrisi** ve altındaki alan (**Average Precision, AP**) daha bilgilendiricidir.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
```

---

## 7. Dengesiz Veri (Imbalanced Data)

### 7.1 Doğruluk paradoksu

Kredi kartı işlemlerinin %99'u normal, %1'i sahtekarlık. **Her işleme "normal" diyen** aptal bir model:

- Accuracy = **%99** 🎉
- Recall (sahtekarlık için) = **0** — tek bir sahtekarlığı bile yakalamıyor!

**Ders:** Dengesiz verilerde accuracy yanıltıcıdır. Precision, recall, F1, PR-AUC kullanın ve her zaman "hep çoğunluk sınıfını tahmin et" modeliyle (baseline) karşılaştırın.

### 7.2 Çözüm yöntemleri

**1. Sınıf ağırlıkları:** Kayıp fonksiyonunda azınlık sınıfının hatalarına daha fazla ceza ver.

```
L = −(1/n) Σ [ w₁·yᵢ ln(pᵢ) + w₀·(1 − yᵢ) ln(1 − pᵢ) ]
```

`class_weight="balanced"` → ağırlık = n / (2 · sınıftaki örnek sayısı). %1'lik sınıf için ağırlık ~50, %99'luk sınıf için ~0.505.

```python
LogisticRegression(class_weight="balanced")
```

**2. Yeniden örnekleme (yalnızca eğitim setinde!):**
- **Alt örnekleme (undersampling):** Çoğunluk sınıfından rastgele örnek at → bilgi kaybı
- **Üst örnekleme (oversampling):** Azınlık sınıfını çoğalt → aşırı öğrenme riski
- **SMOTE:** Azınlık örnekleri arasında **sentetik** yeni örnekler üret: İki komşu azınlık örneği arasındaki doğru üzerinde rastgele bir nokta seçer.

```
SMOTE mantığı:   x_yeni = xᵢ + λ·(x_komşu − xᵢ),   λ ∈ [0, 1]

   ●xᵢ ─ ─ ─ ◆ ─ ─ ─ ●x_komşu
             ↑ yeni sentetik örnek
```

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
```

> Yeniden örnekleme **eğitim/test ayrımından sonra** ve **yalnızca eğitim verisine** uygulanmalıdır. Aksi halde sentetik örneklerin "kardeşleri" test setine sızar.

**3. Eşik ayarlama:** Modeli olduğu gibi eğitip doğrulama setinde iş hedefine uygun eşiği seç (Bölüm 8).

---

## 8. Laboratuvar: İkili Sınıflandırma ve Eşik Değeri Analizi

**Veri:** Meme kanseri teşhis veri seti (569 hasta, 30 öznitelik).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score,
                             precision_recall_curve, ConfusionMatrixDisplay)

veri = load_breast_cancer()
X = veri.data
y = 1 - veri.target     # orijinalde 1 = iyi huylu; biz 1 = kötü huylu (pozitif) yapıyoruz
print("Kötü huylu oranı:", y.mean().round(3))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(X_train)
X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
```

**Görev 1 — Kendi modeliniz vs. scikit-learn**

```python
bizim = LojistikRegresyon(alpha=0.1, epoch=3000).fit(X_train_s, y_train)
sk = LogisticRegression(max_iter=5000).fit(X_train_s, y_train)
for ad, m in [("Bizim", bizim), ("sklearn", sk)]:
    print(ad); print(classification_report(y_test, m.predict(X_test_s), digits=3))
```

**Görev 2 — Karmaşıklık matrisi**

```python
ConfusionMatrixDisplay.from_estimator(sk, X_test_s, y_test,
                                      display_labels=["İyi huylu", "Kötü huylu"])
plt.show()
```

**Görev 3 — Eşik analizi tablosu**

```python
p = sk.predict_proba(X_test_s)[:, 1]
satirlar = []
for esik in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    tahmin = (p >= esik).astype(int)
    m = metrikler(y_test, tahmin)
    tp, tn, fp, fn = karmasiklik_matrisi(y_test, tahmin)
    satirlar.append({"eşik": esik, **m, "FN (kaçan hasta)": fn, "FP (yanlış alarm)": fp})
print(pd.DataFrame(satirlar).round(3))
```

**Görev 4 — ROC ve PR eğrileri**

```python
fpr, tpr, esikler = roc_curve(y_test, p)
prec, rec, _ = precision_recall_curve(y_test, p)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, p):.3f}")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[0].set_title("ROC"); ax[0].legend()
ax[1].plot(rec, prec); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall"); plt.show()
```

**Görev 5 — Maliyete dayalı eşik seçimi:** Bir kötü huylu tümörü kaçırmanın (FN) maliyeti, gereksiz biyopsinin (FP) maliyetinin 10 katı olsun. `maliyet = 10·FN + 1·FP` değerini minimize eden eşiği bulun. 0.5 eşiğiyle karşılaştırın.

**Görev 6 — Yapay dengesizlik:** Eğitim setinde kötü huylu örneklerin yalnızca %10'unu tutarak dengesiz bir veri oluşturun. `class_weight=None` ve `class_weight="balanced"` ile recall değerlerini karşılaştırın.

---

## 9. Özet

- Lojistik regresyon: `p = σ(wᵀx + b)` — doğrusal skoru sigmoid ile olasılığa çevirir; karar sınırı doğrusaldır.
- Kayıp: **çapraz entropi**; gradyanı `(1/n)Xᵀ(p − y)`. Kapalı form yok, gradyan iniş kullanılır.
- Katsayılar log-odds üzerinden yorumlanır: e^w odds çarpanıdır.
- **Karmaşıklık matrisi** (TP, TN, FP, FN) tüm metriklerin temelidir.
- **Precision:** "pozitif dediklerimin ne kadarı doğru?" **Recall:** "gerçek pozitiflerin ne kadarını buldum?" **F1:** ikisinin harmonik ortalaması.
- **ROC-AUC** eşikten bağımsız sıralama başarısını ölçer; dengesiz veride **PR eğrisi** daha bilgilendiricidir.
- Dengesiz veride accuracy yanıltır; sınıf ağırlıkları, yeniden örnekleme ve eşik ayarlama kullanılır.

## Kendinizi Sınayın

1. z = −1 için σ(z)'yi hesaplayın. Model bu örnek için hangi sınıfı tahmin eder?
2. Gerçek sınıf 0 iken model p = 0.95 verirse çapraz entropi kaybı nedir?
3. TP = 30, FP = 20, FN = 10, TN = 940 için accuracy, precision, recall ve F1'i hesaplayın. Accuracy neden yanıltıcı?
4. Eşiği 0.5'ten 0.3'e indirirseniz precision ve recall nasıl değişir?
5. İki model düşünün: A'nın AUC'si 0.92, B'nin 0.85. A her eşikte B'den iyi midir?

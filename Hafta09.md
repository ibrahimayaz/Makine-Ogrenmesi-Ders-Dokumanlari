# Hafta 9 — Destek Vektör Makineleri (SVM)

## Bu Haftanın Hedefleri

- Maksimum marj fikrini ve neden iyi genelleme sağladığını anlamak
- Destek vektörlerinin ne olduğunu ve karar sınırını nasıl belirlediğini görmek
- Yumuşak marj ve C parametresinin yanlılık-varyans dengesine etkisini kavramak
- Hinge kaybı ile SVM'i gradyan inişle sıfırdan eğitmek
- **Kernel hilesinin** doğrusal olmayan sınırları nasıl mümkün kıldığını adım adım görmek
- `Pipeline` ve `GridSearchCV` ile sızıntısız hiperparametre araması yapmak

---

## 1. Hangi Doğru?

İki sınıfı ayıran bir doğru arıyoruz. Veri doğrusal ayrılabilirse **sonsuz sayıda** böyle doğru vardır:

```
 x₂           A          B          C
  │ ○ ○      ╱          │          ╲
  │○ ○ ○    ╱           │           ╲
  │ ○  ○   ╱   ●        │  ●         ╲  ●
  │       ╱  ● ● ●      │ ● ● ●       ╲● ● ●
  │      ╱    ● ●       │  ● ●         ╲ ● ●
  └──────────────── x₁
```

Üçü de eğitim verisini kusursuz ayırıyor. Ama A ve C bazı noktalara çok yakın geçiyor: Yeni bir nokta biraz kayarsa yanlış tarafa düşer. **B**, iki sınıfa da en uzak olan, yani en "güvenli" doğrudur.

**SVM'in fikri:** Sınıflar arasındaki **boşluğu (marjı) en büyük** yapan sınırı seç.

---

## 2. Maksimum Marjlı Sınıflandırıcı

### 2.1 Geometri

```
 x₂
  │  ○       ┆         ┆
  │    ○     ┆    ╱    ┆
  │  ○   ⊙ ─ ┆ ─ ╱ ─ ─ ┆     wᵀx + b = +1
  │      ○   ┆  ╱      ┆
  │         ─┆─╱───────┆─    wᵀx + b = 0   (karar sınırı)
  │          ┆╱  ⊛     ┆
  │         ╱┆ ─ ─ ─ ─ ┆─    wᵀx + b = −1
  │        ╱ ┆    ●  ● ┆
  │          ┆  ●      ┆
  └─────────────────────────── x₁
             ←─ marj ─→
     ⊙, ⊛ : destek vektörleri (marj sınırlarının tam üzerindeki noktalar)
```

- **Karar sınırı (hiperdüzlem):** wᵀx + b = 0
- **Karar kuralı:** ŷ = işaret(wᵀx + b), etiketler y ∈ {−1, +1}
- **Marj sınırları:** wᵀx + b = +1 ve wᵀx + b = −1
- **Marj genişliği:** 2 / ‖w‖

**Marj formülü nereden geliyor?** Bir x noktasının hiperdüzleme uzaklığı |wᵀx + b| / ‖w‖'dir. Marj sınırındaki noktalarda |wᵀx + b| = 1, yani uzaklık 1/‖w‖. İki taraf toplamı 2/‖w‖.

### 2.2 Optimizasyon problemi

Marjı büyütmek = ‖w‖'yi küçültmek. Tüm noktaların doğru tarafta ve marjın dışında kalması koşuluyla:

```
min   ½ ‖w‖²
w, b

koşul:  yᵢ (wᵀxᵢ + b) ≥ 1      her i için
```

`yᵢ(wᵀxᵢ + b) ≥ 1` koşulu iki durumu tek ifadede toplar:
- yᵢ = +1 ise wᵀxᵢ + b ≥ +1
- yᵢ = −1 ise wᵀxᵢ + b ≤ −1

Bu bir **konveks kuadratik programlama** problemidir → tek global çözüm vardır.

### 2.3 Destek vektörleri

Çözümde yalnızca **marj sınırı üzerindeki** noktalar (yᵢ(wᵀxᵢ + b) = 1) sınırın yerini belirler. Bunlara **destek vektörleri** denir.

**Çarpıcı sonuç:** Destek vektörü olmayan noktaları silseniz veya hareket ettirseniz (marjın dışında kaldıkları sürece) karar sınırı **hiç değişmez**. SVM, sınırı yalnızca "en zor" örneklere bakarak çizer.

### 2.4 Elle küçük bir örnek

Üç nokta:
- x₁ = (1, 1), y = −1
- x₂ = (3, 3), y = +1
- x₃ = (4, 1), y = +1

Sezgi: Negatif nokta (1,1). Pozitiflerden ona en yakın olanlar: (3,3) uzaklık √8 = 2.83; (4,1) uzaklık 3. En yakın pozitif (3,3) olduğu için sınır (1,1) ile (3,3) arasında, ortadan geçen ve bu doğruya dik bir hiperdüzlem olmalı. Orta nokta (2, 2), doğrultu (1, 1).

w = k·(1, 1) diyelim. Destek vektörleri için:
- (3,3): k·6 + b = +1
- (1,1): k·2 + b = −1

Çıkarırsak 4k = 2 → **k = 0.5**, **b = −2**. Yani **w = (0.5, 0.5), b = −2**.

Kontrol: (4,1) için wᵀx + b = 2.5 − 2 = 0.5 ... ≥ 1 koşulu **sağlanmıyor**! Demek ki (4,1) de marjı sıkıştırıyor; üç nokta birden destek vektörü olmalı. Çözümü yeniden kuralım:

```
(1,1): w₁ + w₂ + b = −1
(3,3): 3w₁ + 3w₂ + b = +1
(4,1): 4w₁ + w₂ + b = +1
```

İlk ikisinden: 2w₁ + 2w₂ = 2 → w₁ + w₂ = 1. İkinci ve üçüncüden: −w₁ + 2w₂ = 0 → w₁ = 2w₂. Buradan **w₂ = 1/3, w₁ = 2/3, b = −2**.

Marj = 2/‖w‖ = 2 / √(4/9 + 1/9) = 2 / 0.745 = **2.68**.

```python
import numpy as np
from sklearn.svm import SVC

X = np.array([[1, 1], [3, 3], [4, 1]])
y = np.array([-1, 1, 1])
svm = SVC(kernel="linear", C=1e6).fit(X, y)       # çok büyük C ≈ sert marj
print("w =", svm.coef_, " b =", svm.intercept_)   # w ≈ [[0.667 0.333]], b ≈ [-2.]
print("Destek vektörleri:\n", svm.support_vectors_)
print("Marj:", 2 / np.linalg.norm(svm.coef_))
```

> Bu örnek önemli bir dersi gösteriyor: Sezgimiz (en yakın iki nokta) yetmedi; optimizasyon tüm kısıtları birlikte dikkate alır.

---

## 3. Yumuşak Marj (Soft Margin)

### 3.1 Sorun

Gerçek veride:
- Sınıflar genelde **tam ayrılamaz** (örtüşme vardır) → sert marj probleminin çözümü yoktur.
- Tek bir **aykırı değer**, sert marjı dramatik biçimde daraltabilir.

```
 Sert marj (aykırıya kurban):         Yumuşak marj:
  ○ ○ ○    │ ●                          ○ ○ ○  ┆   ┆  ●
  ○ ○   ●  │● ●  ← tek aykırı ●          ○ ○   ●┆   ┆● ●
  ○ ○     ││ ● ●   sınırı itiyor        ○ ○    ┆   ┆ ● ●
         dar marj                        aykırıya izin, geniş marj
```

### 3.2 Gevşek değişkenler (slack variables)

Her nokta için bir **ihlal miktarı** ξᵢ ≥ 0 tanımlanır:

```
min   ½‖w‖² + C · Σ ξᵢ
w,b,ξ

koşul:  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0
```

| ξᵢ | Nokta nerede? |
|---|---|
| 0 | Doğru tarafta, marjın dışında ✓ |
| 0 < ξᵢ < 1 | Doğru tarafta ama marjın içinde |
| ξᵢ = 1 | Tam karar sınırında |
| ξᵢ > 1 | **Yanlış tarafta** (yanlış sınıflandırılmış) |

### 3.3 C parametresi

C, iki hedef arasındaki dengeyi kurar: **geniş marj** (½‖w‖²) ve **az ihlal** (Σξᵢ).

| C | Davranış | Marj | Risk |
|---|---|---|---|
| **Büyük** (ör. 1000) | İhlallere çok sert ceza → her noktayı doğru sınıflandırmaya çalışır | Dar | **Aşırı öğrenme** (yüksek varyans) |
| **Küçük** (ör. 0.01) | İhlallere tolerans → geniş marj | Geniş | **Eksik öğrenme** (yüksek yanlılık) |

> C, düzenlileştirme katsayısının **tersi** gibi davranır (lojistik regresyondaki C ile aynı mantık): C ↑ → düzenlileştirme ↓.

### 3.4 Hinge kaybı: SVM'e başka bir bakış

Kısıtlı problemi kısıtsız bir kayıp fonksiyonu olarak yazabiliriz:

```
L(w, b) = ½‖w‖² + C · Σ max(0, 1 − yᵢ(wᵀxᵢ + b))
                        └──────── hinge kaybı ────────┘
```

Tek bir örnek için m = y·(wᵀx + b) (işaretli skor, "marj değeri") olsun:

```
 kayıp
  3 │╲
  2 │ ╲                Hinge:    max(0, 1 − m)
  1 │  ╲               Log-loss: ln(1 + e^(−m))   (lojistik regresyon)
    │   ╲__
  0 │──────●────────── m = y·(wᵀx + b)
       0   1
```

- m ≥ 1: Doğru tarafta ve marjın dışında → **kayıp 0**. SVM bu noktaları tamamen umursamaz (bu yüzden sadece destek vektörleri önemlidir).
- m < 1: Doğrusal artan ceza.

Lojistik regresyonun log-loss'u ise hiçbir zaman tam 0 olmaz; her nokta bir miktar etki eder.

### 3.5 Sıfırdan doğrusal SVM (alt-gradyan inişi)

Hinge kaybı m = 1'de türevlenemez ama **alt-gradyan** kullanabiliriz. Kaybı n'ye bölünmüş biçimde (λ = 1/(nC)) yazarsak:

```
L = (λ/2)‖w‖² + (1/n) Σ max(0, 1 − yᵢ(wᵀxᵢ + b))

Her örnek için:
  eğer yᵢ(wᵀxᵢ + b) ≥ 1:   ∂/∂w = λw,              ∂/∂b = 0
  aksi halde:               ∂/∂w = λw − yᵢxᵢ,       ∂/∂b = −yᵢ
```

```python
class BenimLinearSVM:
    def __init__(self, lam=0.01, alpha=0.01, epoch=1000):
        self.lam, self.alpha, self.epoch = lam, alpha, epoch

    def fit(self, X, y):                 # y ∈ {-1, +1}
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epoch):
            m = y * (X @ self.w + self.b)
            ihlal = m < 1                                    # marjı ihlal edenler
            dw = self.lam * self.w - (y[ihlal, None] * X[ihlal]).sum(axis=0) / n
            db = -y[ihlal].sum() / n
            self.w -= self.alpha * dw
            self.b -= self.alpha * db
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
```

**Test edelim ve C'nin etkisini görelim:**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.6, random_state=4)
y = np.where(y == 0, -1, 1)

def svm_ciz(model, X, y, ax, baslik):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=["tab:blue", "tab:red"])
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=25)
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="k", lw=1.5)
    ax.set_title(baslik)

fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
svm_ciz(BenimLinearSVM(lam=0.01, alpha=0.05, epoch=3000).fit(X, y), X, y, ax[0], "Bizim SVM")
for a, C in zip(ax[1:], [0.01, 1, 100]):
    m = SVC(kernel="linear", C=C).fit(X, y)
    svm_ciz(m, X, y, a, f"sklearn, C = {C}  (destek vektörü: {len(m.support_)})")
plt.show()
```

Gözlem: C küçüldükçe marj genişler ve **destek vektörü sayısı artar** (marjın içine düşen tüm noktalar destek vektörüdür).

---

## 4. Kernel Hilesi (Kernel Trick)

### 4.1 Sorun: Doğrusal ayrılamayan veri

```
 1 boyut:   ● ● ○ ○ ○ ○ ● ●         Tek bir eşikle ayrılamaz!
           −3 −2 −1  0  1  2  3
```

**Fikir:** Veriyi **daha yüksek boyutlu bir uzaya** taşı; orada doğrusal ayrılabilir hale gelebilir.

φ(x) = (x, x²) dönüşümünü uygulayalım:

```
 x²
  9 │ ●                       ●
    │
  4 │    ●                 ●
    │ ─────────────────────────── x² = 2.5  ← artık düz bir çizgiyle ayrılır!
  1 │       ○           ○
  0 │            ○  ○
    └──────────────────────────── x
     −3  −2  −1  0  1  2  3
```

2 boyutta da benzer: İç içe iki halka (çember) şeklindeki veri, φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²) ile 3 boyuta taşınınca iç halka "aşağıda", dış halka "yukarıda" kalır ve bir düzlemle ayrılır.

### 4.2 Sorun: Yüksek boyut pahalıdır

d = 100 öznitelik için 2. derece tüm terimler ~5.000, 3. derece ~170.000 boyut üretir. RBF kernelinin karşılık geldiği uzay ise **sonsuz boyutludur**! Bu vektörleri açıkça hesaplamak imkânsız.

### 4.3 Hile: Sadece iç çarpımlar gerekir

SVM optimizasyon problemi (dual formu) ve tahmin fonksiyonu, veriyi **yalnızca iç çarpımlar** aracılığıyla kullanır:

```
f(x) = Σ  αᵢ yᵢ ⟨φ(xᵢ), φ(x)⟩ + b
     i∈SV
```

(αᵢ: dual problemin katsayıları; destek vektörü olmayanlar için αᵢ = 0.)

**Kernel fonksiyonu** K(x, z), φ'yi hiç hesaplamadan ⟨φ(x), φ(z)⟩ iç çarpımını **doğrudan** verir.

### 4.4 Kernel hilesini sayılarla doğrulayalım

2 boyutlu x = (x₁, x₂) için dönüşüm:

```
φ(x) = (x₁², √2·x₁x₂, x₂²)
```

x = (1, 2), z = (3, 4) olsun.

**Yol 1 — Açıkça dönüştür, sonra iç çarpım (3 boyutta):**

```
φ(x) = (1, √2·2, 4)  = (1, 2.828, 4)
φ(z) = (9, √2·12, 16) = (9, 16.971, 16)
⟨φ(x), φ(z)⟩ = 9 + 48 + 64 = 121
```

**Yol 2 — Kernel ile (2 boyutta kalarak):**

```
K(x, z) = (xᵀz)² = (1·3 + 2·4)² = 11² = 121  ✓
```

Aynı sonuç! Ama Yol 2'de 3 boyutlu vektörleri hiç oluşturmadık. Boyut ne kadar büyük olursa olsun kernel hesabının maliyeti orijinal boyutta kalır.

```python
x, z = np.array([1, 2]), np.array([3, 4])
phi = lambda v: np.array([v[0]**2, np.sqrt(2) * v[0] * v[1], v[1]**2])
print("Açık dönüşüm:", phi(x) @ phi(z))   # 121.0
print("Kernel      :", (x @ z) ** 2)      # 121
```

### 4.5 Yaygın kerneller

| Kernel | Formül | Parametreler | Ne zaman? |
|---|---|---|---|
| **Doğrusal** | K(x, z) = xᵀz | — | Çok öznitelik (metin), doğrusal ayrılabilir veri |
| **Polinom** | K(x, z) = (γ·xᵀz + r)^d | d (derece), γ, r | Öznitelik etkileşimleri önemliyse |
| **RBF (Gauss)** | K(x, z) = exp(−γ‖x − z‖²) | γ | Varsayılan seçim, çoğu durumda iyi çalışır |
| **Sigmoid** | K(x, z) = tanh(γ·xᵀz + r) | γ, r | Nadiren |

### 4.6 RBF kernelini anlamak

```
K(x, z) = exp(−γ · ‖x − z‖²)
```

RBF bir **benzerlik ölçüsüdür**:
- x = z → K = 1 (en benzer)
- x ve z uzaklaştıkça → K → 0

Örnek, γ = 0.5:

| ‖x − z‖ | ‖x − z‖² | K |
|---|---|---|
| 0 | 0 | 1.000 |
| 1 | 1 | e^(−0.5) = 0.607 |
| 2 | 4 | e^(−2) = 0.135 |
| 3 | 9 | e^(−4.5) = 0.011 |

Tahmin fonksiyonu f(x) = Σ αᵢyᵢ K(xᵢ, x) + b, her destek vektörünün etrafına bir **Gauss "tepesi"** koyup bunları toplamak gibidir. Pozitif destek vektörleri yukarı tepe, negatifler aşağı çukur oluşturur; f(x) = 0 seviye eğrisi karar sınırıdır.

**γ'nın etkisi** — her tepenin genişliğini belirler:

| γ | Tepe genişliği | Karar sınırı | Risk |
|---|---|---|---|
| **Küçük** (0.01) | Geniş: her destek vektörü uzaklara etki eder | Yumuşak, neredeyse doğrusal | Eksik öğrenme |
| **Büyük** (100) | Dar: her destek vektörü sadece kendi çevresini etkiler | Her noktanın etrafında "ada" | **Aşırı öğrenme** |

```
 γ küçük:                      γ büyük:
   ╭───────────╮                 ○ ○  ⬭     ⬭
   │  ○  ○  ○  │  ● ●            ○ ⬭ ○   ●⬭ ●
   │ ○  ○  ○   │ ●  ●  ●         ⬭  ○  ⬭  ●  ⬭●
   ╰───────────╯                 her noktanın etrafında küçük adacıklar
```

**C ve γ birlikte ayarlanmalıdır.** Tipik arama aralığı: C ∈ {0.1, 1, 10, 100}, γ ∈ {0.001, 0.01, 0.1, 1}.

### 4.7 Sıfırdan kernel tahmin fonksiyonu

Eğitilmiş bir scikit-learn SVM'inin içindeki bileşenleri (destek vektörleri, dual katsayılar αᵢyᵢ, b) kullanarak tahmini kendimiz hesaplayalım:

```python
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0)
gamma = 1.0
svm = SVC(kernel="rbf", gamma=gamma, C=1.0).fit(X, y)

def rbf(A, B, gamma):
    # Tüm çiftler için ‖a − b‖²
    kare_uzaklik = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * kare_uzaklik)

def bizim_karar(X_yeni):
    K = rbf(X_yeni, svm.support_vectors_, gamma)      # (n_yeni, n_sv)
    return K @ svm.dual_coef_.ravel() + svm.intercept_[0]

print(np.allclose(bizim_karar(X[:5]), svm.decision_function(X[:5])))   # True
print("Destek vektörü sayısı:", len(svm.support_), "/", len(X))
```

Tahmin için sadece destek vektörleri kullanılır; diğer noktalar modelden "silinebilir".

---

## 5. Pratik Bilgiler

### 5.1 Ölçekleme şarttır

SVM mesafe/iç çarpım tabanlıdır. Ölçeklenmemiş öznitelikler hem marjı hem RBF benzerliğini bozar. **Her zaman `StandardScaler` ile birlikte kullanın** — ve bunu Pipeline içinde yapın.

### 5.2 Çok sınıflı SVM

SVM doğası gereği ikili sınıflandırıcıdır. scikit-learn `SVC` çok sınıflı veride **One-vs-One** stratejisi kullanır: K sınıf için K(K−1)/2 ikili model eğitilir ve oylanır.

### 5.3 Olasılık tahmini

SVM doğal olarak olasılık değil, **mesafe skoru** (`decision_function`) üretir. `SVC(probability=True)` ek bir çapraz doğrulama ile (Platt ölçeklemesi) olasılık üretir ama eğitimi yavaşlatır.

### 5.4 Hesaplama maliyeti

Kernel SVM eğitimi kabaca **O(n²) – O(n³)** zaman alır. 10.000'e kadar örnekte rahat çalışır; 100.000+ örnekte çok yavaşlar. Büyük veride:
- Doğrusal problemler için `LinearSVC` veya `SGDClassifier(loss="hinge")`
- Kernel yaklaşımları (`RBFSampler`, `Nystroem`) + doğrusal model

### 5.5 SVM regresyonu (SVR)

Aynı fikir regresyona uygulanabilir: Tahminin etrafında genişliği ε olan bir "tüp" tanımlanır; **tüpün içindeki hatalar cezalandırılmaz**, dışındakiler doğrusal cezalandırılır.

```python
from sklearn.svm import SVR
SVR(kernel="rbf", C=10, epsilon=0.1)
```

### 5.6 SVM'in güçlü ve zayıf yönleri

| Güçlü | Zayıf |
|---|---|
| Yüksek boyutta etkili (d > n bile olabilir) | Büyük veride yavaş |
| Kernel ile esnek, doğrusal olmayan sınırlar | C ve γ ayarına duyarlı |
| Sadece destek vektörlerine bağlı → bellek verimli tahmin | Doğal olasılık çıktısı yok |
| Konveks optimizasyon → global çözüm | Kernel SVM'i yorumlamak zor |

---

## 6. Pipeline ve GridSearchCV

### 6.1 Hiperparametre araması neden Pipeline ile yapılmalı?

Hafta 2'den hatırlayın: Ölçekleyici **sadece eğitim verisinde** fit edilmeli. Çapraz doğrulamada her katın doğrulama kısmı da "test" gibi davranır. Ölçeklemeyi CV'den **önce** tüm eğitim setine uygularsak, her katta doğrulama verisinin istatistikleri sızmış olur.

```
YANLIŞ:                                    DOĞRU (Pipeline):
X_s = scaler.fit_transform(X_train)        pipe = Pipeline([scaler, svm])
GridSearchCV(SVC(), ...).fit(X_s, y)       GridSearchCV(pipe, ...).fit(X_train, y)
  → her katın doğrulama kısmı                → her katta scaler SADECE o katın
    ölçekleyicinin fit'ine katıldı             eğitim kısmında fit edilir
```

### 6.2 GridSearchCV nasıl çalışır?

```
Her parametre kombinasyonu için (ör. C=1, γ=0.1):
    Her CV katı için:
        Pipeline'ı katın eğitim kısmında fit et (scaler + SVM)
        Katın doğrulama kısmında skorla
    Ortalama CV skorunu kaydet
En iyi kombinasyonu seç → tüm eğitim setinde yeniden eğit (refit=True)
```

Toplam eğitim sayısı = (kombinasyon sayısı) × (kat sayısı). 4 C × 4 γ × 5 kat = **80 eğitim**.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

pipe = Pipeline([
    ("olcek", StandardScaler()),
    ("svm", SVC()),
])

# Parametre adları: <adım_adı>__<parametre>
param_izgara = [
    {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100],
     "svm__gamma": [0.001, 0.01, 0.1, 1]},
    {"svm__kernel": ["poly"], "svm__C": [0.1, 1, 10], "svm__degree": [2, 3]},
]
arama = GridSearchCV(pipe, param_izgara, cv=5, scoring="accuracy", n_jobs=-1)
```

**RandomizedSearchCV:** Parametre uzayı büyükse, rastgele örneklenmiş n kombinasyonu dener. Çoğu zaman grid search'e yakın sonucu çok daha hızlı verir.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
dagilim = {"svm__C": loguniform(1e-2, 1e3), "svm__gamma": loguniform(1e-4, 1e1)}
rs = RandomizedSearchCV(pipe, dagilim, n_iter=40, cv=5, random_state=0, n_jobs=-1)
```

**İç içe çapraz doğrulama (nested CV):** GridSearchCV'nin en iyi CV skoru, aynı veriyle hem seçim hem değerlendirme yapıldığı için **hafifçe iyimserdir**. Tarafsız tahmin için ayrı bir test seti ya da dış CV döngüsü gerekir:

```python
from sklearn.model_selection import cross_val_score
dis_skor = cross_val_score(arama, X, y, cv=5)   # her dış katta ayrı bir grid search
```

---

## 7. Laboratuvar: Karar Sınırı Görselleştirme; Pipeline ile GridSearchCV

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
```

**Görev 1 — Kernel karşılaştırması:** `make_moons` ve `make_circles` verilerinde doğrusal, polinom (d=3) ve RBF kernellerin karar sınırlarını çizin. Destek vektörlerini işaretleyin.

```python
veriler = {"Ay": make_moons(200, noise=0.2, random_state=0),
           "Halka": make_circles(200, noise=0.1, factor=0.4, random_state=0)}
kerneller = {"Doğrusal": SVC(kernel="linear", C=1),
             "Polinom (d=3)": SVC(kernel="poly", degree=3, C=1, coef0=1),
             "RBF": SVC(kernel="rbf", gamma=1, C=1)}

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for i, (vad, (X, y)) in enumerate(veriler.items()):
    for j, (kad, m) in enumerate(kerneller.items()):
        m.fit(X, y)
        svm_ciz(m, X, y, ax[i, j], f"{vad} – {kad} (doğruluk {m.score(X, y):.2f})")
plt.tight_layout(); plt.show()
```

**Görev 2 — C ve γ ızgarası:** Ay verisinde C ∈ {0.1, 1, 100} ve γ ∈ {0.1, 1, 10} için 3×3 karar sınırı ızgarası çizin. Hangi kombinasyonlar aşırı öğreniyor, hangileri eksik öğreniyor?

```python
X, y = make_moons(200, noise=0.25, random_state=1)
fig, ax = plt.subplots(3, 3, figsize=(14, 13))
for i, C in enumerate([0.1, 1, 100]):
    for j, g in enumerate([0.1, 1, 10]):
        m = SVC(kernel="rbf", C=C, gamma=g).fit(X, y)
        svm_ciz(m, X, y, ax[i, j], f"C={C}, γ={g}, SV={len(m.support_)}")
plt.tight_layout(); plt.show()
```

**Görev 3 — Kernel hilesini doğrulayın:** Bölüm 4.4'teki dönüşümü kullanarak `make_circles` verisini φ(x₁, x₂) = (x₁², √2·x₁x₂, x₂²) ile 3 boyuta taşıyın, 3B saçılım grafiği çizin ve bu uzayda **doğrusal** SVM eğitin. Doğruluğu, orijinal uzaydaki `SVC(kernel="poly", degree=2, coef0=0, gamma=1)` ile karşılaştırın.

```python
X, y = make_circles(300, noise=0.08, factor=0.4, random_state=0)
Phi = np.c_[X[:, 0]**2, np.sqrt(2) * X[:, 0] * X[:, 1], X[:, 1]**2]

fig = plt.figure(figsize=(7, 6)); ax3 = fig.add_subplot(projection="3d")
ax3.scatter(Phi[:, 0], Phi[:, 1], Phi[:, 2], c=y, cmap="coolwarm")
plt.show()

print("φ uzayında doğrusal SVM:", SVC(kernel="linear").fit(Phi, y).score(Phi, y))
print("Polinom kernel (d=2)   :", SVC(kernel="poly", degree=2, coef0=0, gamma=1).fit(X, y).score(X, y))
```

**Görev 4 — Gerçek veride GridSearchCV:** Meme kanseri verisinde:

```python
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

arama = GridSearchCV(pipe, param_izgara, cv=5, scoring="roc_auc", n_jobs=-1)
arama.fit(X_tr, y_tr)
print("En iyi parametreler:", arama.best_params_)
print("En iyi CV ROC-AUC  :", round(arama.best_score_, 4))
print("Test ROC-AUC       :", round(arama.score(X_te, y_te), 4))

# Sonuçları ısı haritası olarak göster (RBF kısmı)
sonuc = pd.DataFrame(arama.cv_results_)
rbf = sonuc[sonuc["param_svm__kernel"] == "rbf"]
tablo = rbf.pivot_table(index="param_svm__C", columns="param_svm__gamma",
                        values="mean_test_score")
import seaborn as sns
sns.heatmap(tablo, annot=True, fmt=".3f", cmap="viridis")
plt.title("RBF SVM: CV ROC-AUC"); plt.show()
```

**Görev 5 — Sızıntı deneyi:** Ölçeklemeyi Pipeline dışında, tüm veriye uygulayıp aynı aramayı yapın. CV skorları değişti mi? (Bu veri setinde fark küçük olabilir; neden küçük olduğunu tartışın. Fark ne zaman büyür?)

**Görev 6 — Lojistik regresyon ve Random Forest ile karşılaştırma:** Aynı CV katlarıyla üç modeli karşılaştıran bir tablo hazırlayın.

---

## 8. Özet

- SVM, sınıflar arasındaki **marjı en büyük** yapan hiperdüzlemi bulur: min ½‖w‖², koşul yᵢ(wᵀxᵢ + b) ≥ 1. Marj = 2/‖w‖.
- Sınırı yalnızca **destek vektörleri** belirler.
- **Yumuşak marj** ihlallere izin verir; **C** büyükse dar marj (aşırı öğrenme riski), küçükse geniş marj (eksik öğrenme riski).
- Eşdeğer görünüm: **Hinge kaybı** max(0, 1 − y·f(x)) + L2 düzenlileştirme.
- **Kernel hilesi:** Yüksek boyutlu dönüşümü hiç hesaplamadan K(x, z) = ⟨φ(x), φ(z)⟩ ile çalışmak.
- **RBF kernel:** Benzerlik ölçüsü; γ büyükse dar tepeler (aşırı öğrenme), küçükse yumuşak sınır.
- Ölçekleme şart; C ve γ, **Pipeline + GridSearchCV** ile sızıntısız biçimde aranmalı.

## Kendinizi Sınayın

1. Bir SVM w = (3, 4), b = −10 öğrenmiş. Marj genişliği nedir? x = (2, 2) hangi sınıfa atanır ve marjın içinde mi?
2. Destek vektörü olmayan bir eğitim örneğini veri setinden çıkarırsak karar sınırı değişir mi? Neden?
3. x = (1, 0) ve z = (0, 1) için RBF kernel değerini γ = 1 ile hesaplayın. Ya x = z = (1, 1) olsaydı?
4. Eğitim doğruluğu %100, CV doğruluğu %70 olan bir RBF SVM için C ve γ'yı nasıl değiştirirsiniz?
5. 4 C değeri, 5 γ değeri ve 10-katlı CV ile GridSearchCV kaç model eğitir (son refit dahil)?

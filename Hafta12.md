# Hafta 12 — Yapay Sinir Ağlarına Giriş

## Bu Haftanın Hedefleri

- Perceptron'un nasıl öğrendiğini elle çalıştırmak ve sınırlamasını (XOR) görmek
- Çok katmanlı ağların (MLP) yapısını ve ileri yayılımı (forward pass) anlamak
- Aktivasyon fonksiyonlarının rolünü ve farklarını kavramak
- **Geri yayılım** (backpropagation) algoritmasını küçük bir ağda sayılarla adım adım hesaplamak
- NumPy ile sıfırdan bir MLP yazmak
- SGD, momentum ve Adam optimizasyon algoritmalarını; dropout ve diğer düzenlileştirme tekniklerini öğrenmek
- PyTorch ile bir MLP eğitmek

---

## 1. Biyolojik İlham ve Yapay Nöron

Beyindeki bir nöron, dendritlerinden sinyaller alır, bunları toplar ve toplam bir eşiği aşarsa aksonu üzerinden sinyal gönderir ("ateşler").

Yapay nöron bu fikrin basit bir matematiksel modelidir:

```
  x₁ ──w₁──╮
            │
  x₂ ──w₂──┼──► Σ  z = w₁x₁ + w₂x₂ + w₃x₃ + b ──► f(z) ──► çıktı
            │
  x₃ ──w₃──╯        (ağırlıklı toplam)        (aktivasyon)
            b (bias)
```

1. Girdiler ağırlıklarla çarpılıp toplanır, bias eklenir: **z = wᵀx + b**
2. Sonuç bir **aktivasyon fonksiyonundan** geçer: **a = f(z)**

Tanıdık geliyor mu? f = sigmoid seçilirse bu **lojistik regresyonun ta kendisidir** (Hafta 5). Sinir ağları, bu basit birimlerin **katmanlar halinde** birbirine bağlanmasıyla oluşur.

---

## 2. Perceptron

### 2.1 Model

Rosenblatt'ın (1958) perceptronu, aktivasyon olarak **basamak fonksiyonu** kullanır:

```
ŷ = 1   eğer wᵀx + b > 0
ŷ = 0   aksi halde
```

### 2.2 Öğrenme kuralı

Her örnek için:

```
w ← w + η · (y − ŷ) · x
b ← b + η · (y − ŷ)
```

| Durum | y − ŷ | Etki |
|---|---|---|
| Doğru tahmin | 0 | Değişiklik yok |
| y = 1, ŷ = 0 (pozitifi kaçırdı) | +1 | w'ye x eklenir → z bir dahaki sefere büyür |
| y = 0, ŷ = 1 (yanlış alarm) | −1 | w'den x çıkarılır → z küçülür |

### 2.3 Elle örnek: VE (AND) kapısı

| x₁ | x₂ | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Başlangıç: w = (0, 0), b = 0, η = 1

**Epoch 1:**

| Örnek | z = wᵀx + b | ŷ | y | y − ŷ | Yeni w | Yeni b |
|---|---|---|---|---|---|---|
| (0,0) | 0 | 0 | 0 | 0 | (0, 0) | 0 |
| (0,1) | 0 | 0 | 0 | 0 | (0, 0) | 0 |
| (1,0) | 0 | 0 | 0 | 0 | (0, 0) | 0 |
| (1,1) | 0 | 0 | 1 | **+1** | (1, 1) | 1 |

**Epoch 2:**

| Örnek | z | ŷ | y | y − ŷ | Yeni w | Yeni b |
|---|---|---|---|---|---|---|
| (0,0) | 1 | 1 | 0 | **−1** | (1, 1) | 0 |
| (0,1) | 1 | 1 | 0 | **−1** | (1, 0) | −1 |
| (1,0) | 0 | 0 | 0 | 0 | (1, 0) | −1 |
| (1,1) | 0 | 0 | 1 | **+1** | (2, 1) | 0 |

Bu şekilde devam edildiğinde perceptron **5. epoch'ta** hatasız hale gelir ve **w = (2, 1), b = −2** bulunur. Kontrol: 2x₁ + x₂ − 2 > 0 yalnızca (1,1) için sağlanır (2 + 1 − 2 = 1) ✓

**Perceptron yakınsama teoremi:** Veri doğrusal olarak ayrılabiliyorsa, perceptron sonlu sayıda adımda ayıran bir hiperdüzlem bulur.

```python
import numpy as np

class Perceptron:
    def __init__(self, eta=1.0, epoch=20):
        self.eta, self.epoch = eta, epoch

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for e in range(self.epoch):
            hata = 0
            for xi, yi in zip(X, y):
                y_hat = int(xi @ self.w + self.b > 0)
                guncelleme = self.eta * (yi - y_hat)
                self.w += guncelleme * xi
                self.b += guncelleme
                hata += int(guncelleme != 0)
            print(f"Epoch {e + 1}: hata = {hata}, w = {self.w}, b = {self.b}")
            if hata == 0:
                break
        return self

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Perceptron().fit(X, np.array([0, 0, 0, 1]))      # VE: yakınsar
Perceptron().fit(X, np.array([0, 1, 1, 0]))      # XOR: asla yakınsamaz!
```

### 2.4 XOR problemi ve perceptronun sınırı

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```
 x₂
  1 │ ●(0,1)=1      ○(1,1)=0
    │
  0 │ ○(0,0)=0      ●(1,0)=1
    └─────────────────────── x₁
       0              1
```

**Tek bir düz çizgiyle** ● ve ○ noktalarını ayırmak imkânsızdır. Tek katmanlı perceptron XOR'u öğrenemez. (Minsky ve Papert, 1969 — bu gözlem "ilk yapay zekâ kışına" katkıda bulundu.)

**Çözüm:** Birden fazla katman! XOR = (x₁ VEYA x₂) VE DEĞİL (x₁ VE x₂). İki gizli nöron VEYA ve VE'yi hesaplar, çıktı nöronu bunları birleştirir:

```
  x₁ ──┬──► [h₁ = VEYA(x₁,x₂)] ──(+1)──╮
       │╲ ╱                              ├──► [çıktı = h₁ VE DEĞİL h₂]
       │ ╳                               │
       │╱ ╲                              │
  x₂ ──┴──► [h₂ = VE(x₁,x₂)]  ──(−2)──╯
```

Gizli katman, girdi uzayını XOR'un **doğrusal ayrılabilir** olduğu yeni bir uzaya dönüştürür: (h₁, h₂) uzayında noktalar (0,0)→0, (1,0)→1, (1,0)→1, (1,1)→0 olur ve bir doğruyla ayrılır. Bu, sinir ağlarının temel gücüdür: **Öznitelik dönüşümlerini kendileri öğrenirler.**

---

## 3. Çok Katmanlı Perceptron (MLP)

### 3.1 Yapı

```
 Girdi katmanı     Gizli katman 1    Gizli katman 2     Çıktı katmanı
   (d nöron)         (h₁ nöron)        (h₂ nöron)        (K nöron)

     x₁ ───────────── ● ─────────────── ● ─────────────── ŷ₁
         ╲  ╱ ╲  ╱       ╲  ╱ ╲  ╱         ╲  ╱
     x₂ ──╳────╳──── ● ───╳────╳──── ● ───╳──── ŷ₂
         ╱  ╲ ╱  ╲       ╱  ╲ ╱  ╲         ╱  ╲
     x₃ ───────────── ● ─────────────── ● ─────────────── ŷ₃

        W⁽¹⁾, b⁽¹⁾          W⁽²⁾, b⁽²⁾          W⁽³⁾, b⁽³⁾
```

- **Tam bağlı (dense, fully connected):** Her nöron bir önceki katmandaki tüm nöronlara bağlıdır.
- Katman l'deki parametre sayısı: (önceki katman nöron sayısı × bu katman nöron sayısı) + bu katman nöron sayısı (bias).

**Örnek:** 784 → 128 → 64 → 10 ağı:
- 784·128 + 128 = 100.480
- 128·64 + 64 = 8.256
- 64·10 + 10 = 650
- **Toplam: 109.386 parametre**

### 3.2 İleri yayılım (Forward pass)

Her katmanda iki işlem:

```
z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾        (doğrusal kısım)
a⁽ˡ⁾ = f(z⁽ˡ⁾)                     (aktivasyon)
```

a⁽⁰⁾ = x (girdi), son katmanın çıktısı ŷ'dir.

### 3.3 Neden aktivasyon fonksiyonu şart?

Aktivasyon olmasaydı:

```
ŷ = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾ = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾) = W'x + b'
```

Kaç katman olursa olsun sonuç **tek bir doğrusal model** olur! Doğrusal olmayan aktivasyonlar, ağın karmaşık fonksiyonları öğrenmesini sağlar.

**Evrensel yaklaşım teoremi:** Yeterli sayıda nörona sahip tek bir gizli katmanlı ağ, herhangi bir sürekli fonksiyonu istenilen hassasiyetle yaklaşık olarak temsil edebilir. (Pratikte derin ağlar aynı işi çok daha az nöronla yapar.)

---

## 4. Aktivasyon Fonksiyonları

| Fonksiyon | Formül | Aralık | Türev | Not |
|---|---|---|---|---|
| **Sigmoid** | 1 / (1 + e⁻ᶻ) | (0, 1) | σ(z)(1 − σ(z)) | İkili çıktı katmanında; gizli katmanda artık nadiren |
| **Tanh** | (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ) | (−1, 1) | 1 − tanh²(z) | Sıfır merkezli; RNN'lerde |
| **ReLU** | max(0, z) | [0, ∞) | 1 (z > 0), 0 (z < 0) | **Gizli katmanlar için varsayılan** |
| **Leaky ReLU** | max(0.01z, z) | (−∞, ∞) | 1 veya 0.01 | "Ölü ReLU" sorununa çözüm |
| **GELU** | z·Φ(z) | ≈(−0.17, ∞) | — | Transformer'larda yaygın |
| **Softmax** | eᶻᵏ / Σⱼ eᶻʲ | (0, 1), toplam 1 | — | Çok sınıflı çıktı katmanı |

```
   Sigmoid              Tanh                 ReLU               Leaky ReLU
 1 │     ___          1 │     ___              │      ╱            │      ╱
   │   ╱                │   ╱                  │    ╱              │    ╱
.5 │  ╱               0 │──╱──────             │  ╱                │  ╱
   │╱                   │ ╱                    │╱                 ─┼─╱
 0 │──────            -1│___                ───┘───────          ╱ │
```

### 4.1 Kaybolan gradyan sorunu (vanishing gradient)

Sigmoid'in türevi en fazla **0.25**'tir (z = 0'da). Geri yayılımda gradyanlar katman katman çarpılır (Bölüm 5). 10 katmanlı bir ağda:

```
0.25¹⁰ ≈ 0.00000095
```

İlk katmanlara ulaşan gradyan neredeyse sıfırdır → bu katmanlar **öğrenemez**. Ayrıca |z| büyükken sigmoid doygunluğa ulaşır ve türevi ~0 olur.

**ReLU'nun avantajı:** z > 0 için türevi tam **1**'dir; gradyan küçülmeden geçer. Ayrıca hesaplaması çok ucuzdur.

**ReLU'nun dezavantajı — "ölü nöronlar":** Bir nöronun z değeri tüm girdiler için negatif olursa türevi hep 0 olur ve nöron bir daha asla güncellenmez. Leaky ReLU, negatif bölgede küçük bir eğim bırakarak bunu önler.

### 4.2 Çıktı katmanı ve kayıp fonksiyonu eşleşmesi

| Problem | Çıktı nöronu | Aktivasyon | Kayıp |
|---|---|---|---|
| Regresyon | 1 | Yok (doğrusal) | MSE |
| İkili sınıflandırma | 1 | Sigmoid | İkili çapraz entropi |
| Çok sınıflı (K sınıf) | K | Softmax | Kategorik çapraz entropi |

---

## 5. Geri Yayılım (Backpropagation)

### 5.1 Problem

Ağın binlerce/milyonlarca parametresi var. Gradyan iniş için **her parametrenin kayba etkisini** (∂L/∂w) bilmemiz gerekiyor. Her birini ayrı ayrı sayısal türevle hesaplamak çok yavaş olurdu.

**Geri yayılım**, zincir kuralını kullanarak tüm gradyanları **tek bir geri geçişte** verimli biçimde hesaplar.

### 5.2 Zincir kuralı hatırlatması

y = f(g(x)) ise:

```
dy/dx = (dy/dg) · (dg/dx)
```

Bir ağda kayıp, son katmandan başlayarak her parametreye bir zincir ile bağlıdır:

```
w⁽¹⁾ → z⁽¹⁾ → a⁽¹⁾ → z⁽²⁾ → a⁽²⁾ → L

∂L/∂w⁽¹⁾ = ∂L/∂a⁽²⁾ · ∂a⁽²⁾/∂z⁽²⁾ · ∂z⁽²⁾/∂a⁽¹⁾ · ∂a⁽¹⁾/∂z⁽¹⁾ · ∂z⁽¹⁾/∂w⁽¹⁾
           └──────────── δ⁽²⁾ ─────┘
           └──────────────────────────── δ⁽¹⁾ ──────────────────┘
```

**Anahtar fikir:** Çıktıdan geriye doğru giderken her katmandaki **hata sinyalini (δ)** hesapla ve bir önceki katmana aktar. Aynı ara sonuçlar tekrar tekrar kullanılır → verimli.

### 5.3 Genel formüller

```
Çıktı katmanı (sigmoid + BCE veya softmax + CE):
    δ⁽ᴸ⁾ = ŷ − y

Gizli katmanlar (geriye doğru):
    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ · δ⁽ˡ⁺¹⁾) ⊙ f'(z⁽ˡ⁾)        (⊙: eleman eleman çarpım)

Gradyanlar:
    ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · a⁽ˡ⁻¹⁾ᵀ
    ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### 5.4 Sayılarla tam bir örnek

**Ağ:** 2 girdi → 2 gizli nöron (sigmoid) → 1 çıktı (sigmoid). Kayıp: ikili çapraz entropi.

```
        x₁=1 ──0.1──► h₁ ──0.5──╮
             ╲   ╱               ├──► ŷ
              ╳  (−0.3)          │
             ╱   ╲               │
        x₂=2 ──0.2─► h₂ ─(−0.4)─╯
                0.1
```

**Parametreler:**
- h₁: ağırlıklar (0.1, 0.2), bias 0
- h₂: ağırlıklar (−0.3, 0.1), bias 0
- Çıktı: ağırlıklar v = (0.5, −0.4), bias c = 0
- Girdi x = (1, 2), gerçek etiket y = 1, öğrenme oranı η = 0.5

#### İleri yayılım

```
z₁ = 0.1·1 + 0.2·2 + 0     = 0.5       h₁ = σ(0.5)  = 0.6225
z₂ = −0.3·1 + 0.1·2 + 0    = −0.1      h₂ = σ(−0.1) = 0.4750

z₃ = 0.5·0.6225 + (−0.4)·0.4750 + 0 = 0.3113 − 0.1900 = 0.1213
ŷ  = σ(0.1213) = 0.5303

L = −ln(0.5303) = 0.6343
```

#### Geri yayılım

**Adım 1 — Çıktı katmanı hatası:**

```
δ₃ = ŷ − y = 0.5303 − 1 = −0.4697
```

(Negatif: tahmin olması gerekenden küçük → çıktıyı artıracak yönde güncelleme yapılacak.)

**Adım 2 — Çıktı katmanı gradyanları:**

```
∂L/∂v₁ = δ₃ · h₁ = −0.4697 · 0.6225 = −0.2924
∂L/∂v₂ = δ₃ · h₂ = −0.4697 · 0.4750 = −0.2231
∂L/∂c  = δ₃      = −0.4697
```

**Adım 3 — Hatayı gizli katmana geri yay:**

```
δ₁ = δ₃ · v₁ · σ'(z₁) = −0.4697 · 0.5    · (0.6225 · 0.3775) = −0.2349 · 0.2350 = −0.0552
δ₂ = δ₃ · v₂ · σ'(z₂) = −0.4697 · (−0.4) · (0.4750 · 0.5250) =  0.1879 · 0.2494 =  0.0469
```

Yorum: h₁ çıktıya pozitif bağlı (v₁ > 0); çıktıyı artırmak için h₁ artmalı → δ₁ negatif (gradyan inişte artış yönü). h₂ negatif bağlı (v₂ < 0); çıktıyı artırmak için h₂ **azalmalı** → δ₂ pozitif.

**Adım 4 — Gizli katman gradyanları:**

```
∂L/∂w(h₁) = δ₁ · x = −0.0552 · (1, 2) = (−0.0552, −0.1104)
∂L/∂w(h₂) = δ₂ · x =  0.0469 · (1, 2) = ( 0.0469,  0.0937)
∂L/∂b₁ = −0.0552,    ∂L/∂b₂ = 0.0469
```

#### Güncelleme (η = 0.5)

| Parametre | Eski | Gradyan | Yeni = Eski − 0.5·Gradyan |
|---|---|---|---|
| v₁ | 0.5 | −0.2924 | **0.6462** |
| v₂ | −0.4 | −0.2231 | **−0.2884** |
| c | 0 | −0.4697 | **0.2349** |
| w(h₁) | (0.1, 0.2) | (−0.0552, −0.1104) | **(0.1276, 0.2552)** |
| w(h₂) | (−0.3, 0.1) | (0.0469, 0.0937) | **(−0.3234, 0.0531)** |
| b₁ | 0 | −0.0552 | **0.0276** |
| b₂ | 0 | 0.0469 | **−0.0234** |

#### Doğrulama: Yeni ileri yayılım

```
z₁ = 0.1276 + 0.5104 + 0.0276 = 0.6656    h₁ = 0.6605
z₂ = −0.3234 + 0.1062 − 0.0234 = −0.2406  h₂ = 0.4401
z₃ = 0.6462·0.6605 − 0.2884·0.4401 + 0.2349 = 0.4268 − 0.1269 + 0.2349 = 0.5348
ŷ  = σ(0.5348) = 0.6306
L  = −ln(0.6306) = 0.4611     (önceki: 0.6343)  ✓ kayıp azaldı
```

**Kodla doğrulayalım:**

```python
def sigmoid(z): return 1 / (1 + np.exp(-z))

x = np.array([1.0, 2.0]); y = 1.0
W1 = np.array([[0.1, 0.2], [-0.3, 0.1]]); b1 = np.zeros(2)
v = np.array([0.5, -0.4]); c = 0.0
eta = 0.5

for adim in range(3):
    # İleri
    z_h = W1 @ x + b1;  h = sigmoid(z_h)
    y_hat = sigmoid(v @ h + c)
    L = -np.log(y_hat)
    # Geri
    d3 = y_hat - y
    dv, dc = d3 * h, d3
    d_h = d3 * v * h * (1 - h)
    dW1, db1 = np.outer(d_h, x), d_h
    # Güncelle
    v -= eta * dv; c -= eta * dc; W1 -= eta * dW1; b1 -= eta * db1
    print(f"Adım {adim}: ŷ = {y_hat:.4f}, kayıp = {L:.4f}")
```

### 5.5 Gradyan kontrolü (gradient checking)

Geri yayılım kodunuzun doğru olduğunu **sayısal türevle** kontrol edebilirsiniz:

```
∂L/∂w ≈ [L(w + ε) − L(w − ε)] / 2ε        (ε ≈ 10⁻⁵)
```

İki değer birbirine çok yakın (~10⁻⁷ göreli fark) olmalıdır. Bu yöntem yavaştır, sadece hata ayıklamada kullanılır.

### 5.6 Otomatik türev (autograd)

Modern kütüphaneler (PyTorch, TensorFlow/Keras, JAX) ileri geçişte bir **hesaplama grafı** oluşturur ve `backward()` çağrıldığında yukarıdaki zincir kuralını **otomatik** uygular. Yine de mekanizmayı bilmek; kaybolan gradyan gibi sorunları anlamak ve hata ayıklamak için şarttır.

---

## 6. Sıfırdan MLP (NumPy)

```python
class BenimMLP:
    """Gizli katmanlarda ReLU, çıktıda softmax, kayıp: çapraz entropi."""

    def __init__(self, katmanlar, eta=0.1, seed=0):
        rng = np.random.default_rng(seed)
        # He başlatma: ReLU için uygun ölçek (Bölüm 7.4)
        self.W = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
                  for n_in, n_out in zip(katmanlar[:-1], katmanlar[1:])]
        self.b = [np.zeros(n_out) for n_out in katmanlar[1:]]
        self.eta = eta

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)       # sayısal kararlılık
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def _ileri(self, X):
        self.A, self.Z = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = self.A[-1] @ W + b
            self.Z.append(Z)
            son_katman = (i == len(self.W) - 1)
            self.A.append(self._softmax(Z) if son_katman else np.maximum(0, Z))
        return self.A[-1]

    def _geri(self, Y_onehot):
        n = len(Y_onehot)
        delta = (self.A[-1] - Y_onehot) / n               # softmax + CE: ŷ − y
        for l in reversed(range(len(self.W))):
            dW = self.A[l].T @ delta
            db = delta.sum(axis=0)
            if l > 0:                                     # bir önceki katmana yay
                delta = (delta @ self.W[l].T) * (self.Z[l - 1] > 0)   # ReLU türevi
            self.W[l] -= self.eta * dW
            self.b[l] -= self.eta * db

    def fit(self, X, y, epoch=100, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        K = len(np.unique(y))
        Y = np.eye(K)[y]                                  # one-hot
        self.kayiplar = []
        for _ in range(epoch):
            sira = rng.permutation(len(X))
            for bas in range(0, len(X), batch):           # mini-batch SGD
                idx = sira[bas:bas + batch]
                self._ileri(X[idx]); self._geri(Y[idx])
            P = self._ileri(X)
            self.kayiplar.append(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
        return self

    def predict_proba(self, X): return self._ileri(X)
    def predict(self, X): return self._ileri(X).argmax(axis=1)
```

**XOR'u çözelim:**

```python
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
mlp = BenimMLP([2, 8, 2], eta=0.5).fit(X_xor, y_xor, epoch=500, batch=4)
print(mlp.predict(X_xor))      # [0 1 1 0]
```

**Doğrusal olmayan bir karar sınırı öğrenme:**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, y = make_moons(500, noise=0.2, random_state=0)
X = StandardScaler().fit_transform(X)
mlp = BenimMLP([2, 32, 32, 2], eta=0.1).fit(X, y, epoch=200)

xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 300), np.linspace(-2.5, 2.5, 300))
P = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].contourf(xx, yy, P, levels=20, cmap="RdBu_r", alpha=0.7)
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=15)
ax[0].set_title(f"Doğruluk: {(mlp.predict(X) == y).mean():.3f}")
ax[1].plot(mlp.kayiplar); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Kayıp")
plt.show()
```

---

## 7. Optimizasyon Algoritmaları

### 7.1 SGD ve mini-batch (hatırlatma)

```
w ← w − η · ∇L(w)       (gradyan, bir mini-batch üzerinden hesaplanır)
```

Sinir ağlarının kayıp yüzeyi **konveks değildir**: yerel minimumlar, eyer noktaları ve düz platolar içerir. Mini-batch gürültüsü, bu tuzaklardan kaçmaya yardım eder.

**Terimler:**
- **Batch (yığın) boyutu:** Bir güncellemede kullanılan örnek sayısı (32, 64, 128, ...)
- **İterasyon (adım):** Bir güncelleme
- **Epoch:** Tüm eğitim verisinin bir kez görülmesi. 60.000 örnek, batch 100 → 1 epoch = 600 iterasyon

### 7.2 Momentum

Sorun: Dar vadilerde SGD zigzag çizer (Hafta 3'teki ölçeklenmemiş veri gibi).

Çözüm: Geçmiş gradyanların **hareketli ortalamasını** tut; bir top gibi "ivme kazan":

```
v ← β·v + ∇L(w)          (β ≈ 0.9)
w ← w − η·v
```

- Tutarlı yönlerde hız birikir → daha hızlı ilerleme
- Zigzag yönlerinde zıt işaretli gradyanlar birbirini götürür → salınım azalır

```
 SGD:  ╲╱╲╱╲╱╲╱╲──►●            Momentum:  ╲__________►●
       (zigzag)                            (yumuşak, hızlı)
```

### 7.3 Adam (Adaptive Moment Estimation)

Momentum + **her parametre için ayrı öğrenme oranı**:

```
m ← β₁·m + (1 − β₁)·g             (gradyanın ortalaması — 1. moment, momentum)
s ← β₂·s + (1 − β₂)·g²            (gradyan karesinin ortalaması — 2. moment)

m̂ = m / (1 − β₁ᵗ)                 (başlangıç sapması düzeltmesi)
ŝ = s / (1 − β₂ᵗ)

w ← w − η · m̂ / (√ŝ + ε)
```

Varsayılanlar: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸

**Sezgi:** Gradyanı sürekli büyük olan parametrelerde √ŝ büyük → adım küçülür. Seyrek/küçük gradyanlı parametrelerde adım büyür. Her parametre kendi ölçeğine uyarlanır.

```python
class Adam:
    def __init__(self, eta=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.eta, self.b1, self.b2, self.eps = eta, b1, b2, eps
        self.m, self.s, self.t = None, None, 0

    def adim(self, w, g):
        if self.m is None:
            self.m, self.s = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.s = self.b2 * self.s + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        s_hat = self.s / (1 - self.b2 ** self.t)
        return w - self.eta * m_hat / (np.sqrt(s_hat) + self.eps)
```

**Optimizasyon yöntemlerini bir test fonksiyonunda karşılaştıralım** (uzun, dar bir vadi):

```python
def f(w):  return 0.5 * w[0]**2 + 10 * w[1]**2      # x yönünde düz, y yönünde dik
def grad(w): return np.array([w[0], 20 * w[1]])

def calistir(yontem, adim_sayisi=100):
    w = np.array([-8.0, 2.0]); yol = [w.copy()]; v = np.zeros(2); opt = Adam(eta=0.5)
    for _ in range(adim_sayisi):
        g = grad(w)
        if yontem == "SGD":        w = w - 0.09 * g
        elif yontem == "Momentum": v = 0.9 * v + g; w = w - 0.02 * v
        elif yontem == "Adam":     w = opt.adim(w, g)
        yol.append(w.copy())
    return np.array(yol)

xx, yy = np.meshgrid(np.linspace(-9, 9, 200), np.linspace(-3, 3, 200))
plt.contour(xx, yy, 0.5 * xx**2 + 10 * yy**2, levels=30, cmap="gray")
for y_ad in ["SGD", "Momentum", "Adam"]:
    yol = calistir(y_ad)
    plt.plot(yol[:, 0], yol[:, 1], "o-", ms=2, label=y_ad)
plt.legend(); plt.title("Optimizasyon yolları"); plt.show()
```

### 7.4 Ağırlık başlatma

Tüm ağırlıkları **0** ile başlatırsak her gizli nöron aynı şeyi hesaplar ve aynı gradyanı alır → hepsi sonsuza dek aynı kalır (simetri sorunu). Ağırlıklar **rastgele** başlatılmalıdır, ama doğru ölçekte:

- Çok büyük → aktivasyonlar patlar / sigmoid doygunlaşır
- Çok küçük → sinyal katmanlar boyunca söner

| Yöntem | Varyans | Aktivasyon |
|---|---|---|
| Xavier / Glorot | 2 / (n_in + n_out) | Sigmoid, tanh |
| He | 2 / n_in | ReLU |

### 7.5 Öğrenme oranı zamanlaması

Başta büyük adımlar, sonra ince ayar: Adım azaltma (step decay), üstel azaltma, kosinüs azaltma, **ısınma (warm-up)**.

---

## 8. Düzenlileştirme: Aşırı Öğrenmeyle Mücadele

Sinir ağları çok sayıda parametreye sahip olduğu için kolayca aşırı öğrenir.

### 8.1 Dropout

Eğitim sırasında her adımda, her nöronu **p olasılıkla rastgele kapat** (çıktısını 0 yap).

```
 Eğitim (adım 1):           Eğitim (adım 2):            Test:
  ● ─── ● ─── ●              ● ─── ✕     ●               ● ─── ● ─── ●
  ● ─── ✕     ●              ● ─── ● ─── ●               ● ─── ● ─── ●
  ● ─── ● ─── ●              ● ─── ● ─── ●               ● ─── ● ─── ●
  ✕     ● ─── ●              ● ─── ✕     ●               ● ─── ● ─── ●
                                                         (hepsi açık)
```

**Neden işe yarar?**
1. Nöronlar belirli diğer nöronlara **bağımlı olamaz** → daha sağlam, dağıtık öznitelikler öğrenilir.
2. Her adımda farklı bir "alt ağ" eğitilir; test zamanında tüm ağı kullanmak, binlerce alt ağın **topluluğu (ensemble)** gibidir (Hafta 8'deki bagging fikrine benzer).

**Ölçekleme (inverted dropout):** Eğitimde açık kalan nöronların çıktısı 1/(1 − p) ile çarpılır. Böylece test zamanında beklenen çıktı aynı kalır ve ek işlem gerekmez.

```python
def dropout(A, p, egitim=True, rng=np.random.default_rng()):
    if not egitim or p == 0:
        return A
    maske = (rng.random(A.shape) > p) / (1 - p)      # inverted dropout
    return A * maske

A = np.ones((1, 10))
print(dropout(A, p=0.5))   # yaklaşık yarısı 0, diğerleri 2.0 → ortalama ≈ 1
```

Tipik değerler: Tam bağlı katmanlarda p = 0.2–0.5.

> **Önemli:** Test/tahmin sırasında dropout **kapatılmalıdır**. PyTorch'ta `model.eval()`, Keras'ta otomatik.

### 8.2 Diğer teknikler

| Teknik | Fikir |
|---|---|
| **L2 düzenlileştirme (weight decay)** | Kayba λΣw² ekle (Hafta 4'teki Ridge ile aynı) |
| **Erken durdurma** | Doğrulama kaybı belirli sayıda epoch iyileşmezse dur, en iyi ağırlıkları geri yükle |
| **Batch normalization** | Her katmanın girdilerini mini-batch istatistikleriyle normalize et → daha hızlı ve kararlı eğitim |
| **Veri artırma** | Eğitim verisini çeşitlendir (görüntüde döndürme, kaydırma; Hafta 13) |

---

## 9. Laboratuvar: PyTorch ile MLP

**Veri:** Fashion-MNIST — 10 sınıf giyim ürünü (tişört, pantolon, ayakkabı...), 28×28 gri tonlamalı görüntüler, 60.000 eğitim + 10.000 test.

```bash
pip install torch torchvision
```

### 9.1 Veriyi hazırlama

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", cihaz)

donusum = transforms.Compose([
    transforms.ToTensor(),                        # [0, 255] → [0, 1], şekil (1, 28, 28)
    transforms.Normalize((0.2860,), (0.3530,)),   # Fashion-MNIST ortalama/std
])
egitim_tam = datasets.FashionMNIST("veri", train=True, download=True, transform=donusum)
test_veri = datasets.FashionMNIST("veri", train=False, download=True, transform=donusum)
egitim_veri, dogr_veri = random_split(egitim_tam, [50000, 10000])

egitim_yukleyici = DataLoader(egitim_veri, batch_size=128, shuffle=True)
dogr_yukleyici = DataLoader(dogr_veri, batch_size=512)
test_yukleyici = DataLoader(test_veri, batch_size=512)

siniflar = ["Tişört", "Pantolon", "Kazak", "Elbise", "Ceket",
            "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bot"]

goruntuler, etiketler = next(iter(egitim_yukleyici))
fig, ax = plt.subplots(2, 8, figsize=(14, 4))
for a, img, e in zip(ax.ravel(), goruntuler, etiketler):
    a.imshow(img.squeeze(), cmap="gray"); a.set_title(siniflar[e], fontsize=9); a.axis("off")
plt.show()
```

### 9.2 Modeli tanımlama

```python
class MLP(nn.Module):
    def __init__(self, gizli=(256, 128), dropout=0.3):
        super().__init__()
        self.ag = nn.Sequential(
            nn.Flatten(),                             # (1, 28, 28) → 784
            nn.Linear(784, gizli[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gizli[0], gizli[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gizli[1], 10),                  # softmax yok: CrossEntropyLoss içinde
        )

    def forward(self, x):
        return self.ag(x)

model = MLP().to(cihaz)
print(model)
print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))
```

> `nn.CrossEntropyLoss`, softmax ve log'u kendi içinde sayısal olarak kararlı biçimde uygular; bu yüzden modelin son katmanına softmax **eklenmez**. Model ham skorları (logit) döndürür.

### 9.3 Eğitim döngüsü

PyTorch'ta eğitim döngüsünü kendimiz yazarız. Her adımda Bölüm 5–7'de öğrendiğimiz işlemler gerçekleşir:

```python
def bir_epoch(model, yukleyici, kayip_fn, optimizer=None):
    egitim = optimizer is not None
    model.train() if egitim else model.eval()        # dropout açık / kapalı
    toplam_kayip, dogru, n = 0.0, 0, 0
    with torch.set_grad_enabled(egitim):
        for X, y in yukleyici:
            X, y = X.to(cihaz), y.to(cihaz)
            logit = model(X)                          # 1) ileri yayılım
            kayip = kayip_fn(logit, y)                # 2) kayıp
            if egitim:
                optimizer.zero_grad()                 # 3) eski gradyanları sıfırla
                kayip.backward()                      # 4) geri yayılım (autograd)
                optimizer.step()                      # 5) parametre güncelleme
            toplam_kayip += kayip.item() * len(y)
            dogru += (logit.argmax(1) == y).sum().item()
            n += len(y)
    return toplam_kayip / n, dogru / n

def egit(model, optimizer, epoch=15, sabir=3):
    kayip_fn = nn.CrossEntropyLoss()
    gecmis = {"egitim_kayip": [], "dogr_kayip": [], "egitim_dog": [], "dogr_dog": []}
    en_iyi, sayac = float("inf"), 0
    for e in range(epoch):
        ek, ed = bir_epoch(model, egitim_yukleyici, kayip_fn, optimizer)
        dk, dd = bir_epoch(model, dogr_yukleyici, kayip_fn)
        for k, v in zip(gecmis, [ek, dk, ed, dd]):
            gecmis[k].append(v)
        print(f"Epoch {e + 1:2d} | eğitim kayıp {ek:.4f} doğ {ed:.4f} | "
              f"doğrulama kayıp {dk:.4f} doğ {dd:.4f}")
        if dk < en_iyi:                                # erken durdurma
            en_iyi, sayac = dk, 0
            torch.save(model.state_dict(), "en_iyi_model.pt")
        else:
            sayac += 1
            if sayac >= sabir:
                print("Erken durdurma!"); break
    model.load_state_dict(torch.load("en_iyi_model.pt"))
    return gecmis

model = MLP().to(cihaz)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
gecmis = egit(model, optimizer)
```

### 9.4 Değerlendirme

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(gecmis["egitim_kayip"], label="Eğitim"); ax[0].plot(gecmis["dogr_kayip"], label="Doğrulama")
ax[0].set_title("Kayıp"); ax[0].legend()
ax[1].plot(gecmis["egitim_dog"], label="Eğitim"); ax[1].plot(gecmis["dogr_dog"], label="Doğrulama")
ax[1].set_title("Doğruluk"); ax[1].legend()
plt.show()

test_kayip, test_dog = bir_epoch(model, test_yukleyici, nn.CrossEntropyLoss())
print(f"Test doğruluğu: {test_dog:.4f}")      # ~%88–89 beklenir

from sklearn.metrics import ConfusionMatrixDisplay
model.eval()
tahminler, gercekler = [], []
with torch.no_grad():
    for X, y in test_yukleyici:
        tahminler.append(model(X.to(cihaz)).argmax(1).cpu()); gercekler.append(y)
ConfusionMatrixDisplay.from_predictions(torch.cat(gercekler), torch.cat(tahminler),
                                        display_labels=siniflar, xticks_rotation=60)
plt.show()
```

### 9.5 Deneyler

Her deneyde yalnızca bir şeyi değiştirin ve doğrulama doğruluğunu/eğrilerini karşılaştırın:

1. **Optimizer:** `SGD(lr=0.01)`, `SGD(lr=0.01, momentum=0.9)`, `Adam(lr=1e-3)`
2. **Öğrenme oranı:** Adam ile 1e-1, 1e-2, 1e-3, 1e-4
3. **Dropout:** 0.0, 0.3, 0.6 — eğitim ve doğrulama eğrileri arasındaki boşluk nasıl değişiyor?
4. **Aktivasyon:** ReLU yerine Sigmoid kullanın (10 katmanlı derin bir ağda farkı gözlemleyin — kaybolan gradyan)
5. **Mimari:** (64,), (256, 128), (512, 256, 128)
6. **Bonus:** Sıfırdan yazdığınız `BenimMLP`'yi aynı veride (görüntüleri 784'lük vektöre çevirerek) eğitin ve PyTorch sonucu ile karşılaştırın.

### 9.6 Alternatif: Keras ile aynı model

```python
from tensorflow import keras
(xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0

model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"), keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"), keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
gecmis = model.fit(xtr, ytr, epochs=15, batch_size=128, validation_split=1/6,
                   callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
print(model.evaluate(xte, yte))
```

Keras daha az kodla hızlı prototip sunar; PyTorch eğitim döngüsü üzerinde tam kontrol sağlar.

---

## 10. Özet

- **Nöron:** z = wᵀx + b, a = f(z). Sigmoid aktivasyonlu tek nöron = lojistik regresyon.
- **Perceptron** doğrusal ayrılabilir veride yakınsar ama **XOR**'u çözemez → gizli katmanlar gerekir.
- **MLP:** Katman katman z = Wa + b, a = f(z). Doğrusal olmayan aktivasyonlar olmadan derinlik anlamsızdır.
- **ReLU** gizli katmanlarda varsayılandır; sigmoid/tanh derin ağlarda **kaybolan gradyan** sorununa yol açar.
- **Geri yayılım:** Zincir kuralıyla hata sinyalini (δ) çıktıdan girdiye doğru yayar; tüm gradyanlar tek geçişte hesaplanır.
- **Optimizasyon:** SGD → Momentum (ivme) → Adam (ivme + parametre başına uyarlanır öğrenme oranı).
- **Düzenlileştirme:** Dropout, L2, erken durdurma, batch normalization.
- PyTorch eğitim döngüsü: `ileri → kayıp → zero_grad → backward → step`.

## Kendinizi Sınayın

1. Perceptron ile VEYA (OR) kapısını w = (0, 0), b = 0, η = 1 başlangıcıyla elle eğitin.
2. 100 → 50 → 20 → 3 ağında kaç parametre vardır?
3. Bölüm 5.4'teki örnekte gerçek etiket y = 0 olsaydı δ₃ ve ∂L/∂v₁ ne olurdu? Güncellemeden sonra ŷ artar mı azalır mı?
4. Neden tüm ağırlıkları sıfırla başlatamayız?
5. Eğitim doğruluğu %99, doğrulama doğruluğu %85 olan bir ağ için üç düzenlileştirme önerin.
6. Dropout test zamanında neden kapatılır? Kapatılmazsa ne olur?

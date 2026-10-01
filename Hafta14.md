# Hafta 14 — Uygulamada Makine Öğrenmesi ve Etik

## Bu Haftanın Hedefleri

- Ham veriden dağıtıma kadar **uçtan uca** bir ML pipeline'ı kurmak
- Modelleri doğru biçimde **kaydetmek, yüklemek** ve sürümlemek
- Bir modeli basit bir **web servisi (API)** olarak sunmak; dağıtım sonrası izleme kavramlarını öğrenmek
- **Açıklanabilirlik:** Permütasyon önemi ve **SHAP** değerlerinin nasıl hesaplandığını anlamak
- **Yanlılık ve adillik:** Kaynaklarını tanımak, adillik metriklerini hesaplamak ve azaltma yöntemlerini öğrenmek
- Dönem projesini sunmak

---

## 1. Uçtan Uca ML Pipeline

### 1.1 Gerçek dünyada bir ML sistemi

Hafta 1'deki iş akışını hatırlayalım. Gerçek bir sistemde **model kodu, toplam sistemin küçük bir parçasıdır**:

```
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  Veri    │──►│  Veri    │──►│ Öznitelik│──►│  Model   │──►│  Dağıtım │──►│  İzleme  │
 │ toplama  │   │doğrulama │   │ mühendis.│   │ eğitimi/ │   │  (API)   │   │ (drift,  │
 │          │   │ temizlik │   │          │   │ seçimi   │   │          │   │ perform.)│
 └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
       ▲                                                                          │
       └──────────────────── yeniden eğitim tetikleyicisi ◄────────────────────────┘
```

### 1.2 İyi bir pipeline'ın özellikleri

| İlke | Neden? | Nasıl? |
|---|---|---|
| **Tekrarlanabilirlik** | Aynı kod + veri = aynı model | `random_state`, sabit kütüphane sürümleri (`requirements.txt`), veri sürümleme |
| **Sızıntısızlık** | Gerçekçi performans tahmini | Tüm ön işleme `Pipeline` içinde (Hafta 2) |
| **Eğitim-sunum tutarlılığı** | Eğitimde ve üretimde aynı dönüşüm | Ön işleme + modeli **tek nesne** olarak kaydet |
| **Modülerlik** | Bileşenleri değiştirmek kolay olsun | `ColumnTransformer`, özel dönüştürücüler |
| **Test edilebilirlik** | Hataları erken yakala | Veri şeması kontrolleri, birim testleri |

> **Eğitim-sunum çarpıklığı (training-serving skew):** En yaygın üretim hatalarından biri. Örneğin eğitimde yaş sütunu medyanla doldurulmuş ama üretim kodunda 0 ile dolduruluyor. Çözüm: Ön işlemeyi modelle **birlikte** paketlemek.

### 1.3 Özel dönüştürücü (custom transformer) yazmak

Kendi öznitelik mühendisliği adımlarınızı Pipeline'a eklemek için scikit-learn arayüzüne uyan bir sınıf yazabilirsiniz:

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OznitelikUretici(BaseEstimator, TransformerMixin):
    """Adult veri seti için yeni öznitelikler türetir."""

    def fit(self, X, y=None):
        return self                                    # öğrenilecek bir şey yok

    def transform(self, X):
        X = X.copy()
        X["sermaye_net"] = X["capital-gain"] - X["capital-loss"]
        X["haftalik_tam_zamanli"] = (X["hours-per-week"] >= 40).astype(int)
        X["yas_grubu"] = pd.cut(X["age"], [0, 25, 35, 50, 65, 120],
                                labels=["<25", "25-35", "35-50", "50-65", "65+"]).astype(str)
        return X
```

### 1.4 Tam bir pipeline örneği

**Problem:** Bir kişinin yıllık gelirinin 50.000 $'ı aşıp aşmadığını tahmin etmek (Adult veri seti, Hafta 8).

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import loguniform, randint

# 1) Veri
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data.copy()
y = (adult.target == ">50K").astype(int)

# 2) Önce test setini ayır — en sona kadar dokunma!
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Ön işleme
sayisal = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
           "hours-per-week", "sermaye_net"]
kategorik = ["workclass", "education", "marital-status", "occupation",
             "relationship", "race", "sex", "native-country", "yas_grubu"]

on_isleme = ColumnTransformer([
    ("say", Pipeline([("doldur", SimpleImputer(strategy="median")),
                      ("olcek", StandardScaler())]), sayisal),
    ("kat", Pipeline([("doldur", SimpleImputer(strategy="most_frequent")),
                      ("kodla", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                              sparse_output=False))]), kategorik),
], remainder="drop")

pipe = Pipeline([
    ("oznitelik", OznitelikUretici()),
    ("on_isleme", on_isleme),
    ("model", HistGradientBoostingClassifier(random_state=42)),
])

# 4) Hiperparametre araması (sadece eğitim verisinde, CV ile)
dagilim = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_leaf_nodes": randint(15, 63),
    "model__min_samples_leaf": randint(10, 100),
    "model__l2_regularization": loguniform(1e-4, 10),
}
arama = RandomizedSearchCV(pipe, dagilim, n_iter=25, scoring="roc_auc",
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           random_state=42, n_jobs=-1)
arama.fit(X_tr, y_tr)
print("En iyi CV ROC-AUC:", round(arama.best_score_, 4))
print("En iyi parametreler:", arama.best_params_)

# 5) Son değerlendirme — test setinde BİR KEZ
en_iyi = arama.best_estimator_
p_test = en_iyi.predict_proba(X_te)[:, 1]
print("Test ROC-AUC:", round(roc_auc_score(y_te, p_test), 4))
print(classification_report(y_te, en_iyi.predict(X_te), digits=3))
```

---

## 2. Model Kaydetme ve Yükleme

### 2.1 scikit-learn: joblib

```python
import joblib, json, datetime, sklearn

joblib.dump(en_iyi, "gelir_modeli_v1.joblib")           # tüm pipeline tek dosyada

# Meta veriyi de kaydedin: ileride "bu model nedir?" sorusunun cevabı
meta = {
    "model_adi": "gelir_modeli",
    "surum": "1.0.0",
    "egitim_tarihi": datetime.date.today().isoformat(),
    "sklearn_surumu": sklearn.__version__,
    "girdi_sutunlari": list(X_tr.columns),
    "cv_roc_auc": round(arama.best_score_, 4),
    "test_roc_auc": round(roc_auc_score(y_te, p_test), 4),
    "parametreler": {k: (float(v) if isinstance(v, (int, float, np.number)) else str(v))
                     for k, v in arama.best_params_.items()},
}
with open("gelir_modeli_v1.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# Yükleme ve kullanım
model = joblib.load("gelir_modeli_v1.joblib")
yeni_kisi = X_te.iloc[[0]]
print(model.predict_proba(yeni_kisi))
```

**Dikkat edilmesi gerekenler:**
- **Güvenlik:** `pickle`/`joblib` dosyaları yüklenirken **kod çalıştırabilir**. Güvenmediğiniz kaynaklardan gelen model dosyalarını asla yüklemeyin. (Alternatif güvenli formatlar: `skops`, ONNX.)
- **Sürüm uyumluluğu:** Model, kaydedildiği scikit-learn sürümüyle yüklenmelidir. Sürümleri `requirements.txt`'de sabitleyin.
- **Özel sınıflar:** `OznitelikUretici` gibi özel sınıflar, yükleme yapılan ortamda da **içe aktarılabilir** (import) olmalıdır. Bu yüzden sınıfı bir modül dosyasına (ör. `ozellikler.py`) koyun ve eğitim not defterinde de `from ozellikler import OznitelikUretici` ile kullanın. Not defterinde tanımlanan bir sınıfla kaydedilen model, başka bir betikte yüklenemez.

### 2.2 PyTorch

```python
import torch
# Önerilen: sadece ağırlıkları (state_dict) kaydet
torch.save(model_nn.state_dict(), "model.pt")

# Yükleme: önce aynı mimariyi oluştur, sonra ağırlıkları yükle
model_nn = MLP()
model_nn.load_state_dict(torch.load("model.pt", weights_only=True))
model_nn.eval()                                   # dropout/batchnorm tahmin moduna
```

### 2.3 ONNX: Çerçeveden bağımsız format

ONNX (Open Neural Network Exchange), modelleri farklı dillerde (C++, C#, JavaScript) ve platformlarda (mobil, tarayıcı) çalıştırmayı sağlar. `skl2onnx` ve `torch.onnx.export` ile dönüştürülebilir.

### 2.4 Model sürümleme ve deney takibi

Hangi veriyle, hangi parametrelerle, hangi kod sürümüyle hangi sonucun alındığını kaydetmek için **MLflow**, **Weights & Biases** gibi araçlar kullanılır. Veri sürümleme için **DVC** yaygındır.

---

## 3. Dağıtıma Giriş (Deployment)

### 3.1 Sunum biçimleri

| Biçim | Açıklama | Örnek |
|---|---|---|
| **Toplu (batch)** | Periyodik olarak çok sayıda tahmin üretilir, veritabanına yazılır | Her gece tüm müşteriler için kayıp (churn) skoru |
| **Çevrimiçi (online / real-time)** | İstek geldiğinde anında tahmin (API) | Kredi başvurusu anında karar |
| **Akış (streaming)** | Olay akışı üzerinde sürekli tahmin | Kart işlemlerinde sahtekarlık tespiti |
| **Uç (edge)** | Model cihazın üzerinde çalışır | Telefonda yüz tanıma |

### 3.2 FastAPI ile basit bir tahmin servisi

```bash
pip install fastapi uvicorn
```

**`servis.py`:**

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from ozellikler import OznitelikUretici   # özel sınıf, joblib yüklemesi için gerekli

app = FastAPI(title="Gelir Tahmin Servisi", version="1.0.0")
model = joblib.load("gelir_modeli_v1.joblib")      # sunucu başlarken bir kez yükle

class Basvuru(BaseModel):
    age: int = Field(..., ge=17, le=100)
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(0, alias="capital-gain")
    capital_loss: int = Field(0, alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

@app.get("/saglik")
def saglik():
    return {"durum": "çalışıyor"}

@app.post("/tahmin")
def tahmin(b: Basvuru):
    df = pd.DataFrame([b.model_dump(by_alias=True)])
    olasilik = float(model.predict_proba(df)[0, 1])
    return {"olasilik_50K_ustu": round(olasilik, 4),
            "karar": ">50K" if olasilik >= 0.5 else "<=50K",
            "model_surumu": "1.0.0"}
```

```bash
uvicorn servis:app --reload
# Tarayıcıda http://127.0.0.1:8000/docs adresinde otomatik oluşturulan test arayüzü açılır
```

**İstemci tarafı:**

```python
import requests
ornek = {"age": 38, "workclass": "Private", "fnlwgt": 215646, "education": "Bachelors",
         "education-num": 13, "marital-status": "Married-civ-spouse",
         "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
         "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50,
         "native-country": "United-States"}
print(requests.post("http://127.0.0.1:8000/tahmin", json=ornek).json())
```

**Daha ileri adımlar:** Servisi **Docker** konteynerine paketlemek, bir bulut platformunda (AWS, GCP, Azure) çalıştırmak, yük dengeleme, kimlik doğrulama, istek kayıtları (logging). Hızlı prototip ve demo arayüzleri için **Streamlit** veya **Gradio** kullanılabilir.

### 3.3 Dağıtım sonrası: İzleme

Model dağıtıldıktan sonra iş bitmez. Dünya değişir, model eskir.

| Kayma türü | Ne değişir? | Örnek |
|---|---|---|
| **Veri kayması (data drift)** | Girdilerin dağılımı P(X) | Yeni bir pazara açılınca müşteri yaş dağılımı değişir |
| **Kavram kayması (concept drift)** | Girdi-çıktı ilişkisi P(y\|X) | Pandemiyle birlikte alışveriş alışkanlıklarının değişmesi; spam gönderenlerin taktik değiştirmesi |

**Ne izlenmeli?**
- Girdi özniteliklerinin dağılımları (eğitim verisiyle karşılaştırma)
- Tahmin dağılımı (ör. pozitif tahmin oranı aniden ikiye katlandı mı?)
- Gerçek etiketler geldikçe performans metrikleri
- Teknik metrikler: gecikme süresi, hata oranı

**Basit kayma testi: Popülasyon Kararlılık İndeksi (PSI)**

Özniteliği kutulara ayır; her kutuda eğitim oranı (e) ve üretim oranı (a) için:

```
PSI = Σ (aᵢ − eᵢ) · ln(aᵢ / eᵢ)
```

| PSI | Yorum |
|---|---|
| < 0.1 | Önemli değişim yok |
| 0.1 – 0.25 | Orta düzey kayma, incele |
| > 0.25 | Büyük kayma, yeniden eğitimi düşün |

**Elle örnek:** Yaş grupları

| Grup | Eğitim (e) | Üretim (a) | (a − e)·ln(a/e) |
|---|---|---|---|
| < 30 | 0.30 | 0.45 | 0.15 · ln(1.5) = 0.061 |
| 30–50 | 0.50 | 0.40 | −0.10 · ln(0.8) = 0.022 |
| > 50 | 0.20 | 0.15 | −0.05 · ln(0.75) = 0.014 |
| | | **PSI** | **0.097** → sınırda |

```python
def psi(egitim, uretim, kutu=10):
    kenarlar = np.unique(np.quantile(egitim, np.linspace(0, 1, kutu + 1)))
    e = np.histogram(egitim, kenarlar)[0] / len(egitim)
    a = np.histogram(np.clip(uretim, kenarlar[0], kenarlar[-1]), kenarlar)[0] / len(uretim)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))

rng = np.random.default_rng(0)
egitim_yas = rng.normal(40, 12, 10000)
print("Kayma yok :", round(psi(egitim_yas, rng.normal(40, 12, 5000)), 3))
print("Kayma var :", round(psi(egitim_yas, rng.normal(32, 10, 5000)), 3))
```

Alternatif: Sürekli değişkenler için iki örneklem **Kolmogorov-Smirnov testi** (`scipy.stats.ks_2samp`).

---

## 4. Açıklanabilirlik (Explainability)

### 4.1 Neden?

- **Güven:** Doktor, "model bu hastanın riskli olduğunu söylüyor" ifadesiyle yetinmez; **neden** sorusunun cevabını ister.
- **Hata ayıklama:** Model doğru cevaba yanlış nedenle mi ulaşıyor? (Ünlü örnek: Kurt/husky sınıflandırıcısının aslında arka plandaki **karı** öğrenmesi.)
- **Sızıntı tespiti:** Bir öznitelik şüpheli derecede önemliyse sızıntı olabilir (Hafta 2).
- **Yasal gereklilik:** Kredi reddi gibi kararlarda kişilerin bilgilendirilme hakkı (ör. KVKK, GDPR).
- **Adillik denetimi:** Model hassas bir özniteliğe (cinsiyet, ırk) ya da onun vekiline dayanıyor mu?

### 4.2 Yorumlanabilir modeller vs. sonradan açıklama

| Yaklaşım | Örnek | Not |
|---|---|---|
| **Doğası gereği yorumlanabilir** | Doğrusal/lojistik regresyon (katsayılar), sığ karar ağacı (kurallar) | Hafta 3, 5, 7 |
| **Sonradan (post-hoc) açıklama** | Permütasyon önemi, PDP, LIME, **SHAP** | Kara kutu modeller için |

Açıklamalar iki düzeyde olabilir:
- **Küresel (global):** "Model genel olarak hangi özniteliklere dayanıyor?"
- **Yerel (local):** "**Bu** kişi için karar neden böyle çıktı?"

### 4.3 Permütasyon önemi (permutation importance)

**Fikir:** Bir öznitelik gerçekten önemliyse, değerlerini **karıştırmak** (hedefle ilişkisini koparmak) model performansını düşürür.

```
1. Modelin doğrulama skorunu hesapla: S₀
2. Her j özniteliği için:
     a. j sütununu rastgele karıştır (diğer sütunlar aynı kalsın)
     b. Yeni skoru hesapla: Sⱼ
     c. Önem(j) = S₀ − Sⱼ
3. (Rastgeleliği azaltmak için birkaç kez tekrarlayıp ortalama al)
```

**Örnek:** S₀ = 0.92 (ROC-AUC)
- "education" karıştırılınca 0.86 → önem 0.06
- "fnlwgt" karıştırılınca 0.92 → önem ≈ 0 (model bu özniteliği kullanmıyor)

```python
def permutasyon_onemi(model, X, y, skor_fn, tekrar=5, seed=0):
    rng = np.random.default_rng(seed)
    temel = skor_fn(y, model.predict_proba(X)[:, 1])
    onem = {}
    for sutun in X.columns:
        dususler = []
        for _ in range(tekrar):
            X_k = X.copy()
            X_k[sutun] = rng.permutation(X_k[sutun].values)
            dususler.append(temel - skor_fn(y, model.predict_proba(X_k)[:, 1]))
        onem[sutun] = np.mean(dususler)
    return pd.Series(onem).sort_values(ascending=False)

onem = permutasyon_onemi(en_iyi, X_te.sample(3000, random_state=0),
                         y_te.loc[X_te.sample(3000, random_state=0).index], roc_auc_score)
print(onem.round(4))

# scikit-learn karşılığı:
from sklearn.inspection import permutation_importance
```

**Avantajları:** Her modele uygulanabilir, ham öznitelikler (one-hot öncesi) düzeyinde çalışır, **test/doğrulama verisi** üzerinde hesaplandığı için genelleme ile ilgili bilgi verir.
**Dikkat:** Birbiriyle çok ilişkili iki öznitelik varsa, biri karıştırılınca model diğerini kullanmaya devam eder → ikisinin önemi de düşük görünebilir.

### 4.4 SHAP (SHapley Additive exPlanations)

#### Oyun teorisinden gelen fikir

Bir takım projesinde 3 kişi birlikte 90 puan kazandı. Bu puanı **adil** biçimde nasıl paylaştırırız? Lloyd Shapley (Nobel 2012) şu yanıtı verdi: Her oyuncunun, **tüm olası katılım sıralarında** takıma kattığı **ortalama marjinal katkı**.

SHAP bu fikri ML'e uygular:
- **Oyun:** Tek bir örnek için modelin tahmini
- **Oyuncular:** Öznitelikler
- **Kazanç:** Tahminin, ortalama tahminden (temel değer) farkı

#### Elle hesaplama

İki öznitelikli bir model düşünelim: **Gelir (G)** ve **Borç (B)**. Bir başvurucu için:

| Bilinen öznitelikler | Modelin beklenen çıktısı (risk skoru) |
|---|---|
| Hiçbiri (∅) — temel değer | 50 |
| Sadece G | 70 |
| Sadece B | 60 |
| G ve B | 90 ← gerçek tahmin |

("Bilinmeyen" öznitelik için model, o özniteliğin veri setindeki değerleri üzerinden ortalama alınarak değerlendirilir.)

İki öznitelik için iki olası sıra vardır:

| Sıra | G'nin marjinal katkısı | B'nin marjinal katkısı |
|---|---|---|
| G önce, sonra B | v(G) − v(∅) = 70 − 50 = **20** | v(G,B) − v(G) = 90 − 70 = **20** |
| B önce, sonra G | v(G,B) − v(B) = 90 − 60 = **30** | v(B) − v(∅) = 60 − 50 = **10** |
| **Ortalama (SHAP değeri)** | **φ_G = 25** | **φ_B = 15** |

**Doğrulama:** Temel değer + SHAP değerleri = 50 + 25 + 15 = **90** = modelin tahmini ✓

Bu **toplanabilirlik (additivity)** SHAP'ın en önemli özelliğidir: Her tahmin, temel değerden başlayıp her özniteliğin katkısının eklenmesiyle tam olarak açıklanır.

```
 temel değer 50 ──[Gelir +25]──► 75 ──[Borç +15]──► 90 = tahmin
```

**Genel formül** (M öznitelik, S: j hariç bir öznitelik alt kümesi):

```
       Σ      |S|! (M − |S| − 1)!
φⱼ =       ───────────────────── · [ v(S ∪ {j}) − v(S) ]
     S⊆F∖{j}        M!
```

Ağırlık, j'nin S'deki özniteliklerden hemen sonra geldiği sıraların oranıdır.

**Sıfırdan (kesin) Shapley hesabı:**

```python
from itertools import combinations
from math import factorial

def shapley(oyuncular, v):
    M = len(oyuncular)
    phi = {}
    for j in oyuncular:
        digerleri = [o for o in oyuncular if o != j]
        toplam = 0.0
        for boyut in range(M):
            for S in combinations(digerleri, boyut):
                agirlik = factorial(len(S)) * factorial(M - len(S) - 1) / factorial(M)
                toplam += agirlik * (v(frozenset(S) | {j}) - v(frozenset(S)))
        phi[j] = toplam
    return phi

degerler = {frozenset(): 50, frozenset({"G"}): 70, frozenset({"B"}): 60,
            frozenset({"G", "B"}): 90}
print(shapley(["G", "B"], lambda S: degerler[frozenset(S)]))   # {'G': 25.0, 'B': 15.0}
```

Kesin hesap 2^M alt küme gerektirir → çok öznitelikte imkânsız. **SHAP kütüphanesi** verimli yaklaşımlar kullanır:
- **TreeExplainer:** Ağaç tabanlı modeller için kesin ve çok hızlı algoritma
- **KernelExplainer:** Her model için örnekleme tabanlı yaklaşım (yavaş)
- **LinearExplainer**, **DeepExplainer** (sinir ağları)

#### SHAP kütüphanesi ile

```python
# pip install shap
import shap
import matplotlib.pyplot as plt

# Pipeline'ı ikiye ayır: dönüşümler + model
donusturucu = Pipeline(en_iyi.steps[:-1])
agac_model = en_iyi.named_steps["model"]
X_ornek = X_te.sample(1000, random_state=0)
X_donusmus = pd.DataFrame(donusturucu.transform(X_ornek),
                          columns=donusturucu[-1].get_feature_names_out())

aciklayici = shap.TreeExplainer(agac_model)
shap_degerleri = aciklayici(X_donusmus)

# Küresel: hangi öznitelikler genel olarak önemli ve etkileri hangi yönde?
shap.plots.beeswarm(shap_degerleri, max_display=15)

# Yerel: tek bir kişinin tahmini nasıl oluştu?
shap.plots.waterfall(shap_degerleri[0], max_display=12)
```

**Grafikleri okuma:**
- **Beeswarm:** Her nokta bir kişi. Yatay eksen SHAP değeri (sağ → tahmini artırır). Renk özniteliğin değeri (kırmızı yüksek, mavi düşük). Örneğin "capital-gain" yüksekken (kırmızı) noktalar sağdaysa, yüksek sermaye kazancı >50K olasılığını artırıyor demektir.
- **Waterfall:** Temel değerden başlayıp her özniteliğin katkısının eklenerek bu kişinin tahminine ulaşılmasını gösterir (log-odds ölçeğinde).

> **Uyarı:** SHAP **modelin** davranışını açıklar, gerçek dünyadaki **nedenselliği** değil. "Evli olmak geliri artırır" sonucu çıkarılamaz; model sadece bu ilişkiyi kullanıyor.

---

## 5. Yanlılık ve Adillik (Bias and Fairness)

### 5.1 Gerçek dünyadan örnekler

- **İşe alım:** Büyük bir teknoloji şirketinin özgeçmiş tarama aracının, geçmiş (çoğunluğu erkeklerden oluşan) işe alım verisinden öğrenerek "kadın" kelimesi içeren özgeçmişlerin puanını düşürdüğü ortaya çıktı; araç kullanımdan kaldırıldı.
- **Ceza adaleti:** ABD'de kullanılan bir suç tekrarı risk aracı (COMPAS) üzerine yapılan analizlerde, yeniden suç işlemeyen siyahi sanıkların yanlışlıkla "yüksek riskli" etiketlenme oranının beyaz sanıklara göre belirgin biçimde yüksek olduğu raporlandı.
- **Yüz tanıma:** Ticari yüz analiz sistemlerinin koyu tenli kadınlarda açık tenli erkeklere göre çok daha yüksek hata oranlarına sahip olduğu gösterildi (Gender Shades çalışması).
- **Sağlık:** Sağlık harcamasını "ihtiyaç" yerine vekil değişken olarak kullanan bir algoritmanın, aynı hastalık yüküne sahip siyahi hastaları sistematik olarak daha düşük riskli değerlendirdiği bulundu.

### 5.2 Yanlılığın kaynakları

| Kaynak | Açıklama | Örnek |
|---|---|---|
| **Tarihsel yanlılık** | Veri, geçmişteki ayrımcılığı yansıtır | Geçmiş işe alım kararları |
| **Temsil yanlılığı** | Bazı gruplar veride az temsil edilir | Yüz veri setlerinin çoğunluğunun açık tenli olması |
| **Ölçüm yanlılığı** | Hedef veya öznitelik, gerçek kavramın kusurlu bir vekilidir | "Tutuklanma" ≠ "suç işleme"; "sağlık harcaması" ≠ "sağlık ihtiyacı" |
| **Toplama yanlılığı** | Veri toplama süreci bazı grupları farklı etkiler | Yalnızca akıllı telefon uygulamasından toplanan veri |
| **Değerlendirme yanlılığı** | Test verisi tüm grupları temsil etmez | Sadece genel doğruluğa bakmak |
| **Geri besleme döngüsü** | Modelin kararları gelecekteki veriyi etkiler | Polis devriyelerinin modelin işaret ettiği bölgelere yoğunlaşması → orada daha çok kayıt → model o bölgeyi daha çok işaretler |

### 5.3 "Hassas özniteliği çıkarmak yeterli değil mi?"

**Hayır.** Cinsiyet veya ırk sütununu silmek ("körlük yoluyla adillik") genellikle işe yaramaz, çünkü diğer öznitelikler bu bilgiyi **vekil (proxy)** olarak taşır:
- Posta kodu → etnik köken
- Okunan okul, hobiler, ilişki durumu (ör. Adult'taki "relationship = Husband") → cinsiyet

Ayrıca hassas öznitelik silinirse **adilliği ölçmek de imkânsız** hale gelir.

### 5.4 Adillik metrikleri

İki grup (A ve B) için bir kredi onay modeli düşünelim:

| | Grup A | Grup B |
|---|---|---|
| Başvuran | 100 | 100 |
| Gerçekte ödeyebilecek (y = 1) | 70 | 50 |
| Model onayladı (ŷ = 1) | 60 | 30 |
| Ödeyebilecekler içinde onaylanan (TP) | 56 | 25 |
| Ödeyemeyecekler içinde onaylanan (FP) | 4 | 5 |

**1. Demografik eşitlik (demographic parity):** Pozitif karar oranları eşit olmalı.

```
P(ŷ=1 | A) = 60/100 = 0.60      P(ŷ=1 | B) = 30/100 = 0.30
Fark = 0.30          Oran = 0.30 / 0.60 = 0.50
```

"**Dörtte beş (%80) kuralı**" (ABD istihdam hukukundan): Oran < 0.8 ise olumsuz etki (disparate impact) şüphesi. Burada 0.50 → **ihlal**.

**2. Fırsat eşitliği (equal opportunity):** **Hak edenler** arasında onay oranı (TPR, recall) eşit olmalı.

```
TPR_A = 56/70 = 0.80            TPR_B = 25/50 = 0.50
```

Ödeyebilecek durumdaki B grubu üyelerinin yarısı reddediliyor; A grubunda bu oran %20. **Ciddi eşitsizlik.**

**3. Eşitlenmiş olasılıklar (equalized odds):** Hem TPR hem FPR eşit olmalı.

```
FPR_A = 4/30 = 0.133            FPR_B = 5/50 = 0.100
```

**4. Kalibrasyon / tahmin eşitliği:** Model "%70 olasılık" dediğinde, bu her iki grupta da gerçekleşme oranı %70 olmalı (precision eşitliği ile ilişkili).

| Metrik | Eşit olması gereken | Felsefesi |
|---|---|---|
| Demografik eşitlik | P(ŷ=1) | Sonuç eşitliği |
| Fırsat eşitliği | TPR | Hak edenlere eşit fırsat |
| Eşitlenmiş olasılıklar | TPR ve FPR | Hata türlerinde eşitlik |
| Kalibrasyon | P(y=1 \| skor) | Skorların her grup için aynı anlamı taşıması |

### 5.5 İmkânsızlık teoremi

Gruplar arasında **temel oranlar** (gerçek pozitif oranı) farklıysa (burada %70 vs %50), mükemmel olmayan bir sınıflandırıcı **kalibrasyonu ve eşitlenmiş olasılıkları aynı anda** sağlayamaz (Kleinberg ve ark., Chouldechova, 2016–2017). Bu matematiksel bir gerçektir.

**Sonuç:** "Adil" kelimesinin tek bir tanımı yoktur. Hangi metriğin önceleneceği **teknik değil, etik ve toplumsal bir karardır** ve paydaşlarla (etkilenen topluluklar, hukukçular, alan uzmanları) birlikte verilmelidir.

### 5.6 Yanlılığı azaltma yöntemleri

| Aşama | Yöntem | Örnek |
|---|---|---|
| **Ön işleme** (veri) | Veriyi dengele, yeniden ağırlıklandır, temsil eksikliğini gider | Az temsil edilen gruptan daha fazla veri toplama; örnek ağırlıkları |
| **İşlem sırasında** (model) | Adillik kısıtlı optimizasyon | Kayba adillik cezası eklemek (`fairlearn` ExponentiatedGradient) |
| **Son işleme** (karar) | Gruba özel eşik değerleri | TPR'leri eşitleyecek eşikler (`fairlearn` ThresholdOptimizer) |

Hiçbir teknik yöntem, kötü tanımlanmış bir problemi veya yanlı bir hedef değişkeni düzeltemez. **Önce soruyu sorun:** Bu sistemi kurmalı mıyız? Hedef değişken ölçmek istediğimiz şeyi gerçekten ölçüyor mu? Kimler etkilenecek?

### 5.7 Adillik analizi kodu

```python
def grup_metrikleri(y_gercek, y_tahmin, grup):
    satirlar = []
    for g in sorted(pd.unique(grup)):
        m = (grup == g)
        yt, yp = y_gercek[m], y_tahmin[m]
        satirlar.append({
            "grup": g, "n": int(m.sum()),
            "taban_oranı P(y=1)": yt.mean(),
            "seçim_oranı P(ŷ=1)": yp.mean(),
            "TPR": yp[yt == 1].mean(),
            "FPR": yp[yt == 0].mean(),
            "precision": yt[yp == 1].mean() if yp.sum() else np.nan,
            "doğruluk": (yt == yp).mean(),
        })
    return pd.DataFrame(satirlar).set_index("grup").round(3)

y_tahmin = en_iyi.predict(X_te)
cinsiyet = X_te["sex"].values
tablo = grup_metrikleri(y_te.values, y_tahmin, cinsiyet)
print(tablo)
oran = tablo["seçim_oranı P(ŷ=1)"].min() / tablo["seçim_oranı P(ŷ=1)"].max()
print("Demografik eşitlik oranı:", round(oran, 3))
print("TPR farkı:", round(tablo["TPR"].max() - tablo["TPR"].min(), 3))
```

**fairlearn kütüphanesi ile:**

```python
# pip install fairlearn
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer

mf = MetricFrame(metrics={"seçim": selection_rate, "TPR": true_positive_rate},
                 y_true=y_te, y_pred=y_tahmin, sensitive_features=X_te["sex"])
print(mf.by_group); print("Farklar:\n", mf.difference())

# Son işleme: fırsat eşitliği için gruba özel eşikler.
# İdeal olarak eşikler modelin görmediği ayrı bir doğrulama setinde öğrenilmelidir;
# burada basitlik için eğitim verisini kullanıyoruz.
to = ThresholdOptimizer(estimator=en_iyi, constraints="true_positive_rate_parity",
                        prefit=True, predict_method="predict_proba")
to.fit(X_tr, y_tr, sensitive_features=X_tr["sex"])
y_adil = to.predict(X_te, sensitive_features=X_te["sex"], random_state=0)
print(grup_metrikleri(y_te.values, np.asarray(y_adil), cinsiyet))
```

Adillik iyileştirmesinin genel doğrulukta küçük bir düşüşe yol açması sık görülür. Bu **ödünleşim** de açıkça raporlanmalıdır.

### 5.8 Diğer etik konular

- **Gizlilik:** Kişisel verilerin toplanması, saklanması ve kullanımı (KVKK, GDPR). Anonimleştirme her zaman yeterli değildir; birkaç özniteliğin birleşimi kişiyi tanımlayabilir. Gelişmiş yöntemler: diferansiyel gizlilik, federe öğrenme.
- **Şeffaflık ve hesap verebilirlik:** Modelin amacı, eğitim verisi, sınırlamaları ve performansı belgelenmeli. **Model kartları (model cards)** ve **veri seti bilgi formları (datasheets for datasets)** bu amaçla önerilmiştir.
- **İnsan denetimi:** Yüksek riskli kararlarda (sağlık, hukuk, istihdam, kredi) model bir **karar destek** aracı olmalı; son söz ve itiraz mekanizması insanlarda kalmalı.
- **Kötüye kullanım ve çift kullanım:** Deepfake, gözetim, manipülasyon. Teknolojinin nasıl kullanılabileceğini önceden düşünmek.
- **Çevresel etki:** Büyük modellerin eğitimi ciddi enerji tüketir. Gereğinden büyük model kullanmamak da bir etik tercihtir.
- **Düzenlemeler:** AB Yapay Zekâ Yasası (AI Act), yapay zekâ sistemlerini risk düzeylerine göre sınıflandırır ve yüksek riskli sistemlere şeffaflık, insan denetimi ve risk yönetimi yükümlülükleri getirir.

### 5.9 Model kartı şablonu

```markdown
# Model Kartı: Gelir Tahmin Modeli v1.0

## Model Detayları
- Geliştiren / tarih / sürüm:
- Model türü: HistGradientBoostingClassifier (scikit-learn x.y)

## Amaçlanan Kullanım
- Birincil kullanım: Eğitim amaçlı gösterim
- Kapsam dışı kullanımlar: Gerçek kredi, işe alım veya kamu hizmeti kararları

## Veri
- Eğitim verisi: UCI Adult (1994 ABD nüfus sayımı), 39.073 kayıt
- Bilinen sınırlamalar: Eski tarihli, tek ülke, ikili cinsiyet kodlaması

## Performans
- Test ROC-AUC: ...  |  Test F1: ...

## Adillik Değerlendirmesi
| Grup | Seçim oranı | TPR | FPR |
|---|---|---|---|
| ... | ... | ... | ... |

## Etik Hususlar ve Riskler
- Tarihsel gelir eşitsizliklerini yeniden üretebilir ...

## Öneriler
- ...
```

---

## 6. Laboratuvar: Dönem Projesi Sunumları

### 6.1 Sunum yapısı (öneri: 12 dakika sunum + 5 dakika soru)

1. **Problem ve motivasyon** (1 dk): Ne tahmin ediyorsunuz, kimin işine yarar?
2. **Veri** (2 dk): Kaynak, boyut, öznitelikler; EDA'dan 2–3 önemli bulgu
3. **Yöntem** (3 dk): Ön işleme pipeline'ı, denediğiniz modeller, baseline, doğrulama stratejisi
4. **Sonuçlar** (3 dk): Karşılaştırma tablosu (CV + tek seferlik test sonucu), hata analizi
5. **Açıklanabilirlik ve etik** (2 dk): SHAP / permütasyon önemi; varsa hassas gruplar için adillik analizi; sınırlamalar
6. **Sonuç ve gelecek çalışmalar** (1 dk)

### 6.2 Teslim edilecekler

- [ ] Kaynak kod (not defteri veya `.py` dosyaları) + `requirements.txt`
- [ ] Kaydedilmiş son model ve meta veri dosyası
- [ ] Kısa rapor (4–6 sayfa) ve **model kartı**
- [ ] Sunum dosyası
- [ ] (Bonus) Çalışan bir API veya Streamlit/Gradio demosu

### 6.3 Kendi projeniz için kontrol listesi

**Veri ve doğrulama**
- [ ] Test seti en başta ayrıldı ve yalnızca en sonda bir kez kullanıldı
- [ ] Tüm ön işleme adımları Pipeline içinde (sızıntı yok)
- [ ] Hedef sızıntısı yaratabilecek öznitelikler kontrol edildi
- [ ] Zaman serisi / gruplu veri varsa uygun bölme yöntemi kullanıldı
- [ ] Sonuçlar `random_state` ile tekrarlanabilir

**Modelleme**
- [ ] Basit bir baseline ile karşılaştırma yapıldı
- [ ] Probleme uygun metrik seçildi (dengesiz veride accuracy tek başına değil)
- [ ] Hiperparametreler çapraz doğrulama ile seçildi
- [ ] Öğrenme eğrileri / eğitim-doğrulama farkı incelendi

**Yorum ve etik**
- [ ] En önemli öznitelikler ve etki yönleri açıklandı
- [ ] Modelin hatalı olduğu örnekler incelendi
- [ ] Hassas gruplar için performans ayrı ayrı raporlandı (varsa)
- [ ] Sınırlamalar ve olası kötüye kullanımlar dürüstçe belirtildi

### 6.4 Değerlendirme ölçütleri (örnek)

| Ölçüt | Ağırlık |
|---|---|
| Problem tanımı ve veri anlayışı (EDA) | %15 |
| Metodolojinin doğruluğu (pipeline, sızıntısızlık, doğrulama) | %25 |
| Model geliştirme ve karşılaştırma | %20 |
| Sonuçların yorumu, açıklanabilirlik ve hata analizi | %15 |
| Etik değerlendirme ve sınırlamalar | %10 |
| Sunum ve iletişim | %10 |
| Kod kalitesi ve tekrarlanabilirlik | %5 |

---

## 7. Dönem Özeti: Öğrendiklerimizin Haritası

```
                          MAKİNE ÖĞRENMESİ
                                │
        ┌───────────────────────┼─────────────────────────┐
   DENETİMLİ                DENETİMSİZ              DERİN ÖĞRENME
        │                       │                         │
  ┌─────┴──────┐         ┌──────┴──────┐           ┌──────┴──────┐
REGRESYON  SINIFLANDIRMA KÜMELEME  BOYUT İNDİRGEME  MLP   CNN   Transformer
 Doğrusal    Lojistik     k-means     PCA          (H12)  (H13)    (H13)
 Polinom     k-NN         Hiyerarşik  t-SNE/UMAP
 Ridge/Lasso Naive Bayes  DBSCAN      Öznitelik seçimi
 (H3–4)      Karar ağacı  (H10)       (H11)
             Topluluklar
             SVM (H5–9)

 HER ŞEYİN TEMELİ: Veri hazırlama (H2) · Genelleme ve doğrulama (H4) · Metrikler (H5)
                   Uygulama, açıklanabilirlik ve etik (H14)
```

**Akılda kalması gereken on ilke:**

1. Veriyi tanımadan model kurmayın (EDA).
2. Test setini en başta ayırın ve en sona kadar dokunmayın.
3. Ön işlemeyi Pipeline içine koyun; `fit` sadece eğitim verisinde.
4. Her zaman basit bir baseline ile başlayın.
5. Probleme uygun metriği seçin; accuracy her zaman doğru ölçü değildir.
6. Eğitim hatası değil, genelleme hatası önemlidir: çapraz doğrulama kullanın.
7. Yanlılık-varyans dengesini düşünün: Model çok basit mi, çok karmaşık mı?
8. Tablo verisinde ağaç toplulukları, görüntü ve metinde derin öğrenme genellikle güçlü başlangıç noktalarıdır.
9. Modelinizi açıklayabilmeli ve hatalarını anlayabilmelisiniz.
10. Modeliniz insanları etkiliyorsa, kimin nasıl etkilendiğini ölçün ve sorumluluk alın.

## Kendinizi Sınayın

1. Bir modeli `joblib` ile kaydettiniz ama üretim sunucusunda farklı bir scikit-learn sürümü kurulu. Ne gibi sorunlar çıkabilir?
2. Üç öznitelikli bir model için v(∅) = 10, v(A) = 20, v(B) = 15, v(C) = 10, v(AB) = 30, v(AC) = 25, v(BC) = 20, v(ABC) = 40 verildiğinde A'nın Shapley değerini hesaplayın.
3. Permütasyon öneminde neden eğitim verisi yerine doğrulama/test verisi kullanılması önerilir?
4. Bir işe alım modelinde kadın ve erkek adaylar için TPR değerleri 0.65 ve 0.85. Bu hangi adillik ilkesinin ihlalidir? Bunu azaltmak için ne yapılabilir?
5. Cinsiyet sütununu veri setinden çıkarmak neden adilliği garanti etmez?
6. Bir modelin dağıtımdan 6 ay sonra performansının düştüğünü fark ettiniz. Olası nedenler neler ve nasıl teşhis edersiniz?

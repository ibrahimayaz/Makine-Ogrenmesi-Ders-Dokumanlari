# Hafta 5 — Lojistik Regresyon ve Sınıflandırma Metrikleri

## Bu haftanın sonunda

Bu notu bitirdiğinizde:

- Lojistik regresyonun sınıf olasılığını nasıl tahmin ettiğini açıklayabilirsiniz.
- Sigmoid, odds, karar eşiği ve çapraz entropi kavramlarını kullanabilirsiniz.
- İkili lojistik regresyonu NumPy ile sıfırdan eğitebilirsiniz.
- Karmaşıklık matrisi, precision, recall, F1, ROC-AUC ve precision-recall eğrisini yorumlayabilirsiniz.
- Dengesiz sınıflarda uygun metrik ve eşik seçiminin önemini açıklayabilirsiniz.
- Meme kanseri veri setinde hataların maliyetini dikkate alarak sınıflandırma eşiği seçebilirsiniz.

## Sınıflandırma problemi

Sınıflandırmada model, bir örneği önceden belirlenmiş sınıflardan birine atar. Örneğin bir e-posta **spam** veya **spam değil**, bir tümör ise **kötü huylu** veya **iyi huylu** olarak sınıflandırılabilir.

Bir sınıflandırıcı çoğu zaman doğrudan sınıf etiketi üretmek yerine, örneğin pozitif sınıfa ait olma olasılığını tahmin eder. Bu olasılığı bir karar eşiğiyle etikete çeviririz. Varsayılan eşik `0.5` olsa da her problem için en iyi eşik bu olmak zorunda değildir.

Bu nottaki formüllerde hedef `y`, negatif sınıf için `0`, pozitif sınıf için `1` olarak kodlanır. “Pozitif” sınıfı, problemde özellikle bulmak istediğimiz sınıf olarak seçeriz.

## Lojistik regresyon

Lojistik regresyon, girdilerden bir doğrusal puan hesaplar ve bu puanı `0` ile `1` arasında bir olasılığa dönüştürür. “Regresyon” adını taşısa da yaygın kullanımı sınıflandırmadır.

Önce doğrusal puanı hesaplarız:

`z = w₁x₁ + w₂x₂ + ... + wₚxₚ + b`

Ardından **sigmoid** fonksiyonunu uygularız:

`σ(z) = 1 / (1 + e⁻ᶻ)`

Modelin pozitif sınıf olasılığı:

`P(y = 1 | x) = σ(z)`

Sigmoid, çok negatif puanları sıfıra, çok pozitif puanları bire yaklaştırır. Örneğin `z = 0` için olasılık `0.5`, `z = 2` için yaklaşık `0.881`, `z = -2` için yaklaşık `0.119` olur.

### Karar sınırı

Eşik `0.5` ise model, olasılık `0.5` veya daha büyük olduğunda pozitif sınıfı seçer. Sigmoid `0` noktasında `0.5` olduğundan bu karar, `z ≥ 0` koşuluna eşdeğerdir. Girdi uzayında `z = 0` denklemi modelin **karar sınırını** oluşturur.

Eşik `0.5` dışında seçilirse karar sınırı da değişir. Eşiği düşürmek daha fazla örneği pozitif sınıfa atamaya eğilimlidir; bu genellikle recall'ı artırırken yanlış pozitifleri de artırabilir.

### Odds ve katsayı yorumu

Pozitif sınıf olasılığının **odds** değeri:

`odds = p / (1 - p)`

Lojistik regresyon, log-odds değerini doğrusal bir ifadeyle modeller:

`log(p / (1 - p)) = w₁x₁ + ... + wₚxₚ + b`

Diğer öznitelikler sabitken `xⱼ` bir birim arttığında odds, `eʷʲ` ile çarpılır. Örneğin `wⱼ = 0.7` ise odds çarpanı `e⁰·⁷ ≈ 2.01` olur. Bu, olasılığın iki katına çıktığı anlamına gelmez; odds ile olasılık farklı büyüklüklerdir. Öznitelik ölçeklendirilmişse katsayı yorumu da ölçek birimi üzerinden yapılır.

## Çapraz entropi kaybı

Modelin tahmin olasılığını gerçek etiketle karşılaştırmak için ikili çapraz entropi (log loss) kullanılır:

`L = -[y log(p) + (1 - y) log(1 - p)]`

- Gerçek etiket `1` ise kayıp `-log(p)` olur; model yüksek olasılık vermelidir.
- Gerçek etiket `0` ise kayıp `-log(1 - p)` olur; model düşük olasılık vermelidir.
- Model yanlış bir tahminden çok emin olduğunda kayıp büyür.

Birden fazla örnek için kayıpların ortalaması alınır. Olasılıklar sayısal olarak tam `0` veya `1` olabileceği için uygulamada logaritma öncesinde küçük bir `ε` değeriyle sınırlandırma yapılır.

### Gradyan ve bir adımlık elle hesap

`pᵢ = σ(w xᵢ + b)` ve `m` eğitim örneği için ortalama çapraz entropi kaybının gradyanları şöyledir:

`∂L/∂w = (1/m) Σ (pᵢ - yᵢ)xᵢ`

`∂L/∂b = (1/m) Σ (pᵢ - yᵢ)`

Gradyan iniş güncellemesi:

`w_yeni = w_eski - η(∂L/∂w)`

`b_yeni = b_eski - η(∂L/∂b)`

Burada `η`, öğrenme oranıdır. İki örnekle başlangıçta `w = 0`, `b = 0` olsun:

| `x` | `y` | Başlangıç `z` | Tahmin `p` | `p - y` |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0.5 | -0.5 |
| -1 | 0 | 0 | 0.5 | 0.5 |

Bu örnekler için:

- `∂L/∂w = [(-0.5 × 1) + (0.5 × -1)] / 2 = -0.5`
- `∂L/∂b = (-0.5 + 0.5) / 2 = 0`
- Öğrenme oranı `η = 0.2` ise güncellemeden sonra `w = 0.1`, `b = 0` olur.

Model, pozitif örneğe daha yüksek puan verme yönünde bir adım atmıştır.

## İkili lojistik regresyonu NumPy ile yazma

Aşağıdaki küçük örnek, bir öznitelik ve tüm veri kümesi üzerinde gradyan inişi kullanır. Amaç optimizasyon adımını görünür kılmaktır; gerçek uygulamalarda öznitelik ölçekleme, daha sağlam durdurma ölçütleri ve doğrulama da gerekir.

```python
import numpy as np

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.0, 0.0, 1.0, 1.0])

w = np.zeros(X.shape[1])
b = 0.0
ogrenme_orani = 0.1
adim_sayisi = 5000
epsilon = 1e-15

for adim in range(adim_sayisi):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))

    kayip = -np.mean(
        y * np.log(np.clip(p, epsilon, 1 - epsilon))
        + (1 - y) * np.log(np.clip(1 - p, epsilon, 1 - epsilon))
    )
    dw = X.T @ (p - y) / len(y)
    db = np.mean(p - y)

    w -= ogrenme_orani * dw
    b -= ogrenme_orani * db

    if adim % 1000 == 0:
        print(f"Adım {adim:4d} | kayıp = {kayip:.4f}")

olasiliklar = 1 / (1 + np.exp(-(X @ w + b)))
tahminler = (olasiliklar >= 0.5).astype(int)

print("Katsayı:", w)
print("Sabit:", b)
print("Olasılıklar:", np.round(olasiliklar, 3))
print("Tahminler:", tahminler)
```

Kayıp eğitim sırasında genel olarak azalmalıdır. Son tahminler `0, 0, 1, 1` olur. Gerçek veri kümelerinde gradyan inişinin davranışı özellik ölçeklerine ve öğrenme oranına bağlıdır; ölçekleme yapılmaması yakınsamayı yavaşlatabilir veya kararsızlaştırabilir.

## Çok sınıflı sınıflandırma: softmax

Birbirini dışlayan `K` sınıftan birini seçmek için her sınıfa bir puan hesaplanır:

`zₖ = wₖᵀx + bₖ`

Softmax bu puanları toplamı `1` olan olasılıklara dönüştürür:

`P(y = k | x) = eᶻᵏ / Σⱼ eᶻʲ`

Tahmin, olasılığı en yüksek sınıftır. Sayısal taşmayı azaltmak için uygulamada tüm puanlardan en büyük puan çıkarılır; bu işlem softmax olasılıklarını değiştirmez. Çok sınıflı çapraz entropi, doğru sınıfın tahmin olasılığının logaritmasının negatifidir.

scikit-learn ile çok sınıflı kullanım örneği:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_egitim, y_egitim)

print("Test doğruluğu:", model.score(X_test, y_test))
print("İlk iki örneğin sınıf olasılıkları:")
print(np.round(model.predict_proba(X_test[:2]), 3))
```

`predict_proba` çıktısında her satır bir örneği, her sütun bir sınıfın olasılığını gösterir. `StandardScaler`, `Pipeline` içinde olduğu için yalnızca eğitim verisinden öğrenilir.

## Karmaşıklık matrisi

İkili sınıflandırmada gerçek etiket ile model tahminini karşılaştıran tabloya **karmaşıklık matrisi** denir:

|  | Tahmin negatif | Tahmin pozitif |
|---|---:|---:|
| Gerçek negatif | True Negative (TN) | False Positive (FP) |
| Gerçek pozitif | False Negative (FN) | True Positive (TP) |

- **TP:** Pozitif örnek doğru biçimde pozitif tahmin edilmiştir.
- **TN:** Negatif örnek doğru biçimde negatif tahmin edilmiştir.
- **FP:** Negatif örnek yanlışlıkla pozitif tahmin edilmiştir; yanlış alarmdır.
- **FN:** Pozitif örnek yanlışlıkla negatif tahmin edilmiştir; gözden kaçırmadır.

Örneğin `TN = 50`, `FP = 10`, `FN = 5`, `TP = 35` olsun. Toplam `100` örnekte `85` doğru tahmin vardır.

### Temel metrikler

**Accuracy (doğruluk):** Tüm tahminlerin ne kadarının doğru olduğunu ölçer.

`accuracy = (TP + TN) / (TP + TN + FP + FN) = 85 / 100 = 0.85`

**Precision (kesinlik):** Pozitif tahmin edilenlerin ne kadarı gerçekten pozitiftir?

`precision = TP / (TP + FP) = 35 / 45 ≈ 0.778`

Yanlış pozitiflerin maliyeti yüksekse precision önemli olabilir.

**Recall (duyarlılık / yakalama oranı):** Gerçek pozitiflerin ne kadarı bulundu?

`recall = TP / (TP + FN) = 35 / 40 = 0.875`

Pozitif örnekleri kaçırmanın maliyeti yüksekse recall önem kazanır.

**F1 skoru:** Precision ve recall'ın harmonik ortalamasıdır:

`F1 = 2 × precision × recall / (precision + recall)`

Bu örnekte `F1 = 70 / 85 ≈ 0.824` olur. F1, iki metriğin de iyi olmasını ister; ancak doğru negatifleri hesaba katmadığı için her kullanım amacı için tek başına yeterli değildir.

**Specificity (özgüllük):** Gerçek negatiflerin ne kadarının doğru negatif bulunduğunu ölçer:

`specificity = TN / (TN + FP) = 50 / 60 ≈ 0.833`

İhtiyaca göre dengeli doğruluk (balanced accuracy), özgüllük ile recall'ın ortalamasıdır. Metriklerin paydası sıfır olduğunda değer tanımsız olabilir; kullanılan kütüphanenin bu durumu nasıl ele aldığını kontrol edin.

## ROC eğrisi ve AUC

Eşik değiştikçe:

- **TPR (True Positive Rate)**, yani recall: `TP / (TP + FN)`
- **FPR (False Positive Rate)**: `FP / (FP + TN) = 1 - specificity`

farklı değerler alır. ROC eğrisi, yatay eksende FPR'yi ve dikey eksende TPR'yi gösterir. Eğrinin sol üst köşeye yaklaşması, yanlış pozitif oranı düşükken gerçek pozitifleri yakalama oranının yüksek olduğunu gösterir.

**AUC**, ROC eğrisinin altında kalan alandır. Eşiklerden bağımsız bir sıralama ölçüsü olarak yorumlanabilir: rastgele seçilen bir pozitif örneğin, rastgele seçilen bir negatif örnekten daha yüksek skor alma olasılığıdır. `0.5` civarı rastgele sıralamaya, `1.0` kusursuz sıralamaya karşılık gelir.

Elle küçük bir örnekte pozitif örneklerin skorları `0.9` ve `0.4`, negatif örneklerin skorları `0.8` ve `0.3` olsun. Dört pozitif-negatif çiftinin üçünde pozitif örneğin skoru daha yüksektir; dolayısıyla AUC `3/4 = 0.75` olur. Gerçek veride AUC, ROC eğrisi boyunca trapez alanlarının toplamı olarak hesaplanır.

## Precision-recall eğrisi

Precision-recall (PR) eğrisi, farklı eşiklerde precision ile recall arasındaki değiş tokuşu gösterir. Pozitif sınıf nadir olduğunda ROC eğrisi iyi görünebilir; çünkü çok sayıdaki negatif örnek FPR'yi küçük tutabilir. Bu nedenle dengesiz veri ve nadir pozitif sınıf çalışmalarında PR eğrisi ile sınıf oranını birlikte incelemek yararlıdır.

ROC-AUC ve PR-AUC tek bir karar eşiğine bağlı değildir. Uygulamada seçilecek eşikteki confusion matrix ve ilgili iş maliyetleri ayrıca değerlendirilmelidir.

## scikit-learn ile sınıflandırma ölçümleri

Aşağıdaki örnek Breast Cancer Wisconsin veri setini kullanır. scikit-learn'de hedef `0` kötü huylu, `1` iyi huylu anlamına gelir. Bu derste **kötü huylu** sınıfı pozitif olarak tanımlıyoruz; böylece recall, kötü huylu örnekleri yakalama oranını gösterir.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

veri = load_breast_cancer()
X = veri.data
y = (veri.target == 0).astype(int)  # 1: kötü huylu, 0: iyi huylu

X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
model.fit(X_egitim, y_egitim)

olasilik = model.predict_proba(X_test)[:, 1]
tahmin = (olasilik >= 0.5).astype(int)

print("Karmaşıklık matrisi:\n", confusion_matrix(y_test, tahmin))
print("Accuracy:", accuracy_score(y_test, tahmin))
print("Precision:", precision_score(y_test, tahmin, zero_division=0))
print("Recall:", recall_score(y_test, tahmin, zero_division=0))
print("F1:", f1_score(y_test, tahmin, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, olasilik))
print(classification_report(
    y_test, tahmin, target_names=["İyi huylu", "Kötü huylu"], zero_division=0
))

fpr, tpr, roc_esikleri = roc_curve(y_test, olasilik)
precision, recall, pr_esikleri = precision_recall_curve(y_test, olasilik)
```

`predict_proba` ile alınan olasılıklar AUC ve eğri hesaplamalarında kullanılır; `predict` ise modelin varsayılan sınıflandırma kararını verir. Eğri noktalarını çizmek için `matplotlib` ile `(fpr, tpr)` ve `(recall, precision)` çiftleri grafiğe aktarılabilir.

## Karar eşiği ve hata maliyeti

Bir eşiği yalnızca accuracy en yüksek olsun diye seçmek doğru olmayabilir. Hataların maliyetleri farklıysa, örneğin hastalık taramasında pozitif vakayı kaçırmak yanlış alarmdan daha ağırsa, eşik bu maliyetleri yansıtmalıdır.

Meme kanseri örneğinde kötü huylu sınıfı pozitif kabul edelim. `FN` maliyetini `5`, `FP` maliyetini `1` olarak varsayarsak toplam maliyet:

`maliyet = 5 × FN + 1 × FP`

olur. Aşağıdaki kod, eğitim verisini model kurmak için, doğrulama verisini eşiği seçmek için ve test verisini yalnızca son değerlendirme için kullanır. Böylece test kümesi eşik seçiminde kullanılmaz.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

veri = load_breast_cancer()
X = veri.data
y = (veri.target == 0).astype(int)
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_egitim, X_dogrulama, y_egitim, y_dogrulama = train_test_split(
    X_gecici, y_gecici, test_size=0.25, random_state=42, stratify=y_gecici
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
model.fit(X_egitim, y_egitim)

dogrulama_olasilik = model.predict_proba(X_dogrulama)[:, 1]
esikler = np.arange(0.05, 0.96, 0.01)
maliyetler = []

for esik in esikler:
    dogrulama_tahmin = (dogrulama_olasilik >= esik).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_dogrulama, dogrulama_tahmin, labels=[0, 1]
    ).ravel()
    maliyetler.append(5 * fn + fp)

en_iyi_esik = esikler[np.argmin(maliyetler)]
print("Doğrulamada seçilen eşik:", round(en_iyi_esik, 2))

test_olasilik = model.predict_proba(X_test)[:, 1]
test_tahmin = (test_olasilik >= en_iyi_esik).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_test, test_tahmin, labels=[0, 1]
).ravel()

print("Test karmaşıklık matrisi:\n", confusion_matrix(y_test, test_tahmin))
print("Test maliyeti (5 × FN + FP):", 5 * fn + fp)
print("Test recall:", recall_score(y_test, test_tahmin, zero_division=0))
print("Test precision:", precision_score(y_test, test_tahmin, zero_division=0))
```

Bu maliyet oranı yalnızca öğretici bir varsayımdır; gerçek bir sağlık uygulamasında klinik uzmanlarla belirlenmeli ve ayrıca kalibrasyon, güven aralıkları ve dış doğrulama değerlendirilmelidir. Eşik seçimi veri kümesine bağlıdır. Aynı doğrulama kümesinde çok sayıda karar veya model karşılaştırıldığında seçim iyimserleşebilir; bu durumda çapraz doğrulama veya iç içe çapraz doğrulama düşünülmelidir.

## Dengesiz sınıflar

Sınıf dağılımı belirgin biçimde dengesiz olduğunda accuracy yanıltıcı olabilir. Örneğin 1000 örneğin yalnızca 10'u pozitifse, tüm örneklere negatif diyen modelin accuracy değeri `%99` olur; ancak hiçbir pozitif örneği bulamaz.

Dengesizlikle baş etmek için şu yaklaşımlar kullanılabilir:

- **Uygun metrik seçimi:** Pozitif sınıf önemliyse recall, precision, F1, PR-AUC ve confusion matrix'i inceleyin. Sınıf oranını da raporlayın.
- **Karar eşiği:** Olasılık eşiğini, doğrulama verisi ve hata maliyetlerine göre seçin.
- **Sınıf ağırlıkları:** Daha az görülen veya kaçırılması daha maliyetli sınıfa daha fazla ağırlık verin. scikit-learn'de örneğin `LogisticRegression(class_weight="balanced")` kullanılabilir. Ağırlıklar olasılık kalibrasyonunu etkileyebileceği için sonucu doğrulama verisinde değerlendirin.
- **Yeniden örnekleme:** Eğitim verisinde azınlık sınıfını artırma (oversampling) veya çoğunluk sınıfını azaltma (undersampling) uygulanabilir.
- **SMOTE:** Azınlık sınıfı örnekleri arasında ara noktalar üreterek sentetik örnek oluşturur. Kategorik verilerde doğrudan standart SMOTE uygun olmayabilir; veri türüne uygun yöntem seçilmelidir.

Yeniden örnekleme yalnızca eğitim katında yapılmalıdır. Önce tüm veriye SMOTE uygulayıp sonra eğitim/doğrulama ayrımı yapmak, doğrulama örneklerinin bilgilerini eğitim tarafına sızdırabilir. Çapraz doğrulama kullanılırken örnekleme işlemi her eğitim katında tekrarlanmalıdır; bunun için `imbalanced-learn` içindeki `Pipeline` kullanılabilir.

Sınıf ağırlıklı ve ağırlıksız modelleri karşılaştırırken aynı eğitim/doğrulama ayrımını kullanın:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

veri = load_breast_cancer()
X = veri.data
y = (veri.target == 0).astype(int)
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_egitim, X_dogrulama, y_egitim, y_dogrulama = train_test_split(
    X_gecici, y_gecici, test_size=0.25, random_state=42, stratify=y_gecici
)

agirliksiz = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)
agirlikli = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=2000)
)

agirliksiz.fit(X_egitim, y_egitim)
agirlikli.fit(X_egitim, y_egitim)
print("Ağırlıksız doğrulama recall:",
      recall_score(y_dogrulama, agirliksiz.predict(X_dogrulama)))
print("Ağırlıklı doğrulama recall:",
      recall_score(y_dogrulama, agirlikli.predict(X_dogrulama)))
```

Örnek bir eğitim pipeline'ı:

```python
# Gerekirse: pip install imbalanced-learn
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veri = load_breast_cancer()
X = veri.data
y = (veri.target == 0).astype(int)
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote_model = ImbPipeline([
    ("olcekleyici", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("siniflandirici", LogisticRegression(max_iter=2000)),
])
smote_model.fit(X_egitim, y_egitim)
```

Bu pipeline'ı yalnızca `X_egitim`, `y_egitim` üzerinde eğitin; doğrulama ve test verisini orijinal sınıf dağılımında bırakın. Ölçekleme SMOTE'tan önce, sadece eğitim katında öğrenilir.

## Uygulama

1. NumPy örneğinde öğrenme oranını `0.01` ve `1.0` yapın. Kayıp ve tahminler nasıl değişiyor?
2. Elle verilen karmaşıklık matrisi için accuracy, precision, recall, F1 ve specificity değerlerini tekrar hesaplayın.
3. Meme kanseri örneğinde pozitif sınıfı kötü huylu seçmenin precision ve recall yorumunu nasıl değiştirdiğini açıklayın.
4. Test verisinde `0.5` eşiği ile doğrulama verisinde seçtiğiniz eşiği karşılaştırın. `FN` ve `FP` sayılarını birlikte yorumlayın.
5. `class_weight="balanced"` kullanıp sınıf ağırlıklı ve ağırlıksız modelleri aynı test kümesinde karşılaştırın. Eşiği doğrulama verisi dışında seçmeyin.
6. Precision-recall eğrisini çizin ve pozitif sınıf oranını referans çizgisi olarak gösterin.
7. SMOTE'u eğitim verisine uygulayıp test verisindeki sınıf dağılımının değişmediğini doğrulayın.

## Kısa özet

- Lojistik regresyon doğrusal puanı sigmoid ile olasılığa dönüştürür.
- Çapraz entropi, doğru sınıfa yüksek olasılık verilmesini teşvik eder; gradyan iniş katsayıları günceller.
- Softmax, birbirini dışlayan ikiden fazla sınıf için olasılık üretir.
- Accuracy tek başına yeterli olmayabilir; precision ve recall farklı hata türlerini ölçer.
- ROC-AUC ve PR eğrileri eşik değişimini inceler; dengesiz veride PR analizi özellikle yararlıdır.
- Eşik, hata maliyetleri ve doğrulama verisine göre seçilmeli; test verisi son değerlendirmeye saklanmalıdır.
- Sınıf ağırlığı ve yeniden örnekleme yalnızca eğitim sürecinde uygulanmalıdır.

## Kendinizi sınayın

1. Lojistik regresyonda doğrusal puan neden doğrudan sınıf olasılığı olarak kullanılamaz?
2. Sigmoid fonksiyonunun `z = 0` için çıktısı nedir ve `0.5` eşiğinde bunun anlamı nedir?
3. Odds ile olasılık arasındaki fark nedir? Bir katsayı odds'u nasıl etkiler?
4. Bir modelin precision'ı yüksek, recall'ı düşükse ne tür hatalarla karşılaşıyor olabilir?
5. Pozitif sınıf oranı çok düşükken accuracy neden yanıltıcı olabilir?
6. ROC-AUC ile PR eğrisi hangi bilgileri sunar? Dengesiz veride PR eğrisi neden faydalıdır?
7. Eşik seçerken test verisini kullanmak neden değerlendirmeyi yanlı hale getirir?
8. SMOTE'u eğitim/doğrulama ayrımından önce tüm veriye uygulamak neden veri sızıntısına yol açabilir?
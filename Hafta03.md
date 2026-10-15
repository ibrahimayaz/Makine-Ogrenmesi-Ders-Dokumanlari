# Hafta 3 — Doğrusal Regresyon

## Bu haftanın sonunda

Bu notu bitirdiğinizde:

- Doğrusal regresyonun sayısal bir hedefi nasıl tahmin ettiğini açıklayabilirsiniz.
- Bir doğruyu birkaç noktaya uydurmanın temel fikrini anlayabilirsiniz.
- Hata (artık), ortalama karesel hata ve gradyan iniş kavramlarını tanıyabilirsiniz.
- Basit bir regresyon modelini Python ile eğitip temel ölçülerle değerlendirebilirsiniz.

## Doğrusal regresyon nedir?

Doğrusal regresyon, sayısal bir değeri tahmin etmek için bir veya daha fazla öznitelikle hedef arasında yaklaşık doğrusal bir ilişki kurar. Örneğin, bir evin metrekaresinden satış fiyatını tahmin edebiliriz.

Tek bir öznitelikle kullanılan basit model şöyledir:

`ŷ = wx + b`

- `x`: modele verilen öznitelik
- `ŷ`: modelin tahmini
- `w`: doğrunun eğimi; `x` değiştiğinde tahminin ne kadar değiştiğini gösterir
- `b`: başlangıç değeri; `x = 0` iken modelin tahmini

Birden fazla öznitelik varsa model her özniteliğin katkısını toplar:

`ŷ = w1*x1 + w2*x2 + ... + b`

Örneğin ev fiyatı modelinde `x1` metrekare, `x2` oda sayısı olabilir. Modelin öğrendiği ağırlıklar (`w1`, `w2`) her bilginin tahmine katkısını belirler. Bu ilişki her zaman gerçeği tam olarak açıklamaz; model, verideki genel eğilimi yakalamaya çalışır.

## Tahmin hatası ve kayıp

Bir örneğin **artığı**, gerçek değer ile tahmin arasındaki farktır:

`artık = gerçek değer - tahmin`

Modelin veriye ne kadar uyduğunu değerlendirmek için hataları bir araya getiren bir ölçü kullanırız. Yaygın bir seçim **ortalama karesel hata**dır (MSE):

`MSE = (hataların kareleri toplamı) / örnek sayısı`

Hataları kareye almak, pozitif ve negatif hataların birbirini götürmesini önler; büyük hatalara daha fazla ağırlık verir.

### Küçük bir hesaplama

Bir veri kümesindeki gerçek değerler `2, 3, 5`, model tahminleri ise `2, 4, 4` olsun:

- Hatalar: `0, -1, 1`
- Hataların kareleri: `0, 1, 1`
- `MSE = (0 + 1 + 1) / 3 = 2/3`

MSE'nin birimi hedefin biriminin karesidir. Bu nedenle MSE ile birlikte, hata birimini koruyan RMSE veya MAE de raporlanabilir.

## Doğrunun en iyi uyumunu bulma

Modelin `w` ve `b` değerleri, eğitim verisindeki hatayı küçültecek biçimde seçilir. Bunun iki temel yolu vardır.

### En küçük kareler ve normal denklem

**En küçük kareler** yaklaşımı, artıkların kareleri toplamını en küçük yapan katsayıları arar. Tek öznitelikte en iyi uyumlu doğrunun eğimi şu şekilde hesaplanabilir:

`w = Σ((x - x_ort) * (y - y_ort)) / Σ((x - x_ort)^2)`

Ardından:

`b = y_ort - w * x_ort`

Burada `x_ort` ve `y_ort`, sırasıyla öznitelik ve hedef değerlerinin ortalamasıdır.

Örneğin `x = 1, 2, 3` ve `y = 2, 3, 5` için:

- `x_ort = 2`, `y_ort = 10/3`
- `w = 3/2`
- `b = 1/3`
- Uydurulan doğru: `ŷ = 1.5x + 1/3`

Bu doğru noktaların hepsinden geçmez; kareli hataların toplamını küçük tutacak şekilde veriye uyar.

Birden fazla öznitelik olduğunda katsayılar matrislerle **normal denklem** kullanılarak bulunabilir. Veri çok büyükse bu hesap pahalı olabilir; o zaman yinelemeli bir yöntem seçmek daha uygun olabilir.

### Gradyan iniş

Gradyan iniş, hatayı azaltan yönde model katsayılarını küçük adımlarla değiştirir:

1. `w` ve `b` için başlangıç değerleri seçilir.
2. Mevcut katsayılarla tahminler ve hatalar bulunur.
3. Hatanın katsayılara göre nasıl değiştiği hesaplanır. Bu değişim yönü **gradyan** olarak adlandırılır.
4. Katsayılar, gradyanın ters yönünde güncellenir.
5. Bu adımlar hata yeterince azalana veya belirlenen yineleme sayısına ulaşılana kadar tekrarlanır.

İki katsayı için, ortalama karesel hata kullanıldığında gradyanlar:

`dw = (2/n) * Σ((tahmin - gerçek) * x)`

`db = (2/n) * Σ(tahmin - gerçek)`

Güncelleme kuralı:

`w = w - öğrenme_oranı * dw`

`b = b - öğrenme_oranı * db`

Öğrenme oranı, her adımın büyüklüğünü belirler. Çok küçük seçilirse öğrenme yavaşlar; çok büyük seçilirse hata azalmak yerine artabilir veya yöntem en iyi noktayı aşabilir.

![Gradyan iniş sezgisi](gorseller/hafta01/denetimli_regresyon.gif)

Veriyi ölçeklemek gradyan inişin daha düzenli adımlar atmasına yardım edebilir. Birden fazla öznitelik farklı büyüklüklerdeyse öğrenme daha da zorlaşabilir.

### Gradyan iniş türleri

- **Batch gradyan iniş:** Her güncellemede eğitim verisinin tamamını kullanır.
- **Stokastik gradyan iniş:** Her güncellemede tek bir örnek kullanır.
- **Mini-batch gradyan iniş:** Her güncellemede küçük bir örnek grubunu kullanır.

Mini-batch yaklaşımı, hesaplama maliyeti ile güncellemelerin kararlılığı arasında pratik bir denge sağlar.

## Regresyonu değerlendirme ölçüleri

Test verisindeki tahminleri gerçek değerlerle karşılaştırarak modelin daha önce görmediği örneklerdeki başarısını ölçeriz. Veriyi ayırma ve ön işleme kuralları 2. haftadaki gibidir: hazırlama adımları yalnızca eğitim verisinden öğrenilmelidir.

- **MSE:** Hataların karelerinin ortalaması. Büyük hataları daha çok cezalandırır.
- **RMSE:** MSE'nin karekökü. Hata, hedefle aynı ölçü birimindedir.
- **MAE:** Hataların mutlak değerlerinin ortalaması. Hedefle aynı ölçü birimindedir ve çok büyük hatalara MSE kadar ağırlık vermez.
- **R²:** Modelin, basit bir ortalama tahminine kıyasla hedefteki değişimi ne ölçüde açıkladığını özetler. `1` kusursuz tahmine karşılık gelir; `0` ortalamayı tahmin etmekle aynı düzeydedir. Test verisinde negatif de olabilir.

Bu ölçülerin hiçbiri tek başına her şeyi anlatmaz. Örneğin büyük hatalar iş açısından özellikle önemliyse, MAE ile birlikte hata dağılımı ve büyük hataların hangi örneklerde oluştuğu da incelenmelidir.

## Polinom regresyon

Doğrusal model, özniteliklerle katsayıların doğrusal birleşimidir; ancak ilişkinin grafiği eğri olabilir. **Polinom regresyon**, `x²` veya `x³` gibi yeni öznitelikler ekleyerek eğri biçimli ilişkileri yakalayabilir.

Örneğin:

`ŷ = w0 + w1*x + w2*x²`

Daha yüksek dereceli polinomlar daha karmaşık eğriler oluşturabilir. Bu, eğitim verisine daha iyi uyum sağlasa da yeni veride mutlaka daha iyi tahmin yapacağı anlamına gelmez. Dereceyi seçerken modelin eğitim ve değerlendirme verisindeki performansını karşılaştırın.

## Python ile basit regresyon

Aşağıdaki kod yapay bir veri kümesi oluşturur, veriyi eğitim ve test parçalarına ayırır, doğrusal regresyon modelini eğitir ve test ölçülerini yazdırır. `make_regression` her çalıştırmada aynı veriyi üretmesi için `random_state` alır.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Eğim:", model.coef_[0])
print("Başlangıç değeri:", model.intercept_)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

Model yalnızca eğitim verisiyle öğrenir. Ölçüler ise ayrılan test verisindeki tahminlerden hesaplanır. `noise` değeri, hedefe rastgele değişkenlik ekler; bu nedenle noktaların bir doğru üzerinde kusursuz biçimde yer almasını beklemeyin. Çıktıdaki eğim ve başlangıç değeri, modelin eğitim verisinden öğrendiği katsayılardır.

## Uygulama fikri: California konut verisi

Bu haftanın laboratuvarında California konut veri seti üzerinde gerçek bir hedef olan konut değerini tahmin edebilirsiniz. Aynı eğitim-test ayırımını kullanarak şu üç yaklaşımı karşılaştırın:

1. scikit-learn `LinearRegression`
2. Normal denklem
3. Kendi yazdığınız gradyan iniş

Karşılaştırma yaparken modelleri aynı eğitim verisiyle eğitin ve aynı test verisinde MSE, MAE, RMSE ve R² değerlerini hesaplayın. Katsayılar veya ölçüler farklı çıkarsa, veri hazırlama, özellik ölçekleme, öğrenme oranı ve yineleme sayısını kontrol edin. Veri seti indirme gerektirebilir; internet bağlantısı olmayan bir ortamda indirme hatası alabilirsiniz.

## Kısa özet

- Doğrusal regresyon, sayısal bir hedefi özniteliklerin doğrusal birleşimiyle tahmin eder.
- Eğitim verisinde hatayı azaltmak için en küçük kareler veya gradyan iniş kullanılabilir.
- Gradyan inişte öğrenme oranı ve özniteliklerin ölçeği önemlidir.
- MSE, RMSE, MAE ve R² modelin farklı yönlerini gösterir; test verisinde hesaplanmalıdır.
- Polinom öznitelikler eğri ilişkileri yakalayabilir; model karmaşıklığını artırmak aşırı uyuma yol açabilir.

## Uygulama

1. Python örneğini çalıştırın. MSE ve RMSE arasındaki farkı kendi çıktınızla açıklayın.
2. `noise` değerini değiştirin. Modelin test ölçülerinin nasıl değiştiğini gözlemleyin.
3. Yukarıdaki üç nokta (`x = 1, 2, 3`, `y = 2, 3, 5`) için tahminleri ve artıkları hesaplayın.
4. Bir ev fiyatı tahmini için kullanabileceğiniz üç öznitelik yazın. Bunlardan hangilerinin hedef fiyatla ilişkili olabileceğini açıklayın.

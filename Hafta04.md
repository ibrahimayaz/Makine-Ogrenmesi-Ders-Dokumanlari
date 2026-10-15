# Hafta 4 — Genelleme ve Model Değerlendirme

## Bu haftanın sonunda

Bu notu bitirdiğinizde:

- Eksik öğrenme ile aşırı öğrenmeyi ayırt edebilirsiniz.
- Bir modelin yeni verilerde iyi çalışmasının ne anlama geldiğini açıklayabilirsiniz.
- k-katlı çapraz doğrulamanın nasıl yapıldığını anlayabilirsiniz.
- Ridge ve Lasso yöntemlerinin model karmaşıklığını nasıl azalttığını tanıyabilirsiniz.
- Öğrenme ve doğrulama eğrilerini kullanarak model sorunlarını inceleyebilirsiniz.

## Genelleme nedir?

Bir model, eğitim sırasında gördüğü örneklerden öğrenir. Ancak asıl amacımız yalnızca bu örneklerde başarılı olması değildir. Modelin, daha önce görmediği yeni örneklerde de iyi tahmin yapmasını isteriz. Buna **genelleme** denir.

Modelin başarısını anlamak için veriyi genellikle şu parçalara ayırırız:

- **Eğitim verisi:** Modelin öğrenmesi için kullanılır.
- **Doğrulama verisi:** Model seçmek ve ayarları karşılaştırmak için kullanılır.
- **Test verisi:** Seçilen modeli en son, daha önce kullanılmamış veride değerlendirmek için kullanılır.

Veri azsa, her ayar için ayrı bir doğrulama parçası ayırmak zor olabilir. k-katlı çapraz doğrulama bu durumda farklı eğitim ve doğrulama parçalarını sırayla kullanır. Test verisi varsa, model veya ayar seçmek için kullanılmamalı; en son değerlendirmeye saklanmalıdır.

## Eksik öğrenme ve aşırı öğrenme

### Eksik öğrenme

Model verideki temel ilişkiyi bile öğrenemeyecek kadar basitse **eksik öğrenme** oluşur. Örneğin, belirgin biçimde eğri bir ilişkiyi düz bir çizgiyle açıklamaya çalışmak yetersiz kalabilir.

Eksik öğrenmede genellikle hem eğitim verisindeki hem de doğrulama verisindeki hata yüksektir. Daha uygun özellikler kullanmak veya biraz daha esnek bir model seçmek yardımcı olabilir.

### Aşırı öğrenme

Model eğitim örneklerine gereğinden fazla uyum sağlarsa **aşırı öğrenme** oluşur. Model, genel ilişkiyi öğrenmek yerine eğitim verisindeki küçük ayrıntıları ve rastlantısal gürültüyü de ezberleyebilir.

Aşırı öğrenmede eğitim hatası düşük, doğrulama hatası ise daha yüksek olur. Modeli sadeleştirmek, daha fazla veri kullanmak veya düzenlileştirme uygulamak yardımcı olabilir.

| Durum | Eğitim hatası | Doğrulama hatası | Olası açıklama |
|---|---|---|---|
| Eksik öğrenme | Yüksek | Yüksek | Model fazla basit veya yeterince öğrenmemiş |
| Aşırı öğrenme | Düşük | Yüksek | Model eğitim verisine fazla uyum sağlamış |
| İyi genelleme | Düşük veya makul | Eğitim hatasına yakın ve makul | Model yeni veride de başarılı |

Hata değerinin “yüksek” ya da “düşük” olması probleme bağlıdır. Önemli olan, eğitim ve doğrulama sonuçlarını birlikte incelemektir.

## Yanlılık ve varyans dengesi

**Yanlılık (bias)**, modelin problemi fazla basit varsayımlarla açıklamasından kaynaklanan hatadır. **Varyans**, eğitim verisi değiştiğinde model tahminlerinin ne kadar değiştiğini anlatır.

Bir regresyon problemi için tahmin hatasını sezgisel olarak şöyle düşünebiliriz:

`beklenen hata ≈ yanlılığın karesi + varyans + verideki kaçınılmaz gürültü`

- Çok basit modelin yanlılığı yüksek olabilir: önemli örüntüleri yakalayamaz.
- Çok karmaşık modelin varyansı yüksek olabilir: eğitim verisindeki küçük değişikliklere aşırı duyarlı olur.
- Daha uygun bir model, bu iki sorunu dengelemeye çalışır.

Örneğin, ev fiyatı ile evin büyüklüğü arasındaki ilişkiyi düz bir çizgiyle açıklamak bazı durumlarda fazla basit kalabilir. Çok kıvrımlı bir eğri ise her eğitim noktasına yaklaşsa da yeni evlerde kötü tahmin yapabilir. Modeli seçerken yalnızca eğitim hatasının en küçük olmasına değil, doğrulama performansına da bakarız.

## k-katlı çapraz doğrulama

k-katlı çapraz doğrulamada veri yaklaşık eşit büyüklükte `k` parçaya ayrılır:

1. Parçalardan biri doğrulama için seçilir.
2. Kalan `k - 1` parça ile model eğitilir.
3. Model seçilen doğrulama parçasında değerlendirilir.
4. Bu işlem her parça bir kez doğrulama olana kadar tekrarlanır.
5. Sonuçların ortalaması genel performans tahmini olarak kullanılır.

Örneğin, 5 katlı çapraz doğrulamada model beş kez eğitilir ve her seferinde farklı bir parça üzerinde değerlendirilir. Böylece performans, tek bir şanslı veya şanssız veri ayırımına daha az bağlı olur.

### Beş katı elle oluşturma

Aşağıdaki örnek, on örnek için 5 kat oluşturur. Her turda bir kat doğrulama, kalan dört kat eğitim için kullanılır.

```python
import numpy as np

indeksler = np.arange(10)
np.random.default_rng(42).shuffle(indeksler)
katlar = np.array_split(indeksler, 5)

for sira, dogrulama_indeksleri in enumerate(katlar):
    egitim_indeksleri = np.concatenate([
        kat for i, kat in enumerate(katlar) if i != sira
    ])
    print("Tur:", sira + 1)
    print("Eğitim örnekleri:", egitim_indeksleri)
    print("Doğrulama örnekleri:", dogrulama_indeksleri)
```

Her turda doğrulama için ayrılan kat değişir. Böylece on örneğin her biri tam bir kez doğrulama örneği olur. Burada yalnızca katların nasıl oluşturulduğunu gösteriyoruz; aşağıdaki uygulamada modeli de her turda eğiteceğiz.

### Çapraz doğrulama türleri

- **KFold:** Sayısal hedefli veya genel problemlerde veriyi katlara ayırır.
- **StratifiedKFold:** Sınıflandırmada sınıf oranlarını katlarda benzer tutmaya çalışır.
- **GroupKFold:** Aynı kişiye, müşteriye veya gruba ait kayıtların hem eğitim hem doğrulama parçasına düşmesini önler.
- **TimeSeriesSplit:** Zaman sırasını korur; geçmiş verilerle eğitip daha sonraki dönemleri doğrular.

Verinin yapısına uygun bölme yöntemi seçilmelidir. Örneğin aynı hastaya ait kayıtlar iki tarafa da dağılırsa model hastayı tanımayı öğrenip başarısını olduğundan yüksek gösterebilir.

## Düzenlileştirme

Düzenlileştirme, modelin katsayılarını çok büyütmesini sınırlayarak aşırı öğrenmeyi azaltmaya çalışır. Doğrusal regresyonda hataya bir ceza eklenir. Ceza büyüklüğünü `λ` (lambda) veya scikit-learn'de `alpha` belirler.

- Ceza çok küçükse model katsayıları az kısıtlanır.
- Ceza artırıldıkça katsayılar küçülür ve model daha sade hale gelir.
- Ceza aşırı büyükse model önemli ilişkileri de öğrenemeyebilir ve eksik öğrenme oluşabilir.

Ölçekleri çok farklı olan öznitelikler düzenlileştirmeyi etkileyebilir. Bu nedenle ölçekleme kullanılacaksa, eğitim ve doğrulama katlarının içinde öğrenilmelidir. Aşağıdaki örneklerde `Pipeline` bunu her kat için doğru sırada yapar.

### Ridge (L2 düzenlileştirme)

Ridge, katsayıların karelerine ceza ekler:

`Hata = kareli tahmin hatalarının toplamı + λ × (katsayıların kareleri toplamı)`

Katsayıların büyüklüğünü azaltır; ancak çoğunlukla katsayıları tam olarak sıfırlamaz. Böylece model birçok özniteliği küçük katkılarla kullanmaya devam edebilir.

Öznitelik sütunlarının ortalaması çıkarılmış bir gösterimde, Ridge katsayıları şu kapalı form çözümüyle bulunabilir:

`w_Ridge = (XᵀX + λI)⁻¹Xᵀy`

Burada `X` öznitelikleri, `y` hedef değerleri, `w` katsayıları ve `I` birim matrisi gösterir. Bu gösterimde sabit terim ayrıca ele alınır ve ceza almaz. `λ` büyüdükçe katsayılar daha fazla küçültülür.

### Lasso (L1 düzenlileştirme)

Lasso, katsayıların mutlak değerlerine ceza ekler:

`Hata = kareli tahmin hatalarının toplamı + λ × (katsayıların mutlak değerleri toplamı)`

Lasso bazı katsayıları tam olarak sıfıra indirebilir. Bu nedenle model bazı öznitelikleri hiç kullanmamayı seçebilir; Lasso aynı zamanda basit bir öznitelik seçimi de yapabilir.

Lasso çoğunlukla **koordinat inişi** ile çözülür. Bu yöntemde her adımda bir katsayı seçilir, diğer katsayılar sabit tutulur ve seçilen katsayı güncellenir. Bu işlem katsayılar değişmeyi bırakana veya belirlenen yineleme sayısına ulaşana kadar sürer.

Katsayıyı küçültme işlemini **yumuşak eşikleme** ile gösterebiliriz:

`S(z, λ) = işaret(z) × en büyük(|z| - λ, 0)`

Örneğin `z = 0.8`, `λ = 0.3` ise sonuç `0.5` olur. `z = 0.2`, `λ = 0.3` ise sonuç `0` olur. Eşik değeri katsayıyı sıfıra indirebilir.

### Geometrik sezgi ve katsayı yolları

Ridge ve Lasso'yu, katsayıların ne kadar büyüyebileceğine farklı biçimde sınır koyan yöntemler olarak düşünebiliriz:

- Ridge'in ceza sınırı yuvarlak bir bölge gibi düşünülebilir. Katsayılar genellikle küçülür ama tam sıfıra gelmeleri daha seyrektir.
- Lasso'nun ceza sınırı köşeli bir bölge gibi düşünülebilir. Çözüm köşelere denk geldiğinde bazı katsayılar sıfır olur.

**Katsayı yolu**, `λ` değişirken her model katsayısının nasıl değiştiğini gösteren grafiktir. Yatay eksende `λ`, dikey eksende katsayılar yer alır. `λ` arttıkça Ridge katsayıları sıfıra yaklaşır; Lasso'da bazı katsayı yolları sıfıra ulaşıp orada kalabilir. Öznitelikler ölçeklenmeden bu yolları karşılaştırmak yanıltıcı olabilir.

## Öğrenme ve doğrulama eğrileri

### Öğrenme eğrisi

Öğrenme eğrisi, eğitim örneği sayısı arttıkça eğitim ve doğrulama hatasının nasıl değiştiğini gösterir.

- Her iki hata da yüksekse model eksik öğreniyor olabilir.
- Eğitim hatası düşük, doğrulama hatası belirgin biçimde yüksekse aşırı öğrenme olabilir.
- Daha fazla eğitim verisiyle doğrulama hatası düşüyor ve iki hata birbirine yaklaşıyorsa ek veri fayda sağlayabilir.

### Doğrulama eğrisi

Doğrulama eğrisi, bir model ayarı değişirken eğitim ve doğrulama başarısının nasıl değiştiğini gösterir. Örneğin Ridge veya Lasso için farklı `alpha` değerlerini deneriz. Doğrulama hatasının en düşük olduğu bölge, denenmiş değerler arasındaki iyi ayarları gösterir.

Ayarlara karar verirken test verisini tekrar tekrar kullanmayın. Öğrenme ve doğrulama eğrilerini eğitim verisi üzerindeki çapraz doğrulama ile oluşturun; test verisini son değerlendirmeye saklayın.

### Eğrileri çizme örneği

Aşağıdaki kod, Diabetes verisi üzerinde Ridge modeli için öğrenme eğrisini ve `alpha` değerine göre doğrulama eğrisini çizer.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, learning_curve, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
katlar = KFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

boyutlar, egitim, dogrulama = learning_curve(
    model, X, y, cv=katlar, train_sizes=np.linspace(0.2, 1.0, 5),
    scoring="neg_mean_squared_error"
)
plt.plot(boyutlar, np.sqrt(-egitim.mean(axis=1)), marker="o", label="Eğitim")
plt.plot(boyutlar, np.sqrt(-dogrulama.mean(axis=1)), marker="o", label="Doğrulama")
plt.xlabel("Eğitim örneği sayısı")
plt.ylabel("RMSE")
plt.title("Öğrenme eğrisi")
plt.legend()
plt.show()

alpha_degerleri = np.logspace(-3, 3, 7)
egitim, dogrulama = validation_curve(
    model, X, y, param_name="ridge__alpha", param_range=alpha_degerleri,
    cv=katlar, scoring="neg_mean_squared_error"
)
plt.semilogx(alpha_degerleri, np.sqrt(-egitim.mean(axis=1)), marker="o", label="Eğitim")
plt.semilogx(alpha_degerleri, np.sqrt(-dogrulama.mean(axis=1)), marker="o", label="Doğrulama")
plt.xlabel("Düzenlileştirme değeri (alpha)")
plt.ylabel("RMSE")
plt.title("Doğrulama eğrisi")
plt.legend()
plt.show()
```

İlk grafik, eğitim örneği sayısı değiştikçe hataların nasıl değiştiğini gösterir. İkinci grafik, `alpha` değiştikçe eğitim ve doğrulama hatalarının nasıl etkilendiğini gösterir. Her iki grafikte de daha düşük RMSE daha iyi tahmin demektir. Eğitim ve doğrulama eğrileri arasındaki farkı ve doğrulama hatasının en düşük olduğu `alpha` değerlerini inceleyin.

## Diabetes verisinde çapraz doğrulama

scikit-learn'deki Diabetes veri seti, sayısal sağlık ölçümlerinden hastalığın ilerleme ölçüsünü tahmin etmek için kullanılır. Aşağıdaki kodda aynı 5 katı doğrusal regresyon, Ridge ve Lasso modelleri için kullanıyoruz.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
katlar = KFold(n_splits=5, shuffle=True, random_state=42)

modeller = {
    "Doğrusal": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000)),
}

for ad, model in modeller.items():
    mse_degerleri = cross_val_score(
        model, X, y, cv=katlar, scoring="neg_mean_squared_error"
    )
    rmse_degerleri = np.sqrt(-mse_degerleri)
    print(f"{ad}: ortalama RMSE = {rmse_degerleri.mean():.2f}")
```

RMSE, tahmin hatasını hedefle aynı ölçü biriminde gösterir; bu karşılaştırmada daha küçük olması daha iyidir. `cross_val_score` her kat için ayrı model eğitir. `Pipeline` içindeki ölçekleyici yalnızca o turun eğitim katından öğrenilir ve doğrulama katına uygulanır; böylece veri sızıntısı önlenir. En iyi görünen sonucu seçerken yalnızca tek bir ölçüye değil, problem amacına da bakın. Nihai başarıyı ölçmek için varsa ayrılmış test verisini model seçimi tamamlandıktan sonra kullanın.

## Uygulama

1. Diabetes örneğinde 5 katı elle oluşturma kodunu çalıştırın. Her örneğin bir kez doğrulama katında yer aldığını kontrol edin.
2. Karşılaştırma kodunda Ridge için `alpha` değerini `0.1`, `1` ve `10` yapın. Ortalama RMSE nasıl değişiyor?
3. `Lasso` sonucunda katsayıların ne kadarının sıfır olduğunu inceleyin. `alpha` artınca ne olacağını tahmin edin.
4. Bir öğrenme eğrisi ve bir doğrulama eğrisi çizin. Grafiği kullanarak modelin eksik mi, aşırı mı öğrendiğini tartışın.
5. Veriniz aynı kişilere ait birden çok kayıt içeriyorsa neden `GroupKFold` kullanmanız gerekebilir?

## Kısa özet

- Genelleme, modelin daha önce görmediği verilerde de iyi tahmin yapabilmesidir.
- Eksik öğrenmede model temel ilişkiyi yakalayamaz; aşırı öğrenmede eğitim verisini fazla ezberler.
- k-katlı çapraz doğrulama, farklı doğrulama parçalarını sırayla kullanarak daha güvenilir bir performans tahmini sağlar.
- Ridge katsayıları küçültür; Lasso bazı katsayıları sıfırlayabilir.
- Öğrenme eğrileri örnek sayısının, doğrulama eğrileri ise model ayarlarının etkisini gösterir.
- Model seçimi için doğrulama verisini veya çapraz doğrulamayı kullanın; test verisini en son değerlendirmeye saklayın.

## Kendinizi sınayın

1. Eğitim hatası düşük, doğrulama hatası yüksekse modelde hangi sorun olabilir?
2. Eğitim ve doğrulama hatalarının ikisi de yüksekse olası açıklama nedir?
3. 5 katlı çapraz doğrulamada model kaç kez eğitilir ve her örnek kaç kez doğrulama için kullanılır?
4. Ridge ve Lasso'nun katsayılar üzerindeki etkileri nasıl farklıdır?
5. Lasso'nun bazı katsayıları sıfırlaması neden öznitelik seçimine yardımcı olur?
6. Düzenlileştirme cezasını çok artırmak hangi soruna yol açabilir?
7. Test verisini model ayarı seçmek için kullanmak neden sakıncalıdır?

# Makine Öğrenmesi — Ders Notları

Bu depo, 14 haftalık bir **Makine Öğrenmesine Giriş** dersinin haftalık ders notlarını içerir. Notlar Türkçe hazırlanmıştır ve tüm uygulamalar **Python** ile yapılır.

## Dersin Amacı

Bu ders, makine öğrenmesinin temel kavramlarını, en yaygın algoritmaları ve gerçek bir ML projesinin uçtan uca nasıl yürütüldüğünü öğretmeyi amaçlar. Algoritmalar yalnızca kütüphane çağrıları olarak değil, **nasıl çalıştıkları** gösterilerek anlatılır.

Her hafta şu yaklaşımla ilerler:

1. **Sezgi:** Konu günlük hayattan örnekler ve benzetmelerle tanıtılır.
2. **Çalışma mantığı:** Algoritma, küçük bir veri üzerinde **adım adım elle hesaplanarak** gösterilir.
3. **Sıfırdan kodlama:** Algoritma NumPy ile sıfırdan yazılır.
4. **Kütüphane ile karşılaştırma:** Sonuç scikit-learn / PyTorch ile doğrulanır.
5. **Laboratuvar:** Gerçek bir veri setinde uygulama yapılır.

## Öğrenme Çıktıları

Dersi tamamlayan öğrenciler:

- Denetimli, denetimsiz ve pekiştirmeli öğrenmeyi ayırt edebilir ve bir problemi doğru türde tanımlayabilir.
- Veriyi keşfedebilir, temizleyebilir ve **veri sızıntısına** yol açmadan modellemeye hazırlayabilir.
- Doğrusal/lojistik regresyon, k-NN, Naive Bayes, karar ağaçları, topluluk yöntemleri, SVM, kümeleme ve boyut indirgeme algoritmalarının çalışma mantığını açıklayabilir ve uygulayabilir.
- Modelleri çapraz doğrulama ve uygun metriklerle değerlendirebilir; aşırı ve eksik öğrenmeyi teşhis edebilir.
- Sinir ağlarının ve geri yayılımın temel mantığını açıklayabilir; PyTorch ile basit modeller ve transfer öğrenme uygulayabilir.
- Uçtan uca bir ML pipeline'ı kurabilir, modeli kaydedip sunabilir, açıklayabilir ve adillik açısından değerlendirebilir.

## Ön Koşullar

- Temel Python programlama bilgisi
- Temel doğrusal cebir (vektör, matris çarpımı), türev kavramı ve olasılık/istatistik bilgisi

## Haftalık Ders Akışı

| Hafta | Konu | Kapsam | Laboratuvar / Uygulama | Ders Notu |
|---|---|---|---|---|
| 1 | Makine öğrenmesine giriş | ML nedir; denetimli, denetimsiz ve pekiştirmeli öğrenme; ML iş akışı; uygulama alanları | Ortam kurulumu (Jupyter/Colab), NumPy ve pandas tekrarı | [Hafta01.md](Hafta01.md) |
| 2 | Veri hazırlama ve keşifsel veri analizi | Eksik veri, aykırı değerler, ölçekleme, kategorik kodlama, eğitim/doğrulama/test ayrımı, veri sızıntısı | Gerçek bir veri setinde keşifsel analiz ve ön işleme (Titanic) | [Hafta02.md](Hafta02.md) |
| 3 | Doğrusal regresyon | En küçük kareler, kapalı form çözümü, gradyan iniş, polinom regresyon, MSE/MAE/R² | Gradyan inişi sıfırdan kodlama, sonucu scikit-learn ile karşılaştırma | [Hafta03.md](Hafta03.md) |
| 4 | Genelleme ve model değerlendirme | Aşırı ve eksik öğrenme, yanlılık-varyans ödünleşimi, k-katlı çapraz doğrulama, Ridge/Lasso | Öğrenme eğrileri, düzenlileştirme katsayısının etkisi | [Hafta04.md](Hafta04.md) |
| 5 | Lojistik regresyon ve sınıflandırma metrikleri | Sigmoid, çapraz entropi kaybı, karmaşıklık matrisi, precision/recall/F1, ROC-AUC, dengesiz veri | İkili sınıflandırma, eşik değeri analizi | [Hafta05.md](Hafta05.md) |
| 6 | Örnek tabanlı ve olasılıksal modeller | k-NN, mesafe ölçüleri, boyutluluk laneti, Bayes teoremi, Naive Bayes | Metin sınıflandırma (spam tespiti) | [Hafta06.md](Hafta06.md) |
| 7 | Karar ağaçları | Entropi, bilgi kazancı, Gini, budama, yorumlanabilirlik | Ağaç görselleştirme; **proje önerisi teslimi** | [Hafta07.md](Hafta07.md) |
| — | **Ara sınav** | 1–7. haftalar | | |
| 8 | Topluluk yöntemleri | Bagging, Random Forest, boosting (AdaBoost, Gradient Boosting), XGBoost/LightGBM'e genel bakış | Tablo verisinde model karşılaştırması | [Hafta08.md](Hafta08.md) |
| 9 | Destek vektör makineleri | Maksimum marj, yumuşak marj, kernel hilesi (RBF, polinom) | Karar sınırı görselleştirme; Pipeline ile GridSearchCV | [Hafta09.md](Hafta09.md) |
| 10 | Kümeleme | k-means, hiyerarşik kümeleme, DBSCAN, siluet skoru, dirsek yöntemi | Müşteri segmentasyonu | [Hafta10.md](Hafta10.md) |
| 11 | Boyut indirgeme ve öznitelik seçimi | PCA (özdeğer yorumu), t-SNE ve UMAP ile görselleştirme, filtre/sarmalayıcı yöntemler | MNIST üzerinde PCA ve t-SNE | [Hafta11.md](Hafta11.md) |
| 12 | Yapay sinir ağlarına giriş | Perceptron, çok katmanlı ağ, aktivasyon fonksiyonları, geri yayılım, SGD/Adam, dropout | PyTorch veya Keras ile MLP | [Hafta12.md](Hafta12.md) |
| 13 | Derin öğrenmeye bakış | CNN temelleri (evrişim, havuzlama), transfer öğrenme; dizi modelleri ve Transformer'a kısa giriş | Önceden eğitilmiş bir CNN ile görüntü sınıflandırma | [Hafta13.md](Hafta13.md) |
| 14 | Uygulamada ML ve etik | Uçtan uca pipeline, model kaydetme ve dağıtıma giriş, açıklanabilirlik (SHAP), yanlılık ve adillik | **Dönem projesi sunumları** | [Hafta14.md](Hafta14.md) |

## Haftalara Göre Ayrıntılı İçerik

### Hafta 1 — Makine Öğrenmesine Giriş
- Klasik programlama ile makine öğrenmesinin karşılaştırılması (spam örneği)
- Mitchell'in tanımı (T, P, E); örnek, öznitelik, hedef, model kavramları
- Denetimli (regresyon, sınıflandırma), denetimsiz ve pekiştirmeli öğrenme
- ML iş akışı ve iris verisiyle uçtan uca ilk model
- Uygulama alanları; ML'in ne zaman uygun olduğu
- **Lab:** Colab/yerel kurulum, NumPy (vektörel işlemler, broadcasting, `axis`) ve pandas (filtreleme, eksik veri, `groupby`)

### Hafta 2 — Veri Hazırlama ve Keşifsel Veri Analizi
- EDA: tek ve iki değişkenli analiz, kutu grafiği okuma, Pearson korelasyonunun elle hesabı
- Eksik veri türleri (MCAR, MAR, MNAR) ve doldurma stratejileri
- Aykırı değer tespiti: IQR ve Z-skoru (elle örnek)
- Ölçekleme: standardizasyon, min-max, robust; sıfırdan `StandardScaler`
- Kategorik kodlama: ordinal ve one-hot (sıfırdan)
- Eğitim/doğrulama/test ayrımı, `stratify`, zaman serisi bölme
- Veri sızıntısı türleri ve `Pipeline` / `ColumnTransformer` ile önlenmesi
- **Lab:** Titanic üzerinde EDA, öznitelik türetme ve sızıntı deneyi

### Hafta 3 — Doğrusal Regresyon
- Model `ŷ = wx + b`, artıklar ve MSE kaybı
- Kapalı form çözümün türetilmesi ve elle hesap; normal denklem
- Gradyan iniş: sezgi, gradyanların türetilmesi, bir adımın elle hesabı
- Öğrenme oranının ve ölçeklemenin etkisi; batch / stokastik / mini-batch GD
- MSE, RMSE, MAE, R² metriklerinin elle hesabı
- Polinom regresyon ve derecenin etkisi
- **Lab:** California konut verisinde kendi GD'miz, normal denklem ve scikit-learn karşılaştırması

### Hafta 4 — Genelleme ve Model Değerlendirme
- Eksik ve aşırı öğrenmenin belirtileri ve çözümleri
- Yanlılık-varyans ayrıştırması ve simülasyonla ölçülmesi
- k-katlı çapraz doğrulama (sıfırdan) ve CV türleri
- Ridge (kapalı form) ve Lasso (koordinat inişi, yumuşak eşikleme); geometrik sezgi ve katsayı yolları
- Öğrenme eğrileri ve doğrulama eğrileri ile teşhis
- **Lab:** Öğrenme eğrileri, λ'nın etkisi, Lasso ile öznitelik seçimi

### Hafta 5 — Lojistik Regresyon ve Sınıflandırma Metrikleri
- Sigmoid, karar sınırı, odds ve katsayı yorumu
- Çapraz entropi kaybı, gradyanın türetilmesi ve bir adımın elle hesabı
- Sıfırdan lojistik regresyon; softmax ile çok sınıflı sınıflandırma
- Karmaşıklık matrisi, accuracy, precision, recall, F1
- ROC eğrisi ve AUC'nin elle hesabı; precision-recall eğrisi
- Dengesiz veri: sınıf ağırlıkları, yeniden örnekleme, SMOTE
- **Lab:** Meme kanseri verisinde eşik analizi ve maliyete dayalı eşik seçimi

### Hafta 6 — Örnek Tabanlı ve Olasılıksal Modeller
- k-NN'in adım adım çalışması, k seçimi, ağırlıklı oylama
- Mesafe ölçüleri: Öklid, Manhattan, Chebyshev, Minkowski, kosinüs
- Boyutluluk laneti (sayısal deney)
- Bayes teoremi ve tıbbi test paradoksu
- Naive Bayes: elle spam sınıflandırma, Laplace düzeltmesi, log olasılıklar
- **Lab:** SMS spam tespiti; Naive Bayes, k-NN ve lojistik regresyon karşılaştırması

### Hafta 7 — Karar Ağaçları
- Entropi, Gini, bilgi kazancı
- Tenis veri setinde ağacın elle inşası
- Sayısal özniteliklerde eşik seçimi; CART ve regresyon ağaçları
- Sıfırdan karar ağacı
- Ön budama ve maliyet-karmaşıklık budaması
- Yorumlanabilirlik ve öznitelik önemi
- **Lab:** Ağaç görselleştirme ve karar sınırları; **proje önerisi teslimi**

### Ara Sınav — 1–7. haftalar

### Hafta 8 — Topluluk Yöntemleri
- Kalabalığın bilgeliği: oylamanın olasılık hesabı
- Bootstrap, OOB, bagging ve varyans azaltma
- Random Forest (sıfırdan, OOB skoru ile)
- AdaBoost: ağırlık güncellemesinin elle hesabı ve sıfırdan uygulama
- Gradient Boosting: artıklar üzerinden elle örnek, fonksiyon uzayında gradyan iniş
- XGBoost, LightGBM, CatBoost; oylama ve yığınlama (stacking)
- **Lab:** Adult veri setinde adil model karşılaştırması

### Hafta 9 — Destek Vektör Makineleri
- Maksimum marj ve destek vektörleri (elle örnek)
- Yumuşak marj, C parametresi, hinge kaybı; sıfırdan doğrusal SVM
- Kernel hilesi: sayısal doğrulama; RBF ve γ'nın etkisi
- Pratik bilgiler: ölçekleme, çok sınıflı SVM, SVR
- Pipeline + GridSearchCV / RandomizedSearchCV, iç içe CV
- **Lab:** Kernel karşılaştırması, C-γ ızgarası, gerçek veride hiperparametre araması

### Hafta 10 — Kümeleme
- k-means: iterasyonların elle hesabı, sıfırdan kod, k-means++
- Hiyerarşik kümeleme: bağlantı türleri, elle örnek, dendrogram okuma
- DBSCAN: çekirdek, sınır ve gürültü noktaları; elle örnek ve sıfırdan kod
- Dirsek yöntemi ve siluet skorunun elle hesabı
- Yöntemlerin farklı veri şekillerinde karşılaştırılması
- **Lab:** Müşteri segmentasyonu ve segment profilleri

### Hafta 11 — Boyut İndirgeme ve Öznitelik Seçimi
- PCA: kovaryans, özdeğer ve özvektörlerin elle hesabı; sıfırdan PCA; SVD
- Açıklanan varyans ve bileşen sayısının seçimi
- t-SNE ve UMAP'in çalışma mantığı ve grafiklerinin doğru okunması
- Öznitelik seçimi: filtre, sarmalayıcı (sıfırdan ileri seçim, RFE) ve gömülü yöntemler
- **Lab:** MNIST üzerinde PCA, yeniden oluşturma, t-SNE görselleştirmesi

### Hafta 12 — Yapay Sinir Ağlarına Giriş
- Yapay nöron ve perceptron (VE kapısının elle eğitimi), XOR problemi
- MLP yapısı, ileri yayılım ve parametre sayımı
- Aktivasyon fonksiyonları ve kaybolan gradyan sorunu
- Geri yayılımın sayılarla tam hesabı
- Sıfırdan NumPy MLP
- SGD, momentum, Adam; ağırlık başlatma; dropout ve diğer düzenlileştirme teknikleri
- **Lab:** PyTorch ile Fashion-MNIST üzerinde MLP (alternatif olarak Keras)

### Hafta 13 — Derin Öğrenmeye Bakış
- Evrişimin elle hesabı, dolgu, adım, çıktı boyutu formülü
- Havuzlama, CNN mimarisi ve öznitelik hiyerarşisi; ResNet
- Veri artırma ve transfer öğrenme (öznitelik çıkarma ve ince ayar)
- RNN, LSTM; dikkat mekanizmasının elle hesabı; Transformer
- **Lab:** Önceden eğitilmiş ResNet-18 ile görüntü sınıflandırma (sıfırdan eğitim, öznitelik çıkarma ve ince ayar karşılaştırması)

### Hafta 14 — Uygulamada ML ve Etik
- Uçtan uca pipeline ve özel dönüştürücüler
- Model kaydetme (joblib, PyTorch, ONNX) ve sürümleme
- FastAPI ile tahmin servisi; veri/kavram kayması ve PSI ile izleme
- Açıklanabilirlik: permütasyon önemi, Shapley değerlerinin elle hesabı, SHAP
- Yanlılık kaynakları, adillik metrikleri, imkânsızlık teoremi, azaltma yöntemleri, model kartları
- **Lab:** Dönem projesi sunumları (sunum yapısı, kontrol listesi, değerlendirme ölçütleri)

## Değerlendirme

| Bileşen | Kapsam |
|---|---|
| Ara sınav | 1–7. haftalar |
| Dönem projesi | Öneri (7. hafta), rapor, kod, model kartı ve sunum (14. hafta) |
| Final sınavı | Tüm dönem |

Ağırlıklar, dersi veren öğretim elemanı tarafından ders başında duyurulur. Proje değerlendirme ölçütleri [Hafta14.md](Hafta14.md) içinde yer alır.

## Kurulum

**Seçenek A — Google Colab:** Kurulum gerektirmez. Temel kütüphaneler hazır gelir.

**Seçenek B — Yerel ortam:**

```bash
python -m venv ml-ders
# Windows
ml-ders\Scripts\activate
# macOS / Linux
source ml-ders/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

**İlgili haftalarda gereken ek paketler:**

| Paket | Hafta |
|---|---|
| `imbalanced-learn` | 5 |
| `xgboost`, `lightgbm` | 8 |
| `umap-learn` | 11 |
| `torch`, `torchvision` (veya `tensorflow`) | 12, 13 |
| `transformers` | 13 (isteğe bağlı) |
| `shap`, `fairlearn`, `fastapi`, `uvicorn`, `joblib` | 14 |

```bash
pip install imbalanced-learn xgboost lightgbm umap-learn torch torchvision shap fairlearn fastapi uvicorn
```

> Laboratuvarlarda kullanılan veri setlerinin çoğu (seaborn, OpenML, torchvision) kod çalıştırıldığında internetten indirilir. Bu yüzden ilk çalıştırmada internet bağlantısı gerekir.

## Kullanılan Veri Setleri

| Veri seti | Hafta | Kaynak |
|---|---|---|
| Iris | 1, 7 | scikit-learn |
| Titanic | 1, 2 | seaborn |
| California Housing | 3 | scikit-learn |
| Diabetes | 4 | scikit-learn |
| Breast Cancer Wisconsin | 5, 7, 8, 9, 11 | scikit-learn |
| SMS Spam Collection | 6 | UCI (GitHub kopyası) |
| Adult Census Income | 8, 14 | OpenML |
| Sentetik müşteri verisi / Mall Customers | 10 | Kodla üretilir / Kaggle |
| Digits, MNIST | 11 | scikit-learn, OpenML |
| Fashion-MNIST | 12 | torchvision |
| Hymenoptera (karınca/arı) | 13 | PyTorch eğitim materyalleri |

## Dosya Yapısı

Her hafta için iki dosya bulunur: okunabilir ders notu (`.md`) ve aynı içeriğin çalıştırılabilir Jupyter not defteri (`.ipynb`).

```
ML Ders Dokümanı/
├── README.md                      ← bu dosya
├── Hafta01.md / Hafta01.ipynb     Makine öğrenmesine giriş
├── Hafta02.md / Hafta02.ipynb     Veri hazırlama ve keşifsel veri analizi
├── Hafta03.md / Hafta03.ipynb     Doğrusal regresyon
├── Hafta04.md / Hafta04.ipynb     Genelleme ve model değerlendirme
├── Hafta05.md / Hafta05.ipynb     Lojistik regresyon ve sınıflandırma metrikleri
├── Hafta06.md / Hafta06.ipynb     Örnek tabanlı ve olasılıksal modeller
├── Hafta07.md / Hafta07.ipynb     Karar ağaçları
├── Hafta08.md / Hafta08.ipynb     Topluluk yöntemleri
├── Hafta09.md / Hafta09.ipynb     Destek vektör makineleri
├── Hafta10.md / Hafta10.ipynb     Kümeleme
├── Hafta11.md / Hafta11.ipynb     Boyut indirgeme ve öznitelik seçimi
├── Hafta12.md / Hafta12.ipynb     Yapay sinir ağlarına giriş
├── Hafta13.md / Hafta13.ipynb     Derin öğrenmeye bakış
└── Hafta14.md / Hafta14.ipynb     Uygulamada ML ve etik
```

### Jupyter not defterleri hakkında

- Not defterleri, ders notlarındaki anlatımı markdown hücrelerinde, Python örneklerini ise kod hücrelerinde içerir.
- Hücreler **yukarıdan aşağıya sırayla** çalıştırılmalıdır; sonraki hücreler önceki hücrelerde tanımlanan değişkenleri kullanır.
- Her not defterinin başında, o hafta gereken paketleri kuran (yorum satırı halinde) bir kurulum hücresi vardır.
- Tek başına çalışmayan gösterim amaçlı kod parçaları (ör. "yanlış kullanım" örnekleri, FastAPI servis dosyası) markdown içinde kod bloğu olarak bırakılmıştır.
- İsteğe bağlı paket (xgboost, lightgbm, umap-learn, shap, fairlearn, tensorflow, transformers) kullanan hücreler, paket kurulu değilse hata vermek yerine uyarı yazar.
- Not defterinin sonunda alıştırmalar için boş bir **Çalışma Alanı** bulunur.

## Notları Nasıl Kullanmalı?

1. Her haftanın notunu derse gelmeden önce okuyun. Özellikle elle hesaplama örneklerini kâğıt üzerinde siz de yapın.
2. "Sıfırdan" yazılan kodları kopyalamak yerine kendiniz yazmaya çalışın, sonra kütüphane sonucuyla karşılaştırın.
3. Haftanın `.ipynb` not defterini Jupyter veya Colab'da açıp kod hücrelerini sırayla çalıştırın; laboratuvar görevlerini aynı not defterinde tamamlayın.
4. Her notun sonundaki **"Kendinizi Sınayın"** sorularıyla konuyu pekiştirin.

## Önerilen Kaynaklar

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, O'Reilly
- Gareth James ve ark., *An Introduction to Statistical Learning (with Python)* — ücretsiz: statlearning.com
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Springer
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning* — ücretsiz: deeplearningbook.org
- scikit-learn kullanıcı kılavuzu: scikit-learn.org/stable/user_guide.html
- PyTorch eğitimleri: pytorch.org/tutorials

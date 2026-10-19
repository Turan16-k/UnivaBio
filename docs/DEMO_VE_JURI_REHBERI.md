# LittleHeart: demo videosu ve jüri hazırlık rehberi

Bu rehber senin için. Videoda ve jüriyle konuşurken kullanacağın cümleler İngilizce, açıklamalar Türkçe.

## Teslimden önce yapılacaklar (sırayla)

1. **GitHub'a yükle** (yaklaşık 10 dakika)
   - github.com'da yeni ve **public** bir repo aç, adı `littleheart` olsun. README ekleme seçeneğini işaretleme.
   - `littleheart` klasöründe sırayla şu komutları çalıştır:
     ```
     git init
     git add .
     git commit -m "LittleHeart: AI heart murmur screening for children"
     git branch -M main
     git remote add origin https://github.com/KULLANICI_ADIN/littleheart.git
     git push -u origin main
     ```
   - Repo'da **Settings → Pages → Source: GitHub Actions** seç. Birkaç dakika sonra canlı link oluşur: `https://KULLANICI_ADIN.github.io/littleheart/`
   - Linki aç, "Case 1"e tıkla ve çalıştığını gör.
2. **Videoyu çek** (aşağıdaki senaryoyla, 3 dakika). Windows'ta Win + Alt + R ekran kaydını başlatır. Daha iyi sonuç için OBS kullanabilirsin.
3. **Devpost'ta teslim et:** proje adı, kısa açıklama, video linki (YouTube'a liste dışı olarak yükleyebilirsin), GitHub linki, canlı demo linki, `docs/LittleHeart_Project_Description.pdf` ve `docs/LittleHeart_Code.pdf`.
4. **Son tarih:** 20 Ekim 23:45 EDT, yani Türkiye saatiyle **21 Ekim 06:45**. Son saate bırakma.

## Demo videosu senaryosu (yaklaşık 3 dakika)

**0:00–0:25 · Sorun** (About murmurs sekmesi açık)
> "About one in a hundred babies is born with a heart defect, and rheumatic heart disease affects tens of millions of young people, mostly in low-income countries. Both often show up first as a heart murmur. But hearing a murmur takes years of training, and the echocardiogram that confirms it is scarce exactly where these diseases are most common. So I built LittleHeart."

**0:25–1:35 · Canlı demo** (Screen a child sekmesi)
- **Case 1**'e tıkla: "These are real children from a Brazilian screening dataset that my model never saw during training."
- Sonuç kutusunu göster: "LittleHeart says: murmur suspected, refer for an echocardiogram. And here is the cardiologist's actual finding, revealed only after the analysis. It agreed."
- Göğüs şemasını göster: "The chest diagram shows which valve areas sounded abnormal."
- Bir kaydın **▶** düğmesine bas, sesi çal: "This is the spectrogram, a picture of the sound. Below it, the bars show how likely a murmur is in each 4-second window. Red bars cross the referral threshold."
- Sonucu "No murmur" çıkan bir vaka aç (vaka listesi aşağıda): "Here is a healthy heart: green everywhere."
- **Record** sekmesini göster: "A health worker can record live with a stethoscope adapter, or upload files."
- İsteğe bağlı: mikrofonla konuşarak kayıt yap. "If I just talk into it, LittleHeart refuses to guess: no heartbeat found, record again. That's the safety check."
- "Print screening note" düğmesini göster.

**1:35–2:30 · Nasıl çalışıyor, nasıl test edildi** (Validation sekmesi)
> "Each recording becomes a spectrogram, and a small convolutional neural network that I trained in PyTorch scores every 4-second window. It runs entirely in the browser, so no recording ever leaves the device."
- Skor kartlarını göster: "I tested it with cross-validation split by child, so the model never hears the same child in training and testing. On the official PhysioNet 2022 challenge score it reaches __CV_SCORE__, compared to __BASE_SCORE__ for a simple baseline. The best team in the international challenge scored 0.78 on their hidden test set."
- Karışıklık matrisini göster: "It catches __SENS__ of children with murmurs and clears __SPEC__ of those without."
- Güvenlik kontrolü kartı: "Because the AI has only heard heart sounds, it first checks there's a real heartbeat rhythm. Noise and tones are always rejected."

**2:30–3:00 · Dürüstlük ve sonraki adımlar** (Honest status kartı)
> "This is a research prototype, not a diagnosis. It still misses some murmurs, and it was trained on one population and one stethoscope. Next I want to test it with low-cost stethoscope adapters and partner with a school screening programme for rheumatic heart disease."

## Jüriye kodu anlatırken: soru ve cevaplar

| Muhtemel soru | Cevap ve göstereceğin yer |
|---|---|
| Veri nereden? | PhysioNet'teki CirCor DigiScope veri seti: 942 çocuk, 3.163 kayıt, kardiyologlar etiketledi. İndirme: `ml/download_data.py` |
| Ses nasıl sayıya dönüşüyor? | `ml/features.py` içindeki `log_mel`. Ses 64 ms'lik parçalara bölünür, her parçanın frekans dağılımı FFT ile hesaplanır, 32 mel bandında toplanır ve logaritması alınır. Ortaya bir "ses resmi" çıkar. |
| Model nedir? | `ml/train.py` içindeki `MurmurCNN`: 4 evrişim (convolution) bloğu, ortalama havuzlama (global average pooling) ve 3 çıkış (murmur yok / var / belirsiz). Yaklaşık 61.000 ağırlık. |
| Neden 4 saniyelik pencereler? | 4 saniyede 4–8 kalp atışı olur, model her pencereye ayrı karar verir. Böylece "kaydın neresinde üfürüm var" da gösterilebilir. |
| Çocuk bazında karar nasıl veriliyor? | `decide` fonksiyonu: herhangi bir kapakçıkta üfürüm eşiği aşılırsa sevk. Aşılmazsa ve bir kayıt belirsizse tekrar kayıt. Aksi halde üfürüm yok. PhysioNet yarışmasındaki sıra da böyle. |
| Ezber (overfitting) yapmadığını nereden biliyorsun? | Veriyi kayıt bazında değil çocuk bazında böldüm, aynı çocuk hem eğitimde hem testte yok. 5 katlı çapraz doğrulama yaptım. Eşikleri test verisinde değil, ayrı bir doğrulama verisinde seçtim. Demo vakaları eğitime hiç girmedi. |
| Neden ağırlıklı doğruluk (weighted accuracy)? | Yarışmanın resmi ölçütü bu. Üfürümü kaçırmak daha tehlikeli olduğu için üfürüm vakaları 5, belirsizler 3, normaller 1 kat sayılıyor. Kod: `weighted_accuracy` |
| Tarayıcıda nasıl çalışıyor? | Model ONNX formatına çevrildi, ONNX Runtime Web ile çalışıyor (`web/js/model.js`). Spektrogram kodu JavaScript'e birebir çevrildi (`web/js/features.js`). `tests/test_pipeline.py` iki dilin aynı sayıları ürettiğini test ediyor. |
| Konuşma kaydedersem ne olur? | `ml/heartbeat_check.py`: düşük frekanslı ses yüksekliğinde dakikada 40–220 atımlık düzenli bir ritim aranır. Ritim yoksa model hiç tahmin yapmaz, "tekrar kaydedin" der. |
| Basit modele göre CNN ne kazandırıyor? | Validation sekmesindeki karşılaştırma: CNN __CV_SCORE__, basit lojistik regresyon __BASE_SCORE__. |
| Yapay zeka asistanı kullandın mı? | Evet, kurallar buna izin veriyor. Önemli olan her parçayı anlayıp açıklayabilmen. Bu tabloyu çalış. |

## En zayıf noktalar ve dürüst cevapları

- **"Üfürümlerin bir kısmını kaçırıyor."** Evet. Bu yüzden bir tanı aracı değil, "kimi önce ekokardiyografiye gönderelim" sorusuna yardım eden bir tarama aracı. Kaçırdığı vakaların çoğu hafif (düşük dereceli) üfürümler olabilir. Bunu analiz etmek bir sonraki adım.
- **"0,78 ile kıyaslamak adil mi?"** Tam adil değil, yarışmanın gizli test verisine erişimimiz yok. Uygulamada da bunu açıkça yazdık, kıyas yalnızca fikir vermek için.
- **"Telefon mikrofonuyla çalışır mı?"** Henüz test edilmedi, sınırlamalarda yazıyor. Steteskop adaptörüyle test etmek planlanan ilk adım.

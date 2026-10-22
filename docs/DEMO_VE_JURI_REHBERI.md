# LittleHeart: demo videosu ve jüri hazırlık rehberi

Bu rehber senin için. Videoda ve jüriyle konuşurken kullanacağın cümleler İngilizce, açıklamalar Türkçe.

## Bağlantılar

- Depo: <https://github.com/Turan16-k/UnivaBio>
- Canlı demo: <https://turan16-k.github.io/UnivaBio/>
- Doğrudan bir vaka açmak için: <https://turan16-k.github.io/UnivaBio/?case=4>
- Her değişiklikten sonra `littleheart` klasöründe `git push` çalıştırman yeterli; GitHub Pages kendiliğinden yeniden yayınlanır.

## Demo videosu senaryosu (yaklaşık 3,5 dakika)

**0:00–0:25 · Sorun** (About murmurs sekmesi)
> "About one in a hundred babies is born with a heart defect, and rheumatic heart disease affects tens of millions of young people, mostly in low-income countries. Both often show up first as a heart murmur. But hearing a murmur takes years of training, and the echocardiogram that confirms it is scarce exactly where these diseases are most common. So I built LittleHeart."

**0:25–1:00 · Rehberli kayıt** (Screen a child → Record)
- **Start guided screening** düğmesine bas: "A health worker is guided through the four listening sites, 15 seconds each."
- Canlı dalga formunu ve göstergeyi göster: "LittleHeart checks the sound live. If there is no heartbeat, it says so before wasting the recording."
- Mikrofona konuş, sonra **Stop**: "If I just talk, the AI refuses to guess. That is the safety check."

**1:00–1:45 · Gerçek vakalar** (Real cases)
- **Case 4**'e tıkla: "These are real children from a Brazilian dataset that my model never saw in training. Murmur suspected, refer for echocardiogram, and here is the cardiologist's finding: it agreed."
- Bir kaydın ▶ düğmesine bas: "The spectrogram is a picture of the sound. The bars show how likely a murmur is in each 4-second window."
- **Case 5**'i aç: "A healthy heart: no murmur."
- Çocuk kimliği yaz, **Save to log**, sonra **Screening log** sekmesi: "On a school screening day, every child goes into this log, stored only on the device, and can be exported for the clinic."

**1:45–2:15 · Kulağını eğit** (Train your ear)
> "Health workers can also practise. Here are 18 real recordings. I guess, then I see the cardiologist's answer and the AI's, from a model that never heard that child."
- Bir kayda cevap ver ve sonucu göster.

**2:15–3:00 · Nasıl çalışıyor, ne kadar iyi?** (Validation)
- Skor kartları: "With 5-fold cross-validation split by child, it reaches 0.79 on the official PhysioNet 2022 murmur score, against 0.60 for a simple baseline. It catches 86% of children with murmurs and clears 91% of those without."
- "Which murmurs does it miss?" kartı: "Almost all missed murmurs are the faintest grade. It caught every loud one."
- "Real-world sound" tablosu: "Phones lose the deepest sounds. My first model caught only 36% of murmurs through a phone-like microphone. I retrained it with randomly damaged audio, and now it catches 83%."
- Dış test tablosu: "I also tested it on a completely different dataset of mostly adult recordings. It did much worse there, AUC 0.57. That tells me it must be validated locally before being used anywhere new, and I say that openly in the app."

**3:00–3:30 · Kapanış**
> "It works offline, installs like an app, speaks four languages, and never uploads a recording. It is a research prototype, not a diagnosis. Next I want to test it with real low-cost stethoscope adapters and a school screening programme."
- Dil seçiciden **TR**'yi seç ve sayfanın Türkçeye döndüğünü göster.

Kayıt için Windows'ta Win + Alt + R ya da OBS kullanabilirsin.

## Jüriye kodu anlatırken: soru ve cevaplar

| Muhtemel soru | Cevap ve göstereceğin yer |
|---|---|
| Veri nereden? | PhysioNet'teki CirCor DigiScope veri seti: 942 çocuk, 3.163 kayıt, kardiyologlar etiketledi. İndirme: `ml/download_data.py` |
| Ses nasıl sayıya dönüşüyor? | `ml/features.py` içindeki `log_mel`. Ses 64 ms'lik parçalara bölünür, her parçanın frekansları FFT ile hesaplanır, 32 mel bandında toplanır ve logaritması alınır. Ortaya bir "ses resmi" çıkar. |
| Model nedir? | `ml/train.py` içindeki `MurmurCNN`: 4 evrişim bloğu, ortalama havuzlama ve 3 çıkış (üfürüm yok / var / belirsiz). Yaklaşık 61.000 ağırlık. |
| "Gerçek dünya için eğitildi" ne demek? | `ml/evaluation.py` içindeki `augmented_window`: eğitim pencerelerinin %60'ı rastgele bozuluyor. Bazılarında 60–250 Hz altı kesiliyor (telefon mikrofonu gibi), bazılarına 3–30 dB arası gürültü ekleniyor. Böylece model temiz olmayan seslere alışıyor. |
| Bunun işe yaradığını nereden biliyorsun? | Aynı katlarla iki model eğittim: biri sadece temiz sesle, biri bozulmuş sesle. Test kayıtlarını da bozup ikisini karşılaştırdım (`evaluate_fold`). Telefon benzeri mikrofonda yakalanan üfürüm %36'dan %83'e çıktı. Eğitimde hiç görmediği tutma gürültüsünde ise biraz kötüleşti. Bunu da yazdım. |
| Çocuk bazında karar nasıl veriliyor? | `decide` fonksiyonu: herhangi bir kapakçıkta üfürüm eşiği aşılırsa sevk. Aşılmazsa ve bir kayıt belirsizse tekrar kayıt. Aksi halde üfürüm yok. |
| Ezber yapmadığını nereden biliyorsun? | Veriyi kayıt bazında değil çocuk bazında böldüm. 5 katlı çapraz doğrulama yaptım. Eşikleri ayrı bir doğrulama verisinde seçtim. Demo vakaları eğitime hiç girmedi. Ayrıca tamamen farklı bir veri setinde de test ettim (`ml/external_validation.py`). |
| Neden ağırlıklı doğruluk? | Yarışmanın resmi ölçütü bu. Üfürümü kaçırmak daha tehlikeli olduğu için üfürüm vakaları 5, belirsizler 3, normaller 1 kat sayılıyor. Kod: `weighted_accuracy` |
| Tarayıcıda nasıl çalışıyor? | Model ONNX formatına çevrildi ve ONNX Runtime Web ile çalışıyor (`web/js/model.js`). Spektrogram kodu JavaScript'e birebir çevrildi (`web/js/features.js`). Testler iki dilin aynı sayıları ürettiğini doğruluyor. |
| İnternetsiz nasıl çalışıyor? | `web/sw.js` bir service worker: ilk ziyarette uygulamayı, modeli ve kayıtları önbelleğe alıyor, sonra internet olmadan oradan sunuyor. |
| Canlı kalite kontrolü nasıl? | `web/js/guided.js` içindeki `checkLive`: her saniye son 6 saniyelik sesi 4 kHz'e indirip kalp atışı kontrolünden geçiriyor. |
| Konuşma kaydedersem ne olur? | `ml/heartbeat_check.py`: düşük frekanslı ses yüksekliğinde dakikada 40–220 atımlık düzenli bir ritim aranır. Ritim yoksa model tahmin yapmaz, "tekrar kaydedin" der. |
| Quiz'deki yapay zeka cevapları hile mi? | Hayır. Her cevap, o çocuğu eğitimde hiç görmemiş çapraz doğrulama modelinden geliyor (`pick_quiz`). |
| Yapay zeka asistanı kullandın mı? | Evet, kurallar buna izin veriyor. Önemli olan her parçayı anlayıp açıklayabilmen. Bu tabloyu çalış. |

## En zayıf noktalar ve dürüst cevapları

- **"Üfürümlerin bir kısmını kaçırıyor."** Evet, yaklaşık 7 üfürümden 1'ini kaçırıyor. Ama tamamen kaçırılan 24 üfürümün 22'si en hafif derece (I/VI). Yüksek sesli (III/VI) üfürümlerin hepsini yakaladı.
- **"Birinciyle aynı skoru mu aldın?"** Hayır, böyle bir iddiam yok. Benim 0,79'um herkese açık veride çapraz doğrulama sonucu. Birincinin 0,78'i daha zor bir gizli test setinde. Sadece doğru aralıkta olduğumu gösteriyor.
- **"Başka veride neden bu kadar kötü?"** PhysioNet 2016 verisi çoğunlukla yetişkinlerden, başka cihazlarla alındı. Etiketleri de "üfürüm" değil "normal/anormal kalp"; örneğin koroner arter hastalığı üfürüm yapmaz. Yine de sonuç, modelin yeni bir yerde kullanılmadan önce yerel olarak doğrulanması gerektiğini gösteriyor. Bunu uygulamada açıkça yazdım.
- **"Telefon mikrofonuyla çalışır mı?"** Telefon benzeri bozulmayı simüle ederek test ettim ve dayanıklı eğitimle büyük iyileşme gördüm. Ama gerçek bir telefon adaptörüyle henüz test etmedim. Bu, planlanan ilk adım.
- **"Demo vakalarında neden 2 hata var?"** Demo vakaları eğitimden önce rastgele seçildi, sonucu güzelleştirmek için değiştirmedim. 4 üfürümün hiçbirini kaçırmadı. Bir sağlıklı çocuktan tekrar kayıt istedi. Bir de uzmanın bile karar veremediği bir kayda "üfürüm yok" dedi.

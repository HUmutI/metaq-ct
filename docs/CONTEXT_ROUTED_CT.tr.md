<!--
  Context-Routed CT — mimari tasarım belgesi
  ARC-CT'nin pediatrik + yetişkin devamı · Kurugol Lab, BCH
  Bu dosya, aynı belgenin İngilizcesi olan CONTEXT_ROUTED_CT.en.md ile
  TEKNİK OLARAK ÖZDEŞTİR. Birinde bir sayı veya karar değişirse ötekinde de değişmeli.
  Kaynak artifact: claude.ai/code/artifact/7d8a60de-810f-4fa4-8868-524e30c7fffb
-->

`ARC-CT v2 · mimari tasarım belgesi · rev. 7`

# Context-Routed CT

Dr. Kurugol'un adlandırmasıyla: **Indication-Conditioned Anatomy-Routed
Attention**. Anatomi bir bulgunun *nerede* olabileceğini belirler;
klinik indication *hangi* bulguların daha dikkatle sorgulanacağını belirler.
Ve indication'dan bağımsız bir yol her zaman açık kalır — çünkü pnömoni için
çekilmiş bir BT'deki beklenmedik pnömotoraks kaçırılamaz.

Indication modele *ne olduğunu* söylemez. *Nereyi ve neyi* daha
dikkatle sorgulayacağını söyler — indication'dan bağımsız temsil ise BT'nin
bütüncül yorumunu korur.

Bu belge Dr. Kurugol'un 25 KB'lık notlarını mimarinin merkezine alıyor:
**çift yol** (koşullu ve koşulsuz patoloji temsili birlikte),
**indication token dizisine cross-attention**, baskılamayan **alaka kapısı**
w = 1 + βr, ve 27 sınıfın dışını yakalayan **4 serbest indication query'si**.
Üstüne, koşulsuz yolun gerçekten koşulsuz kalmasını sağlayan **dört izolasyon
düzeltmesi** (§04) — bunlar olmadan tasarımın manşet güvenlik iddiası sessizce
ihlal ediliyordu. Ölçülmüş sayılar `[ölçüldü]`, tasarım
kararları `[öneri]`, açık sorular
`[açık]`.

> **rev.7 · kapsam — Faz 1 neyi teslim ediyor**
>
> **Novelty C1 + C2'de.** Faz 1 yalnızca bunları teslim eder: indication'ın
> pathology query'lerine cross-attention + FiLM ile girmesi (C1), baskılamayan alaka
> ağırlıklandırması (C2), koşulsuz yolun izolasyonu, ve boş-bölge uçurumunun
> *hata düzeltmesi* olarak kapatılması.
>
> **C3 (öğrenilen λ kapısı) ve C4 (serbest indication query'leri) Faz 1'de yok.**
> İkisi de merdivenin sonunda opsiyonel kol. Gerekçe aşağıda (§07, §08); kısaca:
> C3'ün motivasyonu §03 Gerçek 2 ile karışmış bir ölçüme dayanıyor *ve* ρ
> üzerinden çıkarımda maske gerektirerek ARC-CT'nin maskesiz-çıkarım özelliğini
> kırma riski taşıyor; C4 ise tasarımdaki en büyük kısayol yüzeyi.
>
> Bunun bedeli yok, kazancı var: merdiven 12 → **8 basamak**, makro karşılaştırma
> 36 → **24**, tohum bütçesi ~675 → **~450 GPU-saat**, ve query bankası
> 71 → **67 slot**. **Ve paper yine tam:** C1+C2 kendi çıtalarını geçerse,
> pediatrik kohort zaten tek başına bir katkı — halka açık pediatrik 3B BT VLM
> benchmark'ı yok.
>
> **Faz ayrımı bir bütçe kısıtı değil, atıf hijyeni.** GPU bu projede kısıt değil ve
> çok-GPU paralel eğitim açık; yukarıdaki GPU-saat rakamı bu kararın *sonucu*, sebebi
> değil. Sebep şu: C3 ve C4 aynı anda açıkken manşet iddia — "C1+C2 sınıf sıralamasını
> değiştirir, `Z_gen`'i bozmadan" — artık ayrıştırılamaz; kazanç dört kaynağa dağılır ve
> hakem hangisinin çalıştığını soramaz. Faz 1 önce koşar ki katkı temiz atfedilebilsin.
>
> **Faz 2 bunu takip eder, uzun süre beklemez.** İki senaryo da geçerli. Faz 1 beklendiği
> gibi gelirse C3/C4 doğrudan üstüne, Faz 1 checkpoint'inden sıcak başlatılarak koşulur.
> Faz 1 çıtayı ıskalarsa Faz 2 bir **kurtarma kolu** olur — özellikle C3, sert maskenin
> pediatrikte küçük ve ızgaraya düşmeyen bölgelerde yaptığı hasarı geri alabilecek tek
> kol (§07 boş-bölge uçurumu). Bedeli var: ρ çıkarımda maske gerektirdiği için
> maskesiz-çıkarım özelliği kaybedilir, o takas ayrıca raporlanır. Tek katı koşul,
> Faz 1'in *ölçümünün* Faz 2 karışmadan kapanmış olması.

`01 · başlangıç noktası`

## Nerede duruyoruz

Yeni mimarinin üstüne kurulacağı zemin. Tek değerlendirici, aynı hacimler.
C47 satırları mevcut raporların hiçbirinde yok — 25 Ağustos'ta bitti.

| koşu | tek değişen | peds 27 | peds routed | CT-RATE 27 | adult-18 | yetişkin kaybı |
|---|---|---|---|---|---|---|
| V1 | taban (eski etiket) | 0.8005 | 0.7651 | 0.7954 | 0.8151 | −0.0423 |
| V1-seed1 | sadece tohum · **farklı etiket setinde**, §14 | 0.7933* | — | — | — | — |
| V2 | yeniden çıkarılan etiket | 0.7974 | 0.7657 | 0.8000 | 0.8187 | −0.0387 |
| V3 | + boş-bölge hükmü | 0.7972 | — | 0.8003 | 0.8189 | −0.0385 |
| V4 | + top-K odaksal havuzlama | 0.7971 | — | 0.8005 | 0.8187 | −0.0387 |
| V5 | + çok ölçekli ızgara 24³ | **0.8019** | 0.7662 | 0.7925 | 0.8086 | −0.0488 |
| C16 | ortak, 14.7k yetişkin, *uydurma etiket* | 0.7881 | 0.7596 | 0.7991 | 0.8378 | −0.0196 |
| C47mix | ortak, 42.5k yetişkin, gerçek etiket | 0.7964 | 0.7608 | 0.8228 | 0.8400 | −0.0174 |
| C47harm | + etiket harmonizasyonu | 0.7886 | 0.7587 | 0.8271 | **0.8442** | **−0.0132** |
| ARC-CT | yayınlanmış yetişkin model — **geçilmesi gereken çizgi** | — | — | — | 0.8574 | — |

*Skorlanan peds val **n=1.772** (split 1.774 diyor; 2 hacmin etiket satırı yok) · CT-RATE val n=3.002. ***** V1-seed1 eski etiket setinde ölçüldü (0.7961 ile eşleşir, 0.8005 ile değil) — bandın 0.0028 olması bu yüzden; tablodan 0.0072 okunmamalı. **Tohum bandının nasıl kullanılacağı §14'te baştan yazıldı: karar eşiği olarak kullanılamaz.***

> **çıta durumu**
>
> **Pediatrik çıta: ölçeğe bağlı, ve ilan ettiğimiz ölçekte henüz geçilmedi.**
> Notlardaki beklenti *"pediatric AUC tavanı çok önemli değil… ama inş +80"*.
> V5, 27-sınıf ölçekte **0.8019**; ama §11'de manşet metrik olarak
> **ölçülebilir 23**'ü ilan ettik ve orada V5 = **0.7957**. Yani çıta 27-sınıf
> ölçekte geçilmiş, dürüst ölçekte geçilmemiş görünüyor. Asıl zor olan iş yine de
> **yetişkin çıtası**: bugün 0.8442, fark 0.0132 ± 0.0022 (§13).

`02 · yönerge`

## Yönerge → karar eşlemesi

Notlardaki **başlıca direktifler** ve onlara karşılık gelen kararlar.
**İki yerde açıkça ayrıldım** (`[AYRILDIM]` ile işaretli);
üç yerde direktifi kabul edip *uygulamasını* düzelttim (r_c'nin MLP biçimi,
global query'nin bölme değil ekleme olması, kırpmanın gövdeye yapılması).
Tablo direktiflerin tamamını kapsamıyor; kapanış maddeleri gövdede (§01, §09, §13, §14).

> **notlardan · çerçeveleyen cümle**
>
> > *The indication does not tell the model what is present. It tells the model *where* and *what* to interrogate more carefully, while an indication-independent representation preserves comprehensive interpretation of the CT.*
> > *Anatomy determines *where* a finding can occur, while indication determines *which* findings and visual patterns deserve increased attention.*
>
> Bu iki cümle tüm tasarımın kısıtı. Her mekanizma bunlara karşı test edilecek:
> indication bir bulguyu *bastırıyorsa* mekanizma yanlıştır.

| yönerge | karar | nerede |
|---|---|---|
| Indication'ı QFormer pathology query embedding'lerinde kullan | Ana katkı. **Cross-attention** (query → indication token dizisi) **+ FiLM** kanal kapılaması. Havuzlanmış tek vektör değil, token dizisi. | §05 |
| Koşulsuz BT yolunu koru; `Z = Z_gen + g⊙Z_ind`, öğrenilen kapı | **Çift patoloji bankası**: 27 koşulsuz + 27 koşullu, ağırlıkları bağlı. Füzyon **concat** (onun tercihi), kapılı toplam ablasyonda. | §04, §05 |
| Alaka kapısı `w=1+βr` — **asla** `r` ile çarpma | Aynen. Düşük alakalı sınıf ağırlığı 1'de kalır, 0'a inmez. Insidental bulgu koruması mimariye gömülü. | §06 |
| Anatomi query'lerini koşullandırma | Aynen — kavramsal ayrım temiz kalıyor. (Rev.1'de de böyleydi.) | §04 |
| İki global query'yi böl: genel / klinik-soru | Kabul, ama **bölme değil ekleme**: miras iki global `Z_gen`'de kalır, klinik-soru query'si **40. ayrı query** olarak eklenir — yoksa `Z_gen` 38 token üzerinden ortalanır ve ARC-CT ile özdeş olmaz (§04). | §04 |
| 4 serbest indication query'si ekle (27 sınıfın dışı için) | **Rev.7'de Faz 2'ye alındı.** Kohortumuz tam olarak o vakalarla dolu (osteosarkom takibi, transplant sonrası, timoma) — ama C4 tasarımdaki en büyük kısayol yüzeyi ve "indication yoksa katkısı sıfır" iddiası mekanize değil (§08). | §08 |
| Yaş bandı gömmesi + cinsiyet | Ordinal-kümülatif bant gömmesi; yaş/cinsiyet **bağlam token'ı** olarak indication token dizisine katılıyor — ayrı bir yol değil. | §09 |
| Indication dropout %20–30, uyumsuz-indication testi | Dropout **%30** + karşıolgusal tutarlılık kaybı — rev.5'te **geri getirildi**, ama `Z_gen`'e değil **tahmin logitlerine** uygulanıyor (§12). | §12, §14 |
| Mevcut kayıpları koru, `L_ind = Σ(1+βr)·BCE` ekle | Kabul — o BCE zaten var (`loss_pertoken`), ama **"tek satır" değil**: çift bankayla dilim P=54 olur ve broadcast hata verir; hangi bankayı denetlediği de söylenmeli (§12). | §12 |
| `r_ic`'yi elle kodlama, gömmelerden öğren | Aynen öğrenilen — ama notlardaki **iki seçenekten MLP olanı**: nokta-çarpımı biçimi donuk kulelerle eğitilemez (§06). | §06 |
| Concat, cross-attention'ı yendi (mütevazı veri setinde) | Ablasyon merdiveni zorunlu hale getirildi. Bizim indication'ımız 10 kat uzun, ama peds **eğitim** seti sadece 7.042 hacim (8.817'lik kohortun eğitim bölümü) — yani uyarı bize **birebir** uyuyor. | §14 |
| Her katmanı koşullandırma; son bloklarda tut | Bizde zaten öyle: ResNet hiç dokunulmuyor, koşullandırma yalnızca Q-Former query'lerinde — yani 4 bloklu köprüde. | §04 |
| Serbest metin kullan, kategori değil | Aynen. Ayrıca yapılandırılmış kavram çıkarımı (semptom / bağlam / hedef durum / anatomi) Faz 2 opsiyonu olarak duruyor. | §05 |
| Zayıf lokalizasyon kaybı (`L_localization`) | Artefakt hazır (segmentasyon gerekmiyor) ama **bedava değil**, ve rev.4 yanlış bağlamıştı: cache *rapor bulgularını* eşliyor, indication'ı değil → yalnızca koşulsuz bankaya uygulanır (§12). | §12 |
| **Peds'te maskesiz gidelim / anatomi query'lerini çıkaralım** | **AYRILDIM.** Kararı elle vermek yerine **öğrenilen kapıya** bırakıyorum: maske hep verilir, sertliği λ öğrenilir. λ→0 çıkarsa bu *bulgu* olur. Elle-kapalı sürüm ablasyon kolu olarak duruyor. Gerekçe: §03 Gerçek 2. | §07 |
| **Pnömotoraksı da çıkarabiliriz** | **AYRILDIM.** BT pnömotoraks için referans standart; düşük skorun sebebi modalite değil, bizim türettiğimiz bölge-8 kabuğu. Kalsın ve C3'ün ilan edilmiş test vakası olsun. | §11 |
| Adult-only sınıfları peds skorlamasından çıkart; arterial wall calc? | Dördü de (Emphysema 3 · Coronary 5 · Hiatal hernia 18 · **Arterial wall calc 34**) peds'te kayıp maskeli ve peds skorunda yok → manşet **ölçülebilir 23**. Modelden değil skordan çıkar. Tanım düzeltmesinin 0.654→0.735 kazancı *etiket kalitesi* bulgusu olarak ayrıca raporlanır. | §11 |
| Pediatride doğrudan downsample yerine **kırpma** uygula — bebeklerde akciğerler çok küçük | Kabul. §10 buna göre yeniden yazıldı: gövde kutusuna kırp → 192³'e örnekle. İki teknik ek: **akciğere değil gövdeye** kırpılmalı (yoksa bölge 9 ve 10 yok olur), ve voxel boyutu `H_dem`'e token olarak geri verilmeli (yoksa mutlak ölçek kaybolur). | §10 |
| Çözünürlük??? | **Ölçüldü ve tahmin çürüdü.** Baskın etki dolgu değil kırpma (hacimlerin %72'sinde veri atılıyor, medyan dolgu %5.2). Çevresel-sınıf tahmini sıfırdan ayırt edilemedi (kontrast +0.0154, GA [−0.076, +0.087]). | §10 |

> **rev.1'den düşen**
>
> Önceki sürümdeki **M3 — yaş-koşullu prevalans önselı** (logit düzeltmesi)
> Dr. Kurugol'un tasarımında yok, ve onun `r_ic` alaka mekanizması aynı işin
> savunulabilir kısmını zaten yapıyor. **Opsiyonel ve en düşük öncelikli**
> seviyesine indirildi; Faz 1'de yok.

`03 · kod okuması`

## Kodda bulduğum üç yapısal gerçek

Bunlar ölçüm değil, kaynak kodu okuması. Üçü de mevcut raporlarda yazmıyor ve
üçü de mimariyi doğrudan belirliyor.

> **Gerçek 1 · `train_stage2.py:663–674`**
>
> **Her pathology query zaten sınıf başına bir classifier head — eğitiliyor ama
> test zamanında hiç okunmuyor.**
>
> ```
> path_tokens = qf_tokens[:, A : A+P]        # 27 pathology query token
> path_lat    = normalize(path_tokens)
> sims_pos    = (path_lat * pos_embs).sum(-1)   # her query kendi "X var" prompt'una
> sims_neg    = (path_lat * neg_embs).sum(-1)   # her query kendi "No X" prompt'una
> loss_ptok   = CE(softmax([pos,neg]/τ), label) # ağırlık 0.5, iki config'de de AÇIK
> ```
>
> Yani `query[10+p]` literal olarak sınıf `p`'nin dedektörü ve
> metin latent uzayında yaşıyor. Ama `evaluate.py` `qf_tokens`'ı
> hiç kullanmıyor: sadece (a) 39 query'nin *ortalaması*, (b) maske-havuzlanmış
> organ latent'i. **Eğitilmiş, kalibre olmuş, sınıf-özel bir okuma başlığını
> çöpe atıyoruz.**
>
> Tasarım sonucu: pathology query'ler indication'ı enjekte etmek için *doğru*
> yer — sınıf başına ayrışmışlar ve zaten paylaşılan görüntü-metin uzayında.
> Ve o okuma başlığını geri açmak **sıfır ek eğitim** maliyetli.

> **Gerçek 2 · `evaluate.py:294–312`**
>
> **"Routing penalty" Q-Former'ın hard mask'ini ölçmüyor. Tamamen başka bir
> okuma başlığını ölçüyor — ve o başlık hiç sınıf prompt'una eğitilmedi.**
>
> `_routed_probs`, global latent'i `organ_lat` ile değiştiriyor.
> `organ_lat` ise eğitimde *yalnızca*
> `soft_clip_infonce(organ_lat, region_lat)` ile, yani *bölge cümlelerine*
> hizalanıyor. Değerlendirmede ise *sınıf prompt'larıyla* ("Pneumothorax." /
> "No Pneumothorax.") skorlanıyor — hiç görmediği bir hedef bankasıyla.
>
> Bu bir train/test amaç uyuşmazlığı ve gözlenen deseni *tam olarak* öngörüyor:
> en büyük ceza bölge metninin en seyrek ve en heterojen olduğu yerlerde —
> Pneumothorax (−0.3369, bölge 8) ve Bone lesion (−0.2222, bölge 9).
>
> **Sonuç: "hard masking zarar veriyor" iddiası kanıtlanmış değil.**
> Üç hipotez de canlı ve üçü de ayrıştırılabilir:
>
> - **H1 · maske kalitesi** — türettiğimiz plevral kabuk çok ince. *Test:* bölge 8'i kalınlaştır, cezayı yeniden ölç.
> - **H2 · checkpoint seçim yanlılığı** — **rev.6 düzeltmesi:** `train_stage2.py`'nin docstring'i "mask-free" diyor ama `run_validation` maskeleri Q-Former'a *veriyor*; yani seçim zaten oracle-maskeli. Seçimden dışlanan tek şey `_routed_probs` organ-havuz okuması. Bu, H2'yi zayıflatır ve Gerçek 2'yi güçlendirir; test yine de ara checkpoint'lerde yapılır.
> - **H3 · okuma amacı uyuşmazlığı** — yukarıdaki. *Test:* `organ_lat`'a sınıf prompt kaybı ekle veya bu metriği "routing metriği" diye raporlamayı bırak.
>
> Teklif dokümanındaki *"kum üstüne inşa ediyoruz"* uyarısı bu yüzden erken.
> Üç ölçüm de bir haftadan kısa ve ikisi için gereken her şey diskte hazır.

> **Gerçek 3 · `anatomy_qformer.py:93`**
>
> **Soft routing zaten var — kazara.**
>
> ```
> any_key = out.any(dim=-1, keepdim=True)
> out = torch.where(any_key, out, torch.ones_like(out))   # boş bölge → tamamen serbest
> ```
>
> Bir bölge feature grid'e hiç ulaşamazsa query sessizce *kısıtsız global query*
> oluyor. Güncel 10-bölge maskelerinde ölçülen kaçırma: sağ orta lob **%5.6**,
> central airway **%1.5** (rev.4'teki %10/%3 terk edilmiş 13-bölge şemasının
> rakamıydı). Yetişkinde bile CT-RATE validation'da airway **%4.1**, kalp **%4.5**
> — yani kalp airway'den biraz daha kötü.
>
> Yani sistem zaten "maske güvenilmezse global'e düş" davranışı sergiliyor —
> sadece bunu *koşulsuz* ve *öğrenilmemiş* biçimde yapıyor. Bizim
> katkımız yeni bir mekanizma icat etmek değil: **ilkesiz bir fallback'i
> öğrenilebilir bir kapıya çevirmek.** Bu, savunması çok daha kolay bir iddia.

> **[ŞEKİL 1]** **Şekil 1 — Üç okuma başlığı, biri hiç okunmuyor.** Aynı kuleden üç yol çıkıyor. R1 birincil metrik ve yetişkin kapısını o koruyor. R2 eğitiliyor ama değerlendirmede hiç kullanılmıyor — bedava kazanç adayı ve C1'in etkisinin görüneceği yer. R3'ün eğitim hedefi (bölge cümleleri) ile test hedefi (sınıf prompt'ları) farklı; "routing penalty" işte bu kesikli okta ölçülüyor.

> **bu üç gerçek yönergeye nasıl bağlanıyor**
>
> - **Gerçek 1 → `L_ind` bir satır.** Notlarda önerilen indication kaybı `L_ind = Σ(1+βr_ic)·BCE(ŷ_ic, y_ic)`, "zaten her pathology query'yi pozitif ve negatif sınıf prompt'larına karşı denetlediğin için tamamen yeni bir amaç getirmiyorsun" gerekçesiyle öneriliyor. **Bu tam olarak doğru** — o BCE bizde `loss_pertoken` adıyla mevcut ve açık. `L_ind`, o kaybın ağırlıklandırılmasından ibaret.
> - **Gerçek 1 → koşullu okuma başlığı hazır.** Notlardaki "conditioned head" bizde R2'dir ve zaten eğitiliyor; sadece `evaluate.py` onu okumuyor.
> - **Gerçek 2 → "maskesiz peds" kararını beklet.** Maskeleri terk etmenin ana ampirik gerekçesi olan −0.03'lük routing cezası, hard mask'i *ölçmüyor*. ARC-CT'nin yayınlanmış çekirdek katkısını, karışmış bir ölçüme dayanarak atmak çok pahalı bir hata olur. Üç saatlik ölçüm bunu ayrıştırıyor (§17, Faz 0).
> - **Gerçek 3 → kapı zaten yarı yolda.** "Boş bölge → kısıtsız query" davranışı, maskesiz peds'in *kaza eseri ve koşulsuz* hâli. Onu öğrenilebilir yapmak, "maskeli mi maskesiz mi" sorusunu bir tasarım kararından bir *ölçüme* çevirir.

`04 · sistem`

## Mimari: bütün resim

Miras alınan ARC-CT hattı değişmiyor. Bağlam hattı yeni. Kritik yapısal özellik:
patoloji temsili **iki kopya** hâlinde üretiliyor — biri indication'ı hiç görmeyen,
biri gören — ve ikisi sonda birleşiyor. Beklenmedik bulguyu koruyan şey bu.

### Query bankası: Faz 1 → 40 ayrı query, 67 slot · Faz 2 → 44 / 71

| grup | adet | koşullu mu | dikkat kısıtı | rol |
|---|---|---|---|---|
| Anatomi | 10 | hayır | tek organa maskeli | organ temsili — dokunulmuyor |
| Patoloji · genel | 27 | hayır | organ birleşimine maskeli | koşulsuz yol · **güvenlik ağı** |
| Patoloji · koşullu | 27 | **evet** | aynı maske, aynı ağırlık | indication yolu · ana katkı |
| Indication query **(C4 · FAZ 2)** | 4 | **evet** | kısıtsız | 27 sınıfın dışını yakalar — Faz 1'de **yok** |
| Global · genel (miras, 2 satır) | 2 | hayır | kısıtsız | bütüncül BT özeti |
| Global · klinik soru **(YENİ)** | 1 | **evet** | kısıtsız | 40. ayrı query — miras global'lerin yerine geçmez, üstüne eklenir |

***Faz 1: 67 dikkat slotu · 40 ayrı query vektörü · 1 yeni query** (klinik-soru global). C4 eklenirse 71 / 44 / 5. Koşullu patoloji bankası genel bankayla **ağırlık paylaşır** (aynı `q_c`, biri koşullandırmadan geçer biri geçmez), o yüzden 27 ek slot yeni parametre getirmiyor. Maliyet (Faz 1): **ARC-CT'ye göre 1.72×** (67/39); çift bankanın marjinal payı 67/40 = 1.68×; query'ler arası self-attention O(Q²) ama bu ölçekte ihmal edilebilir. **Ama bu iki bankanın ayrı kalması kendiliğinden olmuyor** — aşağıdaki bölüm.*

### Koşulsuz yolun izolasyonu — dört sızıntı yolu

Çift banka tek başına hiçbir şey garanti etmiyor. Q-Former'ın içinde koşullu ve
koşulsuz query'ler aynı ileri geçişte, ve bilgi **dört ayrı yoldan** koşulsuz tarafa
sızabilir — dördü de sessizce, hiçbir hata vermeden. Bu, kusur kataloğundaki hatalarla
aynı sınıftan: ölçüm çalışır, sayı çıkar, sayı yanlıştır.

> **sızıntı 1 · self-attention**
>
> `QFormerBlock` her blokta tüm query'ler arasında self-attention
> çalıştırıyor (`self.self_attn(h, h, h)`, maske yok). Yani koşulsuz
> patoloji query'leri, koşullu olanları *görür* ve indication bilgisini dolaylı
> olarak alır. Bir blok sonra "koşulsuz" banka artık koşulsuz değildir.
>
> **Düzeltme:** self-attention'a grup maskesi. Kural tek yönlü —
> **koşulsuz grup, koşullu hiçbir query'ye attend edemez**; koşullu grup her şeye
> edebilir.
>
> ```
> koşulsuz grup (39) = anatomi(10) + patoloji-genel(27) + global-genel(2)
> koşullu   grup (28) = patoloji-koşullu(27) + global-klinik(1)      # Faz 2'de +4 indication query
>
> self_attn_mask[i, j] = MASKELİ   eğer i ∈ koşulsuz  ve  j ∈ koşullu
> ```

> **sızıntı 2 · λ kapısı — asıl tehlikeli olan**
>
> C3'ün kapısı `λ_q = softplus(a_q + w_q·[H̄_C ; ρ])` şeklinde yazılmıştı.
> Ama λ, query'nin görüntüye *nereye bakabileceğini* belirliyor. Koşulsuz
> query'lerin λ'sı indication'a bağlıysa, koşulsuz yolun **dikkat geometrisi klinik
> soru tarafından yönlendiriliyor** demektir — self-attention'dan daha derin bir
> sızıntı, çünkü bu doğrudan hangi voxellerin görüldüğünü değiştirir.
>
> **Düzeltme:** λ ikiye ayrılır.
>
> ```
> λ_q^gen = softplus( a_q + w_q·[ ρ ; H̄_dem ] )          # maske geometrisi + yaş/cinsiyet
> λ_q^ind = softplus( a_q + w_q·[ ρ ; H̄_dem ; H̄_ind ] )   # + klinik soru
> ```

> **sızıntı 3 · füzyon başlangıcı**
>
> `Z_final = W[Z_gen ; Z_ind]` rastgele başlatılırsa adım 0'da model
> ARC-CT değildir — `Z_ind` yarısı gürültü olarak karışır ve
> "yayınlanmış ağırlıklardan başlıyoruz" iddiası düşer.
>
> **Düzeltme:** `W` başlangıçta **[ I ; 0 ]**. Sıfır-init disiplinini
> tamamlayan son parça: koşullandırma kapıları 0, β = 0, füzyon `Z_ind`
> yarısını görmüyor ⇒ **adım 0'da Z_final = Z_gen = ARC-CT, bit düzeyinde.**

> **sızıntı 4 · havuzlama — rev.5'te eklendi, ve en tehlikelisi**
>
> Rev.4'ün üç yollu listesi **eksikti**. `QFormer.forward`'un sonunda:
>
> ```
> if self.pool == "mean":
>     pooled = q.mean(dim=1)     # query ekseni üzerinde ortalama = token'lar arası KARIŞTIRMA
>
> # RAC_QFORMER_POOL=mean, her iki config'de de açık
> ```
>
> Ve bu `pooled` tam olarak **R1 / Z_gen** — belgenin kendi Şekil 1'inde
> *"BİRİNCİL METRİK — yetişkin kapısı"* diye etiketlenen şey — **ve aynı zamanda
> checkpoint seçim metriği**. 67 slotlu bankada `q.mean(dim=1)` koşullu
> 32 token'ı da ortalamaya katar, yani:
>
> - `Z_gen` sessizce indication-koşullu olur — diğer üç düzeltme tam uygulanmış olsa bile;
> - checkpoint seçim metriği indication-koşullu olur;
> - §15'in manşet deneyi kendi tahminini yanlışlar ve doğal (yanlış) okuma "üç düzeltmeden biri bozuk" olur.
>
> **Düzeltme:** havuzlama grup-kısıtlı hâle gelir —
> `Z_gen = mean(q[:, gen_idx])`, `Z_ind` ise koşullu indeksler
> üzerinde alaka-ağırlıklı normalize havuz. `RAC_QFORMER_POOL=mean` her iki
> env dosyasından kalkar. Bu, "FFN ve LayerNorm token-başına çalışır"dan
> *türetilemez*; ayrı bir madde.

> **bit-özdeşlik iddiası — düzeltilmiş hâli**
>
> Rev.4 *"adım 0'da model bit düzeyinde ARC-CT"* diyordu. **Değil**, dört
> bağımsız sebeple: **(i)** yayınlanmış peds modeli 39 query'nin *tamamının*
> ortalamasını alıyor; `Z_gen` 38 token üzerinden ortalama alacak (ikinci
> global query koşullu gruba geçti), yani her token'ın ağırlığı 1/39 → 1/38 kayıyor;
> **(ii)** self-attn grup maskesi koşulsuz satırların softmax paydasını değiştiriyor
> (39 key → 38); **(iii)** `−∞` yerine `−8` (§07'deki sızıntı tablosu); **(iv)** global grad-clip 0.5 — yeni modüllerin sıfır *çıktılı*
> ama sıfır *olmayan* gradyanları toplam normu şişirdiği için orijinal
> parametreler daha küçük adım alır.
>
> **Düzeltme:** her iki miras global query de `Z_gen`'de kalır ve klinik-soru
> query'si **40. ayrı query** olarak eklenir (böylece `Z_gen` yine 39
> token üzerinden ortalanır, ARC-CT ile aynı); iddia bir *assertion*'a
> dönüşür: 39-query ile 67-slot modelleri aynı batch'te koşturup
> `‖Z_final − img_lat_ARC‖ < 1e-5` doğrulanmadan hiçbir eğitim başlamaz.
> Geçmezse iddia yazıdan çıkar.

> **temiz olduğu kanıtlanabilen yollar**
>
> Geri kalan yollar yapısal olarak güvenli ve bunu söylemek önemli:
> **görüntü çapraz-dikkati** (her iki grup da aynı koşulsuz `F`'ye bakar),
> **FFN**, **LayerNorm** (`norm_out` dahil — hepsi son eksende,
> token-başına), **rezidüel bağlantılar** (indeks hizalı toplama) ve **dropout**
> (0.0, ve zaten token karıştırmaz). Q-Former içinde batch-bazlı hiçbir istatistik yok.
> Yani **dört** düzeltmeden sonra `Z_gen`'in indication'dan bağımsızlığı
> *ileri geçiş için* yapısaldır.
>
> **Ama parametreler için değil.** Koşullu ve koşulsuz bankalar tüm blok
> ağırlıklarını paylaşıyor, yani koşullu daldan gelen gradyanlar koşulsuz dalın
> kullandığı operatörleri de günceller. Sabit ağırlıkta `Z_gen(I) = Z_gen(I')`
> doğru (§15 bu yüzden güvenli), ama "koşulsuz yol *ARC-CT'dir*" demek doğru
> değil: parametreleri adım 1'den itibaren indication tarafından şekillendirilmiş olur.
> Bu, gizlenecek değil **söylenecek** bir şey.

> **terim netleştirmesi**
>
> "Koşulsuz" burada **indication'dan bağımsız** demek, *bağlamdan bağımsız*
> değil. Yaş ve cinsiyet her iki yola da kasten giriyor: bunlar klinik soru değil,
> hastanın özellikleri. Yaşın `λ^gen`'e girmesi C3'ün tüm gerekçesi
> (yaş = uzamsal ölçek) ve §15'teki karşıolgusal deneyi hiç etkilemiyor — o deneyde
> hasta sabit, yalnızca indication değişiyor.

> **[ŞEKİL 2]** **Şekil 2 — Indication-Conditioned Anatomy-Routed Attention.** Görüntü hattı (gri) hiç değişmiyor: ResNet koşullandırılmıyor, koşullandırma yalnızca Q-Former köprüsünde. Bağlam hattı (amber) indication'ı *token dizisi* olarak taşıyor; yaş ve cinsiyet gömmeleri ayrı bir yol değil, aynı dizinin ek token'ları — bu sayede yaş da uzamsal dikkati etkileyebiliyor. Sağda iki havuz: `Z_gen` indication'ı hiç görmemiş, `Z_ind` görmüş; concat ile birleşiyorlar. Alttaki kutu, bir bulgunun indication yüzünden kaybolmasını engelleyen dört bağımsız katmanı sıralıyor.

`05 · ana katkı`

## C1 — Indication → pathology query

### C1 — Cross-attention + FiLM, çift bankalı

**SEZGI**

Q-Former'ın patoloji query'leri "bu hacimde X var mı?" diye soran 27 ayrı sorgudur.
Radyoloğa *"kistik fibroz takibi"* dediğinde farklı bir soru setiyle bakar:
mukus tıkacı, bronşektazi, tree-in-bud öne çıkar. Biz tam olarak bunu yapıyoruz —
indication metni her patoloji query'sinin *vektörünü* yeniden şekillendiriyor.

Kritik nokta: bu sınıflandırıcıya ipucu vermek değil. Query vektörü
*attention'ın sorusudur*; onu değiştirmek modelin görüntüden *ne
çektiğini* değiştirir, çekilenin nasıl skorlandığını değil.

**BIÇIMSEL TANIM**

```
H_ind = CXR-BERT(indication)                # token DİZİSİ — havuzlanmıyor
H_dem = [ e_yaş(b) ; e_cins ; e_int ]       # demografik token'lar
H_C   = [ H_ind ; H_dem ]                   # koşullu query'ler bunu görür

# 1 · uzamsal soru: query indication token'larına bakar
q̃_c = q_c + g_x · CrossAttn( q_c , H_C , H_C )       # g_x sıfır-init skaler kapı

# 2 · kanal sorusu: hangi öznitelik türü önemli
q̃_c = γ(H̄_C) ⊙ q̃_c + β(H̄_C)     γ = 1 + ε·tanh(W_γ·), ε = 0.2, W sıfır-init
# ε SINIRI ŞART: γ = 1+tanh(·) ∈ (0,2) idi, yani tanh→−1'de γ→0 ve query
# tamamen siliniyordu — "rezidüel, q asla silinmez" güvenlik katmanı yalandı.

# 3 · görüntüye sorgu — anatomik kısıt DURUYOR, ama C3 ile yumuşuyor (§07)
z_c = CrossAttn( q̃_c , F , F ;  rol maskesi + λ kapısı )

# koşulsuz kopya: aynı q_c, koşullandırma yok, λ^gen kapısı
z_c^gen = CrossAttn( q_c , F , F ;  aynı maske + λ^gen )

# 4 · füzyon — W başlangıçta [ I ; 0 ]
Z_final = W [ Z_gen ; Z_ind ]                        # adım 0'da = Z_gen = ARC-CT
```

**REV.1'DEN NE DEĞIŞTI VE NEDEN**

Önceki sürümde yalnızca FiLM vardı ve gerekçem *"indication tek vektöre çöker"*
idi. **Bu bizim korpusumuz için yanlış.** Medyan 21 kelimelik bir indication —
*"History of metastatic osteosarcoma, evaluate pulmonary nodules"* ile
*"Incidental 5 mm nodule follow up"* arasındaki fark — havuzlanmış bir
vektörde kaybolur. Token dizisine cross-attention bunu korur.

Ama **karşı kanıt da var ve ciddiye alınmalı**: notlarda anılan çalışmada
3B göğüs BT'de *basit concatenation cross-attention'ı yenmiş*, yazarlar
bunu veri yetersizliğine bağlamış. Kohortumuz 8.816 hacim, ama cross-attention
parametreleri yalnızca **eğitim** bölümüne fit ediliyor: **7.042 hacim**
(1.774 validation). Yani tam olarak "mütevazı veri" bölgesindeyiz.
Ortak eğitimdeki 42.544 yetişkin hacim bunu telafi etmiyor — CT-RATE'te indication
yalnızca %49.8 mevcut ve medyanı 2 kelime, yani zengin bir indication
koşullandırması öğretebilecek efektif korpus pratikte o 7.042 hacim.
Bu yüzden §14'teki merdiven opsiyonel değil, zorunlu.

**NEDEN PATHOLOGY QUERY, BAŞKA BIR YER DEĞIL**

- **ResNet değil** — genel BT temsili indication'dan bağımsız kalmalı; notlarda da "her katmanı koşullandırma" var. Bizde ResNet'e hiç dokunulmuyor.
- **Anatomi query'leri değil** — anatomi klinik sorudan bağımsızdır. Indication'ın anatomi query'lerini şekillendirmesi, modelin görüntü yerine metinden anatomi uydurmasına kapı açar.
- **Patoloji query'leri evet** — §03 Gerçek 1: bunlar zaten sınıf başına ayrışmış ve metin latent uzayında yaşıyorlar. Modülasyon burada hem yorumlanabilir (‖Δq_c‖ = "bu indication bu sınıfı ne kadar açtı") hem de mevcut kayıp fonksiyonuna doğrudan bağlanıyor.
- **Ve gözlemlenmiş bir gerekçesi var.** §13'teki kırılımda yetişkin kaybının *ölçülebilir* kısmını taşıyan üç sınıf — Mosaic attenuation, Pulmonary fibrotic sequela, Lung nodule — anlamı *yaşa bağlı* olan üç sınıf. Üçü de ARC-CT ağırlığıyla bit-özdeş başladı ve yine de aşındı: ortak eğitim, anlamı kohortlar arasında ayrışan sınıflarda yetişkin becerisini bozuyor. C1'in yaş-koşullu modülasyonu bunun doğal adresi. **Nedensel değil, ilişkisel** — aynı üç sınıf etiketleyici-eşiği uyuşmazlığı listesinde de (§13).

`06 · güvenlik mekanizması`

## C2 — Alaka kapısı

### C2 — w = 1 + β·r · baskılamayan ağırlıklandırma

**YÖNERGENIN EN KESKIN DETAYI**

Notlarda çok net: **query'yi `r_c` ile çarpma**, çünkü bu insidental
bulguları baskılar. Onun yerine `w_c = 1 + β·r_c` kullan.
Düşük alakalı sınıflar ağırlık **1**'de kalır, **0**'a inmez.

```
r_c = σ( MLP[ E(I) ; E(P_c) ; E(I)⊙E(P_c) ] )   # öğrenilen alaka, elle tablo YOK
w_c = 1 + β·r_c ,   β = softplus(b), b init ≈ −6   # β ≥ 0 YAPISAL
Z_ind = Σ_c w_c·z_c / Σ_c w_c                   # NORMALİZE ağırlıklı havuz
        ⊕ z_ind-query ⊕ z_global-klinik
```

Üç tasarım detayı, üçü de gerekli:

- **Nokta çarpımı değil MLP.** `σ(E(I)ᵀE(P_c)/τ)` biçimi zarif ama **eğitilemez**: her iki gömme de donuk kuleden geliyor, arada öğrenilen parametre yok. Alaka, CXR-BERT'in ön-eğitilmiş benzerlik yapısına sabitlenir. Küçük MLP (768·3 → 128 → 1) bunu açar; `E(I)⊙E(P_c)` terimi ucuz bir etkileşim sinyali verir.
- **Normalize havuz.** Ham `Σ w_c z_c` kullanılırsa havuzun normu, *kaç sınıfın alakalı olduğuna* bağlı olur — geniş bir indication ("chest pain") dar bir indication'dan sistematik olarak daha büyük bir vektör üretir. `Σ w`'ye bölmek göreli vurguyu korur, büyüklük yan etkisini atar.
- **β öğrenilen ama işareti kısıtlı — rev.5 düzeltmesi.** Rev.4'te β serbest bir skalerdi ve `L_ind = Σ(1+βr)·L_ptok` ile birlikte **kaçak bir çözümü** vardı: `L_ptok ≥ 0` olduğu için `∂L/∂β = Σ r·L_ptok > 0` *her zaman*, yani gradyan inişi β'yı ilk adımdan itibaren negatife sürüyordu. Sonuç: `w < 1` — manşet güvenlik özelliği optimizasyonun ilk yüz adımında ihlal, ardından `Σw → 0` kutbu. `β = softplus(b)` ile `w ≥ 1` bir umut değil **yapısal garanti** olur.
- **Ve kayıp da normalize edilir**, havuzla aynı sebeple: `L_ind = Σ_c (1+βr_c)L_ptok,c / Σ_c (1+βr_c)`. Normalize edilmemiş hâlde β'yı küçültmek pozitif bir kaybı ölçeklemekten ibaret olurdu; normalize edildiğinde `∂L/∂β ∝ Cov_c(r_c, L_c)`, yani β gerçekten "alakalı *ve* kötü öğrenilmiş" sınıfları öne çıkarmayı öğrenir.

`E(P_c)` bizde `pos_embs` olarak zaten mevcut — ama
**her adımda değil**: `@torch.no_grad` altında ve yalnızca her
validation'da (400–1200 adım) tazeleniyor. Ek ileri geçiş gerektirmiyor, fakat
alaka MLP'sinin sınıf tabanı **kopuk ve bayat**; MLP bu kaymayı soğurur ve
tazeleme aralığı raporlanmalı.

**BEKLENEN DAVRANIŞ**

| indication: "lung cancer follow-up" | r_c | w_c (β=1) |
|---|---|---|
| Lung nodule | 0.95 | 1.95 |
| Lymphadenopathy | 0.88 | 1.88 |
| Pleural effusion | 0.61 | 1.61 |
| Atelectasis | 0.48 | 1.48 |
| Cardiomegaly | 0.08 | 1.08 |
| Coronary calcification | 0.03 | 1.03 |

*Notlardaki örnek. Alt iki satır mekanizmanın tüm noktası: alakasız sınıflar **hâlâ ölçülüyor**. Indication "şunlara özellikle dikkat et" der, "gerisini yok say" demez.*

**NASIL ÖLDÜRÜLÜR**

**Uyumsuz indication testi.** Nodülü olan bir BT'ye *"rule out pneumonia"*
indication'ı ver. Nodül tahmini kaybolmamalı. Kaybolursa β çok büyük veya
koşullandırma çok güçlü — bu tek test C2'nin varlık sebebini doğrular veya çürütür.

`07 · ayrıldığım yer`

## C3 — Yönlendirme kapısı `[FAZ 2]` ve uçurum düzeltmesi `[FAZ 1]`

**Rev.7'de ikiye ayrıldı.** Boş-bölge uçurumunun kapatılması bir *hata
düzeltmesi* ve Faz 1'de; öğrenilen λ kapısı ayrı bir *iddia* ve Faz 2'de.
İkisini birbirine bağlamak gereksizdi — uçurum, C3 hiç olmasa da düzeltilmeli.

> **notlardan**
>
> > *Maybe just go mask-free with pediatrics (TS masks and the resolution is a big problem)*
> > *For Ped. maybe just omit the anatomy queries, but use pathology queries (with indication)*

Bu öneriyi destekleyen kanıt gerçek ve güçlü: pediatride maske-yönlendirmeli AUC her
modelde maskesizden ~0.03 düşük, küçük çocuklarda bölgeler feature grid'e ulaşamıyor,
plevral kabuğu biz türettik ve pnömotoraksta **−0.34** taşıyor, TotalSegmentator
hacim başına 80–133 saniye.

Buna rağmen **maskeleri elle kapatmayı önermiyorum**, dört sebeple:

- **Ana kanıt karışmış.** §03 Gerçek 2: o −0.03, hard mask'i değil, hiç sınıf prompt'una eğitilmemiş bir organ-havuz okuma başlığını ölçüyor. ARC-CT'nin yayınlanmış çekirdek katkısını karışmış bir ölçüme dayanarak atmak pahalı bir hata olur.
- **Ama önce bir ayrım: maskeyi kaldırmak Q-Former'ı kaldırmak değil.** Rev.4 bu ikisini birbirine karıştırıyordu. Q-Former her hâlükârda duruyor. Ortada iki ayrı karar var: **(a)** çapraz-dikkatteki *sert maske* kalksın mı, **(b)** 10 anatomi query'si slotlarını hak ediyor mu. (a) mekanizma sorusu, (b) kapasite sorusu, ve ikisi bağımsız — maskesiz ama anatomi query'li bir model tamamen tutarlı.
- **Anatomi query'leri düşerse ARC-CT'nin yayınlanmış katkısı pediatrik yarıda kapanır.** Bu, "tek model iki popülasyon" iddiasından daha somut bir bedel: per-organ hizalama kaybı (`L_org`) anatomi query'lerine ve organ maskelerine bağlı. Peds'te onlar yoksa `L_org`'un pediatrik tarafta tutunacağı yer kalmıyor — yani pediatrik model artık ARC-CT değil, düz bir Q-Former CLIP'i olur.
- **Maliyet zaten batmış.** 8.816 pediatrik maske üretildi ve diskte. Onları kullanmamak bugün hiçbir şey tasarruf etmiyor. Asıl tasarruf CT-RATE *train* maskelerinde (~1.740 GPU-saat) ve o karar bu ölçümün *sonucuna* bağlı.

> **FAZ 1 · hata düzeltmesi — C3'ten bağımsız**
>
> `build_role_mask`, boş bir bölgeyi dönmeden önce
> `where(any_key, out, ones)` ile **tamamen serbest** hâle getiriyor.
> Yani bir anatomi query'si, bölgesi ızgaraya ulaşmadığında sessizce global query'ye
> dönüşüyor — hiçbir uyarı üretmeden, ve ölçülen oranlarda: sağ orta lob %5.6,
> central airway %1.5, yetişkinde bile airway %4.1.
>
> **Bu, öğrenilen kapıyı beklemeden düzeltilmeli.** İki parça: (i) ρ, override'dan
> *önce* hesaplanıp döndürülmeli (sonra okunursa boş bölge için ρ = 1.0 çıkıyor,
> sinyalin tam tersi); (ii) boş-bölge oranı eğitim ve değerlendirme loglarında
> raporlanmalı, ki "anatomi yönlendirmesi gerçekten yürürlükte miydi" sorusu
> cevaplanabilsin. Bu haliyle mimari değişikliği değil, **görünürlük düzeltmesi**.

### C3 — Öğrenilen sertlik: kararı modele bırak

**BIÇIMSEL TANIM**

```
# ŞU AN: maske bir duvar
logits[q,n] = −∞  eğer n ∉ R(q)
# ve bölge boşsa maske TAMAMEN kalkar — sert uçurum

# ÖNERİ: duvar bir eğim, dikliği öğreniliyor
ρ_r  = |R(r) ∩ grid| / 12³                  # bu hacimde bölge r ne kadar yer kaplıyor
λ_q^gen = softplus( a_q + w_q·[ ρ ; H̄_dem ] )            # koşulsuz banka: indication GÖRMEZ
λ_q^ind = softplus( a_q + w_q·[ ρ ; H̄_dem ; H̄_ind ] )     # koşullu banka: klinik soru dahil
# a_q init = 8 → her ikisi de başlangıçta ≈ sert maske
logits[q,n] −= λ_q · (1 − m[q,n])

# λ → ∞   ARC-CT'nin sert maskesi (kesin geri kazanım)
# λ → 0   maskesiz Q-Former = notlardaki "mask-free peds"
```

> **rev.5 · iki ölümcül uygulama hatası düzeltildi**
>
> **1 · Boş-bölge muhafızı kapıyı etkisizleştiriyor.**
> `build_role_mask` dönmeden önce
> `out = where(any_key, out, ones)` yapıyor; boş bölgede `m ≡ 1`,
> dolayısıyla `λ·(1−m) = 0` — *her λ için, özdeş olarak*. Yani C3,
> var olma sebebi olan uçurum vakasında hiçbir şey yapmıyor ve
> `∂L/∂λ = 0`. **Düzeltme:** muhafızı kaldır. Güvenli, çünkü tüm
> key'lere uygulanan tekdüze bir `−λ` softmax'ta birbirini götürür —
> tamamen cezalı bir query, kısıtsız global query'ye *sürekli* yakınsar.
> (`has_mask=False` satırları için muhafız kalır; orası doğru davranış.)
>
> **2 · ρ muhafızdan sonra okunursa ters sinyal veriyor.** Rev.4
> *"ρ zaten hesaplanıyor (`out.sum(-1)`)"* diyordu; öyle bir satır
> yok, sadece `out.any(-1)` var. Ve `where`'den *sonra*
> hesaplanırsa boş bölge için **ρ = 1.0** çıkar — maksimum kapsama, gerçeğin
> tam tersi. **ρ, override'dan ÖNCE hesaplanıp ayrıca döndürülmeli.**

`H̄_dem` yaş bandını içerdiği için **λ yaşa göre değişebilir**:
model 2 yaşındaki bir çocukta maskeyi gevşetmeyi, 40 yaşındakinde sıkı tutmayı
öğrenebilir. Ama `λ^gen` indication'ı görmez (§04).

> **a_q = 8 "sert" değil — sızıntı tablosu**
>
> N = 12³ = 1728 key ile, izin verilen `k` token için bölge dışına
> kaçan dikkat kütlesi:
>
> ```
> k = 40 (yetişkin akciğer)   →   %1.40
> k = 10                     →   %5.45
> k =  2 (trakea, 2 yaş)     →  %22.4
> k =  1                     →  %36.7
> # 24³ ızgarada (N=13824) k=2 için %69.9
> # k=2'de sızıntıyı %1'in altına indirmek için λ ≥ 11.4 gerekiyor
> ```
>
> Yani `a_q = 8` ile başlamak *yayınlanmış davranıştan başlamak*
> değil. **Düzeltme:** `a_q` kapsama-duyarlı başlatılır
> (`a_q = log((N−k_q)/(0.01·k_q))`, ampirik ρ'dan), ve sızıntı tablosu
> raporlanır.

**NEDEN FAZ 1'DE DEĞIL — IKI GERÇEK RISK**

**1 · Motivasyonu karışmış bir ölçümden geliyor.** C3'ün gerekçesi pediatride
maske-yönlendirmeli AUC'nin ~0.03 düşük olması; ama §03 Gerçek 2, o cezanın
hard mask'i değil bambaşka bir okuma başlığını ölçtüğünü gösteriyor. Faz 0/1–3
bunu ayrıştırmadan C3'ü teslim listesine koymak, çürük bir gerekçe üstüne
mekanizma kurmak olur.

**2 · Çıkarımda maske gerektiriyor — ve bu yayınlanmış bir özelliği kırar.**
λ, ρ'ya bağlı; ρ maskeden geliyor. Ama ARC-CT'nin sattığı özelliklerden biri
**maskesiz çıkarım**. C3 olduğu gibi konursa model artık çıkarımda maske
istiyor demektir — performans regresyonu değil, *iddia* regresyonu, ki
daha kötü. Üç çıkış var ve biri seçilmeli: (a) λ^gen yalnızca yaş/cinsiyetten
beslensin, ρ yalnızca eğitimde kullanılsın; (b) ρ görüntüden kestirilsin;
(c) λ eğitim sonrası query başına sabite dondurulsun. Hiçbiri kendiliğinden
olmuyor.

**YINE DE NEDEN DEĞERLI — FAZ 2'DE**

"Pediatride maskeleri kapattık, daha iyi oldu" bir mühendislik notudur.
**"Model, 5 yaş altındaki çocuklarda anatomi yönlendirmesini kendi kendine
terk etmeyi öğrendi ve λ yaşla monoton artıyor"** bir bulgudur — ve yaş
koşullandırmasının neden gerekli olduğunun doğrudan kanıtıdır. Aynı deney,
aynı maliyet, çok daha güçlü iddia.

**ÖNCEDEN KAYITLI TAHMIN**

(a) λ, yaş bandıyla **monoton artar** — en genç bantta en gevşek.
(b) Kazanç **bölge 8 ve 9'da yoğunlaşır** — Pneumothorax, Pleural thickening,
Bone lesion. Başka yerde çıkarsa mekanizma iddia ettiğimiz sebeple çalışmıyordur
ve öyle raporlanır.

**YAŞA GÖRE ANAHTARLAMA — AYRIK SÜRÜM**

Bir alternatif: *metadata'daki yaşa bakıp* çocukta anatomi query'lerini ve
maskelemeyi tamamen kapatmak, sadece global + patoloji query'leriyle çalışmak.
Bu, C3'ün **ayrık özel hâli**: λ ∈ {0, ∞}, yaş eşiğiyle anahtarlanan.

Lehine: basit, yorumlanabilir, öğrenilen bir kapının bozulma riski yok. Aleyhine
üç şey: **(i)** eşik keyfî — ölçülen bölge-survival gövde boyutuyla
*sürekli* bozuluyor, 18'de (ya da 5'te) bir sıçrama yok; **(ii)**
havuzlanan query kümesi örnek-başına değişir, bu da §04'teki izolasyon
garantisiyle çakışır (`Z_gen`'in bileşimi sabit olmalı); **(iii)**
yukarıdaki `L_org` bedeli.

**Öneri:** C3'ün sürekli kapısı mekanizma olarak kalsın, ayrık yaş-anahtarı
**merdivende bir kol** olsun. Eğer ayrık sürüm kazanırsa onu raporlarız —
ve o zaman "model yaşa göre anatomi yönlendirmesini kapatmayı öğrendi" yerine
"yaşa göre kapattık, işe yaradı" deriz. İkincisi daha zayıf bir cümle ama
*yanlış* değil; pediatrik tarafta zaten yüksek skor hedeflemiyoruz.

> **[ŞEKİL 3]** **Şekil 3 — Uçurumu kaldırmak.** Sağ sütun asıl sorun: mevcut kodda bir bölge grid'e hiç ulaşmadığında maske *tamamen* kalkıyor, yani query bir anda anatomi query'si olmaktan çıkıp global query'ye dönüşüyor — hiçbir uyarı üretmeden. Çocuklarda sağ orta lobda bu %5.6 oranında oluyor. Öğrenilen kapı aynı davranışı sürekli ve denetlenebilir hâle getirir; `a_q` büyük başlatıldığı için eğitim yayınlanmış davranıştan başlar.

`08 · 27 sınıfın ötesi`

## C4 — Serbest indication query'leri `[FAZ 2]`

### C4 — 4 öğrenilen klinik-soru token'ı

**REV.7 · NEDEN FAZ 1'DEN FAZ 2'YE GERI ALINDI**

**C4, tasarımdaki en büyük kısayol yüzeyi.** 4 serbest query: anatomik maske
yok, yalnızca indication'dan koşullanıyor, doğrudan tahmine besleniyor. Model
"indication X diyor → X tahmin et" öğrenmek isterse en ucuz yer burası —
maskeli patoloji query'leri en azından anatomik olarak makul bir yere bakmak
zorunda, bunlarınki değil. AUC'yi düşürmez, muhtemelen *yükseltir*;
sorun tam da bu. Uyumsuz-indication testinde ilk kırılacak yer burası olur.

Ayrıca §08'in kendi "indication yoksa katkısı sıfır" iddiası mekanize değil:
`H_C` indication düşse bile yaş/cinsiyet token'larını taşıyor, yani
`Q̃_I ≠ Q_I` ve query'ler yaştan türeyen bir şey üretmeye devam ediyor.
Faz 2'ye alınmasının ikinci sebebi bu.

**YINE DE NEDEN DEĞERLI**

Notlarda bu "eventually" olarak, foundation-model uzantısı diye geçiyor.
Bizim kohortumuzda **şimdi** gerekli, çünkü pediatrik hastanenin indication'ları
tam olarak 27 sınıfın dışında kalan kavramlarla dolu:

- *"History of thymoma, evaluate mediastinal recurrence"*
- *"Persistent cough following stem cell transplantation"*
- *"Osteosarcoma surveillance"* — kohortumuzun en sık kısa indication'ı
- *"Evaluate postoperative complication"* — Post-surgical change %33 prevalans

Bunların hiçbiri bir sınıf adı değil; ama hepsi *nereye bakılacağını*
söylüyor. 27 sabit query bu bilgiyi taşıyamaz.

**BIÇIMSEL TANIM**

```
Q_I = { q_1^I , … , q_4^I }                # öğrenilen, sınıfa bağlı değil
Q̃_I = Q_I + CrossAttn( Q_I , H_C , H_C )   # serbest metinden koşullanır
Z_I = CrossAttn( Q̃_I , F , F )             # kısıtsız — anatomik maske YOK
                                           # çünkü hangi organa ait olduğu bilinmiyor
```

Bunlar `Z_ind` havuzuna girer, `Z_gen`'e girmez — yani
indication yoksa (dropout veya CT-RATE'in %50'si) katkıları sıfırdır ve model
bozulmaz.

**YORUMLANABILIRLIK BONUSU**

4 query'nin dikkat haritası, "modelin klinik soruyu nereye çevirdiğini" doğrudan
gösterir — ve bunlar sınıf etiketiyle kısıtlı olmadığı için §15'teki deneyin en
çarpıcı görselleri muhtemelen buradan çıkar.

`09 · sorunun doğrudan cevabı`

## Yaş ve cinsiyet modele tam olarak nereden giriyor

Notlardaki soru: *"Age band embeddings falan modele tam nasıl girecek?"*
Cevap üç cümlede: **bağlam dizisine ek token olarak**, **ordinal-kümülatif
parametrelemeyle**, ve **cinsiyet etkileşimi sıfırdan başlayarak**.

> **tek cümlelik cevap**
>
> Yaş ve cinsiyet **ayrı bir yol değil**: indication token dizisinin sonuna
> iki-üç ek token olarak ekleniyorlar (`H_C = [H_ind ; H_dem]`, her demografik token 768-d'ye izdüşürülerek).
> Böylece C1'in cross-attention'ı, C2'nin alaka skoru ve C3'ün λ kapısı — **üçü de** —
> yaşı görür. Ayrı bir MLP dalı olsaydı yaş yalnızca sınıflandırıcı önselına girerdi;
> oysa yaşın asıl işi *uzamsal ölçek*, yani C3'ün işi.

#### İki kohort yaşta neredeyse hiç örtüşmüyor

| yaş | pediatrik n | % | CT-RATE n | % |
|---|---|---|---|---|
| 0–1 | 301 | 3.4 | 0 | 0.0 |
| 1–2 | 222 | 2.5 | 7 | 0.0 |
| 2–5 | 807 | 9.1 | 0 | 0.0 |
| 5–10 | 1291 | 14.6 | 0 | 0.0 |
| 10–13 | 1097 | 12.4 | 0 | 0.0 |
| 13–18 | 2701 | 30.5 | 0 | 0.0 |
| 18–21 | 1305 | 14.7 | 747 | 1.6 |
| 21–25 | 740 | 8.3 | 2102 | 4.5 |
| 25–30 | 250 | 2.8 | 3660 | 7.8 |
| 30–40 | 114 | 1.3 | 10049 | 21.3 |
| 40–50 | 26 | 0.3 | 9095 | 19.3 |
| 50–60 | 13 | 0.1 | 7490 | 15.9 |
| 60–70 | 3 | 0.0 | 6880 | 14.6 |
| 70+ | 0 | 0.0 | 7107 | 15.1 |

*Yaşlar metadata'dan: pediatrik tarafta `search_CHESTCT_8k.xlsx` sütun M, CT-RATE tarafında DICOM `PatientAge`. Vurgulu satırlar tek köprüyü işaretliyor. **Sayım netleştirmesi:** yaş tablosu **8.870 accession** üzerinden; modelleme kohortu **8.816 hacim** (8.817 çekildi), split 7.042 / 1.774, skorlanan **1.772**. Bu belgede "kohort" = 8.816.*

### Neden skaler değil — üç gerekçe

- **Prevalans yaşta ne doğrusal ne monoton.** Koroner kalsifikasyon 30'a kadar fiilen sıfır, sonra patlar. Mukus tıkacı KF'li çocuklarda tepe yapar, gençlerde düşer. Skaler bir yaş, ağın bu şekilleri tek bir doğrusal yönden üretmesini zorlar.
- **Yaş, uzamsal ölçeğin vekilidir.** Kohortta gövde çapı **2.4 kat** değişiyor, ızgara ise sabit 192×192×96. Aynı yapı 6 aylıkta ve 16 yaşındakinde bambaşka sayıda token kaplıyor. Yönlendirme tablosu tam da token sayısına bağlı — bu yüzden yaş yalnızca sınıflandırıcı önselına değil, **C3 üzerinden yönlendirmeye** girmeli.
- **Kohortlar arasında boşluk var.** CT-RATE'te 18 yaş altı 47.137'de 7 hacim. Skaler yaş, modeli hiç verisi olmayan bir aralıkta doğrusal ekstrapolasyona zorlar. Bantlar boşluğu *açıkça* temsil eder.

### Ordinal-kümülatif parametreleme

Düz bir gömme tablosu bantların *sıralı* olduğunu unutur: 2–5 yaş ile 5–10 yaş
arasındaki mesafe, 2–5 ile 60+ arasındakiyle aynı şekilde öğrenilir. Bunu düzeltmek
için gömmeyi artımların kümülatif toplamı olarak yazıyoruz:

```
e_age(b) = e_0 + Σ_{j ≤ b} softplus(Δ_j)      # Δ ∈ R^{9×64}, küçük init

# sıralama yapıya gömülü (softplus ≥ 0 ⇒ monoton yürüyüş)
# aralıklar ÖĞRENİLİYOR — yani doğrusal değil, ama keyfî de değil

e_sex ∈ R^{16}                              # 3 satır: K / E / bilinmiyor
e_int[b, s] ∈ R^{32}   SIFIR-INIT, wd ×10     # saf toplamsaldan başlar

# rev.5 DÜZELTME: bunlar H_ind'e (768-d CXR-BERT token dizisi) doğrudan
# eklenemez — 64/16/32 ≠ 768. Her biri kendi izdüşümünden geçer:
P_age: R^64 → R^768 ,  P_sex: R^16 → R^768 ,  P_int: R^32 → R^768  (P_int sıfır-init)
H_dem = [ P_age·e_age ; P_sex·e_sex ; P_int·e_int ]     # 3 token, her biri 768-d
```

| bant | aralık | klinik ad | peds n | adult n | not |
|---|---|---|---|---|---|
| 0 | < 1 y | infant | 301 | 0 | en büyük ölçek farkı |
| 1 | 1–2 y | toddler | 222 | 7 | CT-RATE'in 7'si muhtemelen metadata hatası |
| 2 | 2–5 y | okul öncesi | 807 | 0 | trakea/özofagus grid'den düşüyor |
| 3 | 5–10 y | okul çağı | 1291 | 0 |  |
| 4 | 10–13 y | ergenlik öncesi | 1097 | 0 |  |
| 5 | 13–18 y | ergen | 2701 | 0 | pediatrik kohortun tepe noktası |
| 6 | 18–25 y | genç erişkin | 2045 | 2849 | **tek köprü bant** — transferin yaşandığı yer |
| 7 | 25–40 y | erişkin | 364 | 13709 |  |
| 8 | 40–60 y | orta yaş | 39 | 16585 | ateroskleroz burada başlıyor |
| 9 | 60+ y | yaşlı | 3 | 13987 | pediatrik tarafta fiilen boş |

*10 bant, iki kohortu da kapsıyor. Model **asla bir kohort/site bayrağı görmez** — yalnızca yaşı görür. Bu kasıtlı: kohort bayrağı en kolay kısayoldur ve "tek model, iki popülasyon" iddiasını anında çürütür.*

> **tasarım sonucu**
>
> Bu, "pediatrik kohortumuzun %27.6'sı yetişkin" sorununu bir kusurdan bir
> *özelliğe* çevirir. Model iki etiket görmez, bir *süreklilik* görür.
> 69 yaşındaki konjenital pulmoner stenoz hastası da, 6 aylık bebek de aynı yaş
> eksenine oturur ve model ikisini de kendi bandından okur.

`10 · notlardaki soru işareti`

## Çözünürlük ve gövde boyutu

Notlarda iki kez soru işaretiyle geçiyor. Bu bölüm rev.5'te **tamamen yeniden
yazıldı**: önceki sürüm eğitim-zamanı log dosyalarından kopyalanmış sayılar
kullanıyordu ve bir sınıfın işareti tersti.

> **geri çekilen iddia**
>
> Rev.4 şöyle diyordu: *"Lung nodule 0.6727 → 0.6647, −0.0080 (eğitim döngüsünün kendi değerlendiricisi) — küçük odaksal
> bulgunun ders kitabı örneği ve çözünürlük artınca kötüleşti."* O tablo
> `runs/peds_finetune_v*/best_auc_update*.txt` dosyalarından, yani
> **eğitim döngüsünün kendi değerlendiricisinden** alınmıştı. §01'in kullandığı
> tek-değerlendirici `eval_matrix`'e göre aynı fark **+0.0021**.
> İşaret ters, ve üstüne kurulu anlatı geçersiz.
>
> Ayrıca rev.4'te Lung nodule'ün −0.0080'i "kötüleşme" diye yorumlanıyordu; o sınıfın
> **ölçülmüş tohum bandı 0.0118** — yani her iki sayı da kendi gürültüsünün içinde.

### Ölçtüğümüz: 24³ ızgaranın kazancı dört sınıftan geliyor

| sınıf | V3 (12³) | V5 (24³) | Δ | val poz. | not |
|---|---|---|---|---|---|
| Bone lesion or fracture | 0.5700 | 0.6153 | +0.0453 | 293 | en iyi destekli kazanç — ama GA 0.115, raporlanabilirlik tabanının altında (§11) |
| Hiatal hernia | 0.7540 | 0.7882 | +0.0342 | 18 | GA genişliği 0.223 |
| Arterial wall calcification | 0.7354 | 0.7659 | +0.0305 | 34 | GA genişliği 0.185 |
| Coronary artery wall calc. | 0.7796 | 0.8062 | +0.0266 | 5 | GA genişliği **0.394** |
| Pulmonary metastases | 0.6907 | 0.7126 | +0.0219 | 94 |  |
| Lung nodule | 0.6726 | 0.6747 | +0.0021 | 910 | tohum bandı 0.0118 → yorumlanamaz |
| 15 sınıf | — | — | negatif | — | en kötüsü Pulmonary cyst −0.0168 |
| **ORTALAMA (27)** | 0.7972 | 0.8019 | **+0.0047** | — |  |
| **ORTALAMA (ölçülebilir 23)** | 0.7941 | 0.7957 | **+0.0016** | — | **tohum bandının içinde** |

*Tek değerlendirici (`eval_matrix/peds`), aynı 1.772 hacim. Düşürülen dört sınıfın (Emphysema, Coronary, Hiatal hernia, Arterial wall calc) V3→V5 toplam katkısı **+0.0886**, yani 27'ye bölündüğünde **+0.0033** — 27-sınıf kazancının **%71'i**.*

> **doğru sonuç, ve rev.4'tekinden daha güçlü**
>
> **Çok ölçekli ızgara, yeterli desteğe sahip sınıflarda ölçülebilir bir kazanç
> üretmiyor.** 27-sınıf ortalamasındaki +0.0047'nin %71'i, validation'da
> sırasıyla **3, 5, 18 ve 34** pozitifi olan dört sınıftan geliyor — biri
> (Coronary) 0.394 genişliğinde bir güven aralığına sahip. Ölçülebilir 23 sınıfta
> kazanç **+0.0016**, yani tohum bandının içinde.
>
> Bunun için "Lung nodule kötüleşti" demeye gerek yok — o iddia hem yanlıştı hem de
> gereksizdi. Ve bu, V5'i "en keskin takas" yapan şeyi de açıklıyor: pediatrik kazancı
> ölçülemez, yetişkin kaybı (−0.0488) ise gerçek.

### Çözüm yönü: kırpma — bu bir direktif, hipotez değil

> **Dr. Kurugol**
>
> > *We can apply cropping for pediatrics data, especially infants etc they have really small lungs — instead of downsampling directly, we can apply cropping and get the most useful data in that CT.*

Bu, rev.5'te hipotez olarak yazdığım şeyin direktif hâli, ve mekanizmayı da bir yerde
düzeltiyor. **Sorun "downsampling" değil, sabit kutu.** Boru hattı zaten
1.5×1.5×3.0 mm'ye örnekliyor — bebek BT'leri sıklıkla bundan ince çekildiği için orada
yapılan şey teknik olarak *downsampling* bile olmayabilir. Asıl israf sabit
**192×192×96** kutusunda: bebeğin gövdesi kutunun küçük bir kısmını kaplıyor,
geri kalanı dolgu. Ve 12³ öznitelik ızgarası *tüm kutu* üzerinden hesaplandığı
için, 1728 token'ın çoğu dolguya bakıyor.

```
# bugün: her token ≈ 24 × 24 × 24 mm fiziksel
192×192×96 girdi  →  12³ ızgara  →  1 token = 16×16×8 voxel

# bebekte gövde ~90×90×60 voxel ise, anatomi ~5.6×5.6×7.5 token'a sığışıyor;
# akciğerler 2–3 token genişliğinde. Izgarayı sıklaştırmak bunu değiştirmez,
# çünkü dolguyu da aynı oranda sıklaştırır.

# kırpma sonrası: aynı 12³ ızgara, ama tamamı dokuya bakıyor
gövde kutusuna kırp (90×90×60)  →  192×192×96'ya örnekle  →  1 token ≈ 11 mm
# token yoğunluğu (organ başına token) yaşa göre yaklaşık sabitlenir
```

> **rev.5 · ÖLÇÜLDÜ — ve baskın etki dolgu değil, KIRPMA çıktı**
>
> Rev.4/5 *"dolgu medyan %18, en küçüklerde %77"* diyordu; o sayı
> `dataset.py`'de bir yorum satırıydı. Şimdi 500 hacimlik rastgele örneklemde
> gerçekten ölçtüm — ve **ikisi de yanlış yönü gösteriyormuş**:
>
> ```
> # PEDS_NPZ: tam FOV, 1.5×1.5×3.0 mm, DEĞİŞKEN boyut (192³ kutusu yükleme anında uygulanıyor)
> düzlem içi (H)  medyan 215   min  95   max 333      # kutu 192
> kesit      (D)  medyan  92   min  32   max 698      # kutu  96
>
> 192×192×96 kutusuna göre:
>   düzlem içi   %71.2 KIRPILIYOR (veri atılıyor)    %27.8 dolgu
>   kesit        %41.4 kırpılıyor                     %57.8 dolgu
>
> kutunun gerçek veriyle dolu oranı   medyan %94.8   çeyrekler %72.1 / %100   min %8.4
> ⇒ DOLGU oranı                       medyan  %5.2   max %91.6
> ⇒ verisi ATILAN hacim oranı         %72.2
> ```
>
> Yani medyan dolgu **%18 değil %5.2**, ve hacimlerin **%72'sinde sorun dolgu değil,
> sabit kutunun çevreyi kesip atması**. Medyan düzlem içi 215 voxel, kutu 192 —
> her kenardan ~12 voxel, yani **~35 mm** merkezden uzağa doğru atılıyor.

> **bu, kırpma direktifini zayıflatmıyor — iki kat güçlendiriyor**
>
> Sabit kutu **iki ayrı şekilde** zarar veriyor ve gövde-normalize kırpma ikisini
> birden çözüyor:
>
> - **Büyük çocuklarda (%72):** çevre kesiliyor. Kesilen şey tam olarak **bölge 9 (göğüs duvarı, kaburga, omurga)** ve **bölge 10 (üst batın)** — şemamızın en dıştaki iki bölgesi. Kırpma gövde kutusuna yapılınca kesilen kısım hava ve masa olur, doku değil.
> - **Bebeklerde (uç %8):** dolgu %91.6'ya çıkıyor, ızgaranın onda dokuzu boşluğa bakıyor — hocanın işaret ettiği durum, ve gerçek, sadece *nadir*.
>
> **Ve bu, elimizdeki en zayıf sınıfı açıklıyor olabilir.** Pediatrik en düşük AUC
> `Bone lesion or fracture` = **0.5700** ve tek adresi bölge 9 —
> yani kesilip atılan bölge. Hemen ardından `Pleural thickening` 0.6784,
> bölge 8. İkisi de çevresel.
>
> **Önceden kayıtlı tahmin:** sınıf-başına AUC, hacmin *kırpılma* oranıyla ters
> orantılı olmalı, ve etki **bölge 8/9/10 sınıflarında** yoğunlaşmalı.

> **tahmin koşuldu · **ÇÜRÜDÜ****
>
> 1.772 pediatrik validation hacmi, V2 checkpoint'i, hacim başına kırpma oranı
> hesaplanıp alt ve üst tertile karşılaştırıldı:
>
> ```
> kırpma oranı   medyan 0.217   çeyrekler 0.000 / 0.428   max 0.968
> düşük-kırpma n=594 (≤0.031)   ·   yüksek-kırpma n=590 (≥0.368)
>
> ÇEVRESEL (5 sınıf)  ortalama AUC farkı  = +0.0070
> MERKEZİ  (13 sınıf) ortalama AUC farkı  = −0.0084
> KONTRAST                                = +0.0154
>    hasta-bazlı bootstrap %95 GA  [−0.0757, +0.0873]   P(≤0) = 0.407
> ```
>
> **Sıfırdan ayırt edilemiyor.** Dahası merkezî sınıfların kendi içindeki dağılım
> (Pulmonary metastases **+0.1243** … Pulmonary fibrotic sequela **−0.0774**)
> kontrastın kendisinden bir mertebe büyük — yani ölçülen şey gürültü.
>
> Ve manşet tahmin **ters yönde çıktı**: `Bone lesion or fracture`,
> bölge 9'un tek sınıfı, düşük-kırpma grubunda **0.5013** → yüksek-kırpma grubunda
> **0.5613**. Yani bölgesi *tamamen mevcutken* model şans seviyesinde.
> Bu sınıfın zayıflığı bir kırpma artefaktı değil.

> **testin sınırı — ve ne öldü, ne ölmedi**
>
> Bu test müdahaleyi değil **alt grupları** karşılaştırıyor: yüksek-kırpma hacimleri
> sistematik olarak daha büyük/daha yaşlı çocuklar, yani hastalık karışımları da farklı.
> Yani ucuz bir tarama, kesin bir ret değil. Gerçek test kırpmayı uygulayıp yeniden
> değerlendirmek.
>
> **Ayakta kalan:** ölçümün kendisi — hacimlerin %72'sinde sabit kutu veri atıyor,
> bu bir olgu. Kırpma o israfı kaldırıyor ve **hocanın direktifi**.
> **Düşen:** "çevresel sınıfları düzeltecek" beklentisi. O yüzden kırpma artık
> *yüksek beklenen kazançlı* bir kalem değil; **ucuz, direktif destekli, getirisi
> belirsiz** bir kalem. Faz 2'de kalıyor ve 2×2 ablasyon kolu bunu ölçecek.

```
1. gövde sınırlayıcı kutusunu bul   # HU eşiği + en büyük bağlı bileşen
2. CT ve MASKE aynı kutuya kırpılır  # ikisi de AYNI koddan, aynı kutuyla
3. 192×192×96'ya yeniden örnekle    # token yoğunluğu yaşa göre sabitlenir
4. voxel boyutu bağlama eklenir       # H_dem'e ek token: mm/voxel
```

#### Akciğere değil, gövdeye kırpılmalı

Cümle "small lungs" diyor ama kutu akciğere kırpılırsa şemamızın **iki bölgesi
yok olur**: bölge 9 (göğüs duvarı, kaburga, omurga) ve bölge 10 (üst batın) —
ikisi de akciğer dışında. Ve bölge 9, Bone lesion'ın (%13.9 prevalans) tek adresi.
Kırpma *gövde* kutusuna yapılır; kollar hariç tutulur.

#### Mutlak ölçek kaybolur — ama geri verilebilir

Kırpma sonrası voxel boyutu hastaya göre değişir, yani "5 mm nodül" artık sabit
sayıda voxel değil. Bu, boyut ölçütü olan bulgular için gerçek bir risk.
**Çözüm mimaride hazır:** mm/voxel değeri `H_dem`'e ek bir token
olarak girer. Yaş zaten uzamsal ölçeğin *vekiliydi*; artık ölçeğin kendisini
veriyoruz.

#### Maske ile CT aynı yoldan geçmeli

Bu kusur bir kez zaten yaşandı: maske ve CT farklı geometrik yollardan 192³'e
ulaşınca 186 voxel genişliğinde bir çocukta akciğer etiketi altındaki ortalama HU
−281, arka plan −528 çıkmıştı. Kırpma kutusu **bir kez** hesaplanıp ikisine de
aynı fonksiyondan uygulanmalı.

#### C3 ile kısmen ikame

Kırpma bölgelerin ızgaraya ulaşma oranını yükseltirse, C3'ün λ kapısının çocukta
gevşetecek bir şeyi azalır. İkisi kısmen aynı problemi çözüyor, o yüzden ablasyon
**2×2** olmalı: kırpma açık/kapalı × kapı açık/kapalı. Aksi hâlde birinin
kazancı ötekine yazılır.

**Maliyet — rev.5'te düzeltildi.** Önceki sürüm "npz yeniden üretilmek zorunda"
diyordu; bu yanlıştı ve ham veriden türetmeyi ima ediyordu. Gerçek durum çok daha ucuz:
**ham NIfTI'ye (3.5 TB) dokunulmuyor** ve **TotalSegmentator yeniden koşmuyor**
(asıl pahalı kalem oydu; `TS_MASKS_PEDS` 8.816 girdiyle diskte).
Kırpma, elimizdeki npz'nin bir dönüşümü:

```
CT   : mevcut npz (tam FOV) → gövde kutusu → 192³        # saf CPU dönüşümü
MASKE: TS_MASKS_PEDS → AYNI kutu → 192³                 # GPU yok, sadece resample

# iki uygulama seçeneği:
(A) yeni npz yaz            # +125 GB disk, yükleme hızlı
(B) kutu = 6 tamsayı/hacim  # ~0 disk, kırpma+resize yükleme anında
    yan dosya olarak sakla  # dataloader CPU'su darboğaz olursa (A)'ya geç
```

(B) muhtemelen doğru başlangıç: 8.816 × 6 tamsayı önemsiz bir dosya, ve mevcut
`_pad_crop_hwd` zaten yükleme anında çalışıyor — yani mimari değişikliği yok.

`11 · sınıf kararları`

## Sınıf şeması: sorunun cevabı

> **notlardan · doğrudan soru**
>
> > *Adult only olan classları (emphysema, hiatal hernia ve calcification) pediatric scorelamadan çıkart. Coronary wall cal. kesin çıkacak ama arterial wall calc? Pneumothorax'ı da çıkarabiliriz.*

**Önce en kritik ayrım:** "skorlamadan çıkart" ile "modelden çıkart" aynı şey değil
ve buradaki fark yetişkin kapısını belirliyor. O sınıflar yetişkin tarafındaki
*en güçlü* sınıflarımız: Arterial wall calc. **0.9365**, Coronary **0.9321**,
Hiatal hernia **0.8363**. Head silinirse bu beceri silinir.

> **önce bir sayı karışıklığını önleyelim — iki kohort, iki ayrı sayı**
>
> "ARC-CT paperinde 0.9'du, şimdi 0.7'lerde" diye okunabilecek bir durum **yok**.
> 0.9 *yetişkin* sayısı, 0.7 *pediatrik* sayısı — aynı sayının düşmüş
> hâli değil, farklı kohortlarda ölçülmüş iki farklı şey. Yetişkin tarafı duruyor:
>
> | sınıf | YETİŞKİN · CT-RATE val | PEDİATRİK · BCH val |  |  |  |  |
> |---|---|---|---|---|---|---|
> | ARC-CT | C47mix | C47harm | V1 | V2 | C47harm |  |
> | Arterial wall calc. | 0.9365 | 0.9367 | 0.9313 | 0.7419 | 0.7339 | 0.7036 |
> | Coronary artery wall calc. | 0.9321 | 0.9280 | 0.9305 | 0.8214 | 0.7806 | 0.8641 |
> | Emphysema | 0.8261 | 0.8044 | 0.8142 | 0.9930 | 0.9911 | 0.9966 |
> | Hiatal hernia | 0.8363 | 0.8107 | 0.8220 | 0.7749 | 0.7496 | 0.8107 |
>
> *Yetişkin sütunları ARC-CT'ye göre en fazla **−0.0143** oynamış ve dördü de kendi güven aralığının içinde. Yani ortak eğitim bu sınıflarda yetişkin becerisini *henüz* bozmadı — pediatrik prevalans %0.3–1.6 olduğu için gradyan zaten çok küçük. Pediatrik sütunlardaki Emphysema 0.99 değeri **3 pozitiften** geliyor.*
>
> **Ama endişenin mekanizması gerçek ve tabloda görünüyor:** Arterial wall calc'ın
> *pediatrik* sayısı koşu koşu düşüyor — 0.7419 → 0.7339 → **0.7036**, yetişkin
> verisi 14.7k'dan 42.5k'ya çıkarken. Bu bir bozulma değil, **kanıt**: model giderek
> sınıfın *yetişkin* anlamını (aterosklerotik kalsifikasyon) öğreniyor ve
> pediatrik "pozitifler" (ligamentum arteriosum, kateter izi, kıkırdak) ona uymuyor.
> Yani iki etiket gerçekten farklı hastalık. Kayıp maskesi bunu ileriye dönük olarak
> kesiyor ve pediatrik skoru anlamsız bir sayı olmaktan çıkarıyor.

> **mekanizma — ve senin kastettiğin tam olarak bu**
>
> 27 head'in hepsi kalır. **(hacim, sınıf) başına kayıp maskesi** verilir: pediatrik
> hacimlerde ilgili sütunlar gradyan üretmez, yetişkin hacimlerde tam güçle eğitilir.
> Bu makine zaten yazıldı — NaN hücrelerini maskeleyen düzeltme (kusur #6) tam olarak
> bu iş için var. **Pediatrik taraf bu sınıflarda modeli hiç etkilemez** — ne gradyan
> üretir ne skora girer — ama yetişkin tarafında head tam güçle eğitilmeye devam eder.
> Raporlamada pediatrik manşet **"ölçülebilir 23"** üzerinden verilir
> (27 eksi Emphysema, Coronary, Hiatal hernia, Arterial wall calc). V2 için karşılaştırma:
> 27-sınıf 0.7974 · 23-sınıf **0.7944** — fark bir tohum bandı kadar.

| sınıf | peds prev. | peds val poz. | adult prev. | adult AUC | karar |
|---|---|---|---|---|---|
| Emphysema | 0.3% | 3 | 19.4% | 0.8261 | peds kayıp maskeli · peds skorunda **yok** |
| Coronary artery wall calc. | 0.3% | 5 | 25.5% | 0.9321 | peds kayıp maskeli · peds skorunda **yok** |
| Hiatal hernia | 0.9% | 18 | 14.3% | 0.8363 | peds kayıp maskeli · peds skorunda **yok** |
| Arterial wall calc. | 1.6% | 34 | 28.4% | 0.9365 | **ÇIKSIN** — peds'te kayıp maskeli, peds skorunda yok. Gerekçe aşağıda; ama beklenen yetişkin kazancı **küçük** (ölçüldü: −0.0052). |
| Pneumothorax | 2.3% | 37 | 0.5% | — | **KALSIN.** C3'ün ilan edilmiş test vakası. |

### Arterial wall calcification: neden çıkarıyoruz — ve neden yetişkin kapısını bu kurtarmayacak

Bu sınıfın yetişkindeki anlamı **aterosklerotik sistemik arter kalsifikasyonu** ve
ARC-CT'nin en güçlü sınıflarından biri: **0.9365**. Çocukta ise pratikte
*başka bir hastalık*. Kendi kanıt dosyamızın alıntıladığı cümleler:

> *"Normal variant calcification of the ligamentum arteriosum"* ·
> *"prominent tracheal and bronchial cartilage calcifications"* ·
> *"calcification in the left brachiocephalic **vein**"* ·
> *"calcifications … reflecting the tract of the Port-A-Cath"* ·
> *"Peripheral calcified densities … in the left pulmonary arteries"*

Ven kalsifikasyonu, kateter izi, hava yolu kıkırdağı, normal varyant, ve sistemik değil
*pulmoner* arter. Tanım daraltıldıktan sonra prevalans %2.78 → %1.61'e indi, ama
geriye kalan 34 pozitifin aterosklerotik olduğunu iddia edemeyiz — çocukta ateroskleroz
yoktur. **Aynı ad altında farklı hastalık üzerinde gradyan üretmek, 0.9365'lik bir
yetişkin dedektörünü bulanıklaştırır.**

> **ama beklentiyi doğru kur**
>
> Yetişkin kapısını bu kurtarmayacak. Ölçtük: C47harm'da bu sınıfın kaybı
> **−0.0052**, C47mix'te **+0.0002** — yani zaten neredeyse korunmuş durumda.
> Sebebi basit: pediatrik prevalans %1.6, yani pediatrik gradyan zaten çok küçük.
> Kayıp maskesi doğru karar ama **bilimsel hijyen** gerekçesiyle doğru
> (34 pozitiflik, farklı hastalığa ait bir sayı manşet ortalamada olmamalı),
> yetişkin kapısı gerekçesiyle değil.
>
> Aynı şey diğer üç nadir sınıf için de geçerli: Coronary −0.0016, Emphysema −0.0119,
> Hiatal hernia −0.0143 — **dördü de kendi güven aralığının içinde**. Maskelense bile
> toplam yetişkin kazancı **en fazla ~0.0018**, muhtemelen sıfır. Açık ise −0.0132.
> **Yetişkin açığı başka bir yerde** — §13.

> **rev.5 · raporlanabilirlik tabanı — ayrı ve bağımsız kural**
>
> Sınıf seçimi **hastalık kimliğine** göre yapılır; pozitif sayısı raporlanır ama
> ölçüt değildir. Ölçüt olsaydı tutarsız olurdu: Pericardial effusion 84 pozitifle
> kalıyor (GA genişliği **0.200**) ama Arterial wall calc 34 pozitifle çıkıyor (0.185).
>
> Bunun yerine **önceden ilan edilen** ikinci bir kural: %95 bootstrap GA genişliği
> **0.10'u aşan** hiçbir sınıf üzerinde mekanizma iddiası kurulmaz. Ölçülen (n=1.772):
> Coronary 0.394 · Hiatal hernia 0.223 · Pericardial effusion 0.200 · Arterial wall calc
> 0.185 · Pulmonary metastases 0.164 · **Pneumothorax 0.161** · Pulmonary cyst 0.144 ·
> Mass 0.118 · **Bone lesion 0.115**.
>
> **Bedeli açık: Pneumothorax ve Bone lesion tabanın altında** — C3'ün bölge-8/9
> tahmini bu kohortta bu n ile *test edilemez*. Sonucu gördükten sonra değil,
> şimdi söylüyoruz. n₁ ≤ 20'de percentile bootstrap dejenere oluyor (Emphysema 3
> pozitifle sahte 0.007 genişlik veriyor); orada Hanley–McNeil raporlanır.

### Pnömotoraks: neden modelden çıkarmamayı öneriyorum

**BT pnömotoraks için referans standarttır** — akciğer grafisinde görünmeyen okült
pnömotoraksları yakalayan modalite BT'dir. "BT'den tam belirlenemiyor" gerekçesi
incelemede sorun çıkarır.

Ölçtüğümüz iki gerçek sorun var ve ikisi de *bizim*: prevalans %2.3 (199 pozitif, validation'da 37) ve routed cezası **−0.3369** — bizim türettiğimiz plevral kabuğa yönlendirildiği
için. Kanıt yan sınıfta: `Pleural effusion` `[8,2,5]`'e, yani
plevra *artı yedek loblar*a gidiyor ve cezası küçük.

Bir radyolog itirazı *rapor* güvenilirliğiyle ilgiliyse (küçük apikal pnömotoraks
raporda anılmayabilir), bu bir etiket-gürültüsü argümanıdır — çözümü yine aynı
maskeleme mekanizması, sınıfı silmek değil.

`12 · amaç fonksiyonu`

## Kayıplar

Notlardaki ilke: mevcut üç kaybı koru, üstüne indication kaybını ekle.
Bizim durumumuzda bu, düşünülenden de küçük bir değişiklik.

```
L = L_clip                      # Jaccard-yumuşatılmış soft InfoNCE (FN_WEIGHT 0.3) — DEĞİŞMEZ
  + L_cls                       # prompt BCE, Z_final üzerinden — DEĞİŞMEZ
  + L_org                       # organ latent ↔ bölge cümlesi — DEĞİŞMEZ
  + L_ptokgen              # KOŞULSUZ banka — ARC-CT'deki hâliyle, ağırlıksız
                                #   (yetişkin kapısını ve adım-0 hikâyesini bu korur)
  + Σ_c (1+β·r_ic)Lindptok,c / Σ_c(1+βr_ic)   # KOŞULLU banka — alaka ağırlıklı, normalize
  + λ_loc · L_loc                  # zayıf lokalizasyon — KOŞULSUZ bankaya (aşağıda)
  + λ_cf · Σ_c 1[r_c<τ]·‖ŷ_c(I) − ŷ_c(I')‖²   # karşıolgusal — TAHMİN üzerinde
```

#### L_ind bir satır

Notlarda *"you aren't really introducing a completely new training objective"*
deniyor ve bu doğru — `loss_pertoken` zaten var. **Ama "tek satır" değil**
(rev.4'ün iddiası): mevcut kod `qf_tokens[:, A:A+P]` diliminde
`P=27` varsayıyor; çift bankayla `P=54` olur ve
`pos_embs` ile broadcast **hata verir**. Ayrıca hangi bankayı
denetlediği söylenmeli. Rev.5 kararı: **koşulsuz banka ARC-CT'deki ağırlıksız
kaybı korur** (yetişkin kapısı ve adım-0 özdeşliği ona bağlı), alaka ağırlıklı
terim **ayrı bir kayıp olarak koşullu bankaya** uygulanır. Bu aynı zamanda C2
ablasyonunu temiz hâle getirir.

#### Karşıolgusal tutarlılık — geri getirildi, doğru büyüklük üzerinde

Notlar "counterfactual indication training occasionally" diyor: aynı BT, farklı
indication'lar; *genel bulgular sabit kalmalı, dikkat değişmeli*.
İlk taslakta bunu bir kayıp terimiyle
(`‖Z_gen(I) − Z_gen(I')‖²`) zorluyordum.

Rev.3'te bu terimi *kaldırmıştım*, gerekçem şuydu: izolasyondan sonra
`Z_gen(I) − Z_gen(I')` tanım gereği sıfır, yani ceza kimliksel olarak
sıfır bir şeyi optimize eder. **Gerekçe doğru ama yanlış büyüklük hakkında.**

Güvenlik iddiası `Z_gen` hakkında değil, **tahmin** hakkında:
"pnömoni için çekilmiş BT'deki pnömotoraks kaybolmasın". Tahmin ise
`Z_final = W[Z_gen ; Z_ind]`'den geliyor ve `W`'nin sağ bloğu
adım 0'dan sonra **kısıtsız** eğitiliyor, `Z_ind` de tamamen
indication'a bağlı. Yani izolasyon `∂Z_gen/∂I = 0`'ı kanıtlıyor ama
`Z_final` hakkında hiçbir şey söylemiyor.

**Rev.5:** terim geri geliyor, **düşük alakalı sınıfların logitleri** üzerinde.
Ayrıca eğitim boyunca `‖W_ind‖ / ‖W_gen‖` oranı bir teşhis olarak
izlenir ve raporlanır — koşullu yolun füzyonu ne kadar ele geçirdiğini gösterir.

#### L_loc: artefakt hazır, ama bedava değil

Artefakt hazır: `region_cache.json`, her hacim için 10 bölgeye LLM ile
atanmış cümleleri tutuyor (55.966 anahtar). **Ama rev.4 bunu yanlış bağlıyordu.**
Cache *rapor bulgularını* bölgelere eşliyor, indication'ı değil. Koşullu
query'yi oraya çekmek §15'in manşet deneyiyle **doğrudan çelişir**: aynı BT'nin
raporu üç indication'da da aynı, yani hedef sabit — oysa C1'in tüm amacı dikkatin
*değişmesi*. Kendi kaybımız manşet figürü düzleştirirdi.
**Rev.5:** `L_loc` yalnızca **koşulsuz bankaya** uygulanır — orada
"bulguların olduğu yere bak" tutarlı ve indication'dan bağımsız bir hedeftir.
Ve "bedava" değil: `return_attn=True` füzyonlu SDPA yolunu kapatıp
batch başına ~77 MB dikkat tensörü materyalize ediyor.

#### Neyi eğitmiyoruz

Modeli indication'ı *yeniden üretmeye* eğitmiyoruz — notlardaki uyarı bu.
Indication bir hedef değil, bir koşul. Aksi hâlde metin kanalı görüntüyü
açıklamak yerine kendini açıklamayı öğrenir.

### Parametre grupları

| bileşen | durum | LR | gerekçe |
|---|---|---|---|
| 3B ResNet-18 | eğitiliyor | 1.1e-5 | ARC-CT ile aynı · koşullandırılmıyor |
| CXR-BERT gövde | DONUK | — | zero-shot prompt uzayı buna bağlı |
| Text LoRA (rapor yolu) | eğitiliyor | 1e-6 | prompt uzayını az kaydırsın |
| Indication yolu | **AYRI DONUK KOPYA** | — | rev.5: "aynı kuleyi LoRA'sız kullan" **uygulanamaz** — LoRA modülü yerinde değiştiriyor ve `to_text_latent` de eğitiliyor. Gerçek donuk yol ikinci bir CXR-BERT kopyası + kendi izdüşümü demek (bellek + ikinci ileri geçiş). |
| Anatomi query (10) | eğitiliyor | 1.1e-5 | bölge anlamları peds'te değişti |
| Yetişkin patoloji satırı (18) | eğitiliyor | 1.1e-6 | **0.1×** — yetişkin becerisini koru |
| Yeni patoloji satırı (9) | eğitiliyor | 1.1e-5 | sıfırdan öğreniliyor |
| C1–C4 + bağlam encoder | eğitiliyor | 1.1e-5 | sıfır-init, tam hızda öğrensin |

`13 · projenin sert kısıtı`

## Yetişkin kapısını kapatmak

Hedef: CT-RATE 18 sınıfta ARC-CT'yi (0.8574) geçmek veya farkı gürültü bandına
indirmek. Bugün en iyimiz 0.8442 — açık **0.0132**.

> **çerçeve değişikliği**
>
> Şu ana kadar bu açığı "kaybettiğimiz beceriyi geri kazanma" problemi olarak
> kurduk. **Bağlam koşullandırma bunu bir kazanma problemine çevirir.**
> 18 yetişkin sınıfın en az üçü (koroner kalsifikasyon, arteriyel kalsifikasyon,
> amfizem) doğrudan yaş-güdümlü, ve CT-RATE'te yaş **%100** mevcut. Yaşı taşıyan
> bağlam hattı bu yüzden yetişkin tarafında da *yeni sinyal* getiriyor — hem
> query koşullandırmasına (C1) hem de yönlendirme kapısına (C3). 0.8574'ü geçmenin
> gerçekçi yolu kayıp telafisi değil, bu. Indication ise CT-RATE'te yalnızca %49.8
> mevcut, o yüzden yetişkin kazancının çoğu yaştan gelmeli.

### Açık tam olarak nerede — ilk kez sınıf bazında

0.0132'yi 18 sınıfa dağıttık. Aşağıdaki tablo ham deltaları veriyor; hangilerinin
gürültüden ayrılabildiği **eşleştirilmiş** testle belirleniyor (bu bölümün sonu).

| sınıf | ARC-CT | C47harm | Δ | peds prev. | not |
|---|---|---|---|---|---|
| Mosaic attenuation pattern | 0.8316 | 0.7513 | −0.0803 | 15.3% | en büyük üç kayıp |
| Pulmonary fibrotic sequela | 0.7114 | 0.6623 | −0.0491 | 19.7% | en büyük üç kayıp |
| Lung nodule | 0.7593 | 0.7199 | −0.0394 | 52.8% | en büyük üç kayıp |
| Peribronchial thickening | 0.8328 | 0.8143 | −0.0185 | 15.1% |  |
| Bronchiectasis | 0.8097 | 0.7946 | −0.0151 | 13.0% |  |
| Hiatal hernia | 0.8363 | 0.8220 | −0.0143 | 0.9% | peds'te ~yok |
| Lung opacity | 0.8761 | 0.8641 | −0.0120 | 28.7% |  |
| Emphysema | 0.8261 | 0.8142 | −0.0119 | 0.3% | peds'te ~yok |
| Lymphadenopathy | 0.7871 | 0.7782 | −0.0089 | 7.9% |  |
| Atelectasis | 0.8060 | 0.7990 | −0.0070 | 32.6% |  |
| Cardiomegaly | 0.9445 | 0.9385 | −0.0060 | 2.7% |  |
| Medical material | 0.9236 | 0.9183 | −0.0053 | 39.8% |  |
| Arterial wall calcification | 0.9365 | 0.9313 | −0.0052 | 1.6% | peds'te ~yok |
| Consolidation | 0.9276 | 0.9250 | −0.0026 | 6.0% |  |
| Coronary artery wall calc. | 0.9321 | 0.9305 | −0.0016 | 0.3% | peds'te ~yok |
| Pleural effusion | 0.9737 | 0.9726 | −0.0011 | 4.3% |  |
| Pericardial effusion | 0.8869 | 0.9029 | +0.0160 | 4.6% | kazandı |
| Interlobular septal thickening | 0.8318 | 0.8564 | +0.0246 | 5.7% | kazandı |

*CT-RATE validation, n=3.002. ARC-CT sütunu yayınlanmış `augmd_seed0` koşusu (0.8574). Net: (−0.2783 + 0.0406)/18 = **−0.0132**.*

> **GERİ ÇEKİLDİ · rev.4'ün manşet bulgusu**
>
> > **GERİ ÇEKİLDİ · rev.4'ün manşet bulgusu**
> >
> > Rev.4 şunu iddia ediyordu: *"schema.py, 30→39 query cerrahisinde 5 satırı kasten
> > yeniden başlattı… bu 5 satır yetişkin kaybının %77'sini taşıyor."*
> > **Bu iddia yanlış ve geri çekiliyor.** Ölçtüm.
> >
> > `arcct_seed0.peds27.pt` ile `arcct_seed0.pt`'nin query
> > tensörlerini satır satır karşılaştırdım:
> >
> > ```
> > 18 yetişkin patoloji satırı  →  maxAbsDiff = 0.000e+00  (bit-özdeş)
> > 2 global satır               →  maxAbsDiff = 0.000e+00  (bit-özdeş)
> > anatomi satırı 0–6           →  maxAbsDiff = 0.000e+00  (bit-özdeş)
> > anatomi satırı 7, 8, 9       →  yeniden-init  # pleura(türetilmiş) · göğüs duvarı · üst batın
> > 9 yeni patoloji satırı       →  taze, std 0.01992  # kaynak tensörün std'si 0.019892
> > ```
> >
> > **Hiçbir patoloji satırı yeniden başlatılmamış.** Gerçekte yapılan cerrahi,
> > şemada *yeni olan* üç anatomik bölgeye taze satır vermek; anlamı hayatta kalan
> > her şey kopyalanmış. Beş isimlik listeyi hiç çalıştırılmamış bir plan dosyasından
> > almıştım; cerrahi kodu `schema.py`'de değil (orada tensör bile yok),
> > `arcct/arcct-bch/model/qformer_surgery.py`'de ve `--reinit` bir
> > komut satırı argümanı.
> >
> > Buna bağlı üç şey de düştü ve rev.5'te belgeden çıkarıldı: kaldıraç
> > **"yeniden-init'i geri al"** (geri alınacak bir şey yok), rev.4'ün **Faz 0
> > maddesi** (~15 GPU-saat; tablodan kaldırıldı, sıra numarası artık başka bir ölçüme
> > ait), ve §11'deki *"o listede olmalıydı"* retoriği.
>
> > **yerine geçen bulgu — daha dar ama sağlam**
> >
> > **18 yetişkin satırın hepsi ARC-CT ağırlığıyla bit-özdeş başladı ve yine de
> > 0.0132 kaybedildi.** Yani yetişkin açığı bir *başlangıç* olayı değil, bir
> > *eğitim* olayı: sıcak başlatma korumadı, pediatrik gradyan aşındırdı.
> >
> > Peki hangi deltalar gürültüden ayırt edilebiliyor? **Rev.5 bunu yanlış ölçmüştü**:
> > her deltayı tek kolun kendi güven aralığının yarı-genişliğiyle kıyaslamıştım, oysa
> > iki model *aynı 3.002 hacimde* ölçülüyor. Doğru araç **eşleştirilmiş**
> > bootstrap — örnekleme gürültüsü büyük ölçüde götürüyor ve aralık daralıyor.
> > Eşleştirilmiş, hasta-bazlı (1.304 hasta, 1.000 tekrar):
> >
> > | sınıf | Δ (C47harm − ARC-CT) | eşleştirilmiş %95 GA | sıfırı dışlıyor |
> > |---|---|---|---|
> > | Mosaic attenuation pattern | −0.0803 | [−0.1031, −0.0591] | **evet** |
> > | Pulmonary fibrotic sequela | −0.0491 | [−0.0657, −0.0331] | **evet** |
> > | Lung nodule | −0.0394 | [−0.0535, −0.0260] | **evet** |
> > | Peribronchial thickening | −0.0186 | [−0.0298, −0.0083] | evet |
> > | Bronchiectasis | −0.0151 | [−0.0271, −0.0030] | evet |
> > | Hiatal hernia | −0.0143 | [−0.0294, −0.0000] | sınırda |
> > | Lung opacity | −0.0120 | [−0.0176, −0.0062] | evet |
> > | Emphysema | −0.0119 | [−0.0231, −0.0013] | evet |
> > | Lymphadenopathy | −0.0090 | [−0.0169, −0.0005] | evet |
> > | Arterial wall calcification | −0.0052 | [−0.0089, −0.0014] | evet |
> > | Atelectasis · Cardiomegaly · Medical material · Consolidation · Coronary · Pleural effusion | −0.0070 … −0.0011 | sıfırı kapsıyor | hayır |
> > | Pericardial effusion | +0.0159 | [+0.0036, +0.0293] | evet (kazanç) |
> > | Interlobular septal thickening | +0.0246 | [+0.0077, +0.0402] | evet (kazanç) |
> >
> > *Referans: `results/adult_gate/augmd_seed0` (bizim e3 yeniden üretimimiz, makro-18 = **0.857398**). Diskte ikinci bir referans daha var — `arcct/reference`, orijinal Bilkent koşusu, **0.858346**; FINAL_REPORT ikisini ayrı ayrı kaydediyor (delta −0.0009). Kapı için kendi yeniden üretimimizi kullanıyoruz, çünkü aynı kod ve ortamda ölçüldüğü için fark modeli yansıtır.*
> >
> > **12/18 delta sıfırı dışlıyor, 10'u kayıp.** Çözülebilen kayıplar brüt negatif
> > toplamın (−0.2785) **−0.2549**'unu, yani **%91.5'ini** taşıyor. Yani pay
> > hesaplamak *meşru* — rev.5'in "hiçbir yüzde raporlanamaz" hükmü fazla
> > temkinliydi ve yanlış araçtan geliyordu.
> >
> > Ve anlamı yaşa bağlı üç sınıf (Mosaic, Fibrotic sequela, Lung nodule) tek başına
> > **−0.1688** taşıyor: brüt kaybın **%61'i**, net kaybın **%71'i** —
> > üçü de bireysel olarak anlamlı.
>
> > **C1 için ne kalıyor — ilişki, neden değil**
> >
> > Ölçülebilir kayıp taşıyan üç sınıf, anlamı **yaşa bağlı** olan üç sınıf. Mosaic:
> > erişkinde küçük hava yolu hastalığı / kronik PE, çocukta post-enfeksiyöz bronşiolitis
> > obliterans. Fibrotic sequela: erişkinde IPF, çocukta radyoterapi/kemoterapi sonrası
> > skar. Lung nodule: erişkinde insidental tarama nodülü, çocukta
> > *osteosarkom metastaz sürveyansı*.
> >
> > Bu, C1'in yaş-koşullu modülasyonu için **hâlâ iyi bir motivasyon** — ama artık
> > "yeniden-init hatasını düzeltiyoruz" değil, çok daha basit bir ifadeyle:
> > *ortak eğitim, anlamı kohortlar arasında ayrışan sınıflarda yetişkin becerisini
> > aşındırıyor; yaş bu ayrışmanın gözlemlenebilir ekseni.*
> >
> > **Nedensel değil, ilişkisel.** Üç sınıf aynı zamanda etiketleyici-eşiği uyuşmazlığı
> > listesinde de. Karıştırıcılar aşağıda.
>
> > **ayrıştırılmamış karıştırıcılar**
> >
> > - **Etiket harmonizasyonu aynı büyüklükte.** İki ortak koşu arasındaki en büyük sınıf hareketi `Interlobular septal thickening`: 0.7805 → 0.8564 (**+0.0759**) — ve o bir "yaşa bağlı anlam" sınıfı değil, bir eşik-uyuşmazlığı sınıfı. Mosaic'i de tek başına 0.0287 oynatıyor, yani −0.0803'ün **%32'si**.
> > - **Eğitim uzunluğu.** C47harm en iyi 4.800'de (8.800'de durdu), C47mix 16.800'de (20.800'de durdu) — 3.5 kat fark. İki koşunun makro-18 farkının eşleştirilmiş bootstrap'ı **+0.0041, %95 GA [0.0013, 0.0070]**: etiket + uzunluk tek başına anlamlı fark üretiyor.
> > - **Tavan etkisi.** ARC-CT tabanı düşük olan sınıflar daha çok kaybediyor. Bu üç sınıfın ARC-CT ortalaması 0.767, diğer 15'inki 0.876.
> > - **Paylaşılan gövde sürüklenmesi** — sınıfa özel değil.
> >
> > **Ayrıştıran deney:** {harmonize/yayınlanmış etiket} × {sabit adım bütçesi} 2×2,
> > hücre başına ≥2 tohum — 8 koşu, ~120 GPU-saat. Erken durdurma *kapalı*, yoksa
> > uzunluk yine karışır.

| # | kaldıraç | ne yapar | maliyet | beklenen etki |
|---|---|---|---|---|
| 1 | Sıfır-init her yerde | C1–C4 adım 0'da etkisiz → eğitim yayınlanmış ağırlıklardan başlar | yapısal | açığın kaynağını ortadan kaldırır |
| 2 | Parametre grupları | 18 yetişkin query satırı + text LoRA 0.1× LR; yeni 9 satır + bağlam modülleri tam LR | bir config | orta–yüksek |
| 3 | Seçim kuralı | `min(adult18, peds23)` üzerinden seç — ama **yalnızca validation'da**; bu adult-18'i seçilen bir büyüklük yaptığı için kapı **test bölünmesinde** ölçülür (§17) | bir fonksiyon + test split | orta |
| 4 | Kohort-başına kayıp maskesi | nadir yetişkin sınıflarının pediatrik gradyanını keser (Arterial, Coronary, Emphysema, Hiatal) | düşük | **küçük — en fazla ~0.0018, dördü de GA içinde**; gerekçe hijyen, kapı değil |
| 5 | Batch kompozisyonu | sabit peds:adult oranı (ör. 1:3), doğal 14:86 yerine | düşük | orta |
| 6 | CT-RATE train maskeleri | yetişkin yarıda da yönlendirme açılır (şu an %86 maskesiz koşuyor) | **~1.740 GPU-saat** | bilinmiyor — önce validation kanıtı |
| 7 | Ağırlık ortalaması (soup) | uyarlanmış model ile `arcct_seed0.pt`'yi harmanla | dakikalar | düşük ama neredeyse bedava |

> **kapı bugün sağlanmıyor — ve "gürültü bandı içinde" savunması geçersiz**
>
> Hasta-bazlı bootstrap (1.304 CT-RATE hastası, 1.000 tekrar):
>
> ```
> # EŞLEŞTİRİLMİŞ hasta-bazlı bootstrap · 3.002 hacim · 1.304 hasta · 1.000 tekrar
> ARC-CT (augmd_seed0) makro-18 = 0.8574   # FINAL_REPORT ile birebir yeniden üretildi
>
> C47harm  = 0.8442    fark = +0.0132   %90 GA [0.0110, 0.0154]
> C47mix   = 0.8400    fark = +0.0174   %90 GA [0.0145, 0.0203]
> P(bizimki ≥ ARC-CT) = 0.000
> ```
>
> **Bu artık eşleştirilmiş bir test.** ARC-CT'nin hacim-başına tahminleri
> `results/adult_gate/augmd_seed0/predictions.npz`'de zaten duruyordu
> (18 Ağustos), aynı 3.002 hacim ve aynı sırada — yani ayrıca bir koşu gerekmedi.
> Eşleştirme aralığı üçte bire indirdi (yarı-genişlik 0.0061 → **0.0022**), çünkü
> örnekleme gürültüsü iki modelde de aynı hacimlerden geliyor ve büyük ölçüde götürüyor.
> Açık artık kesin biliniyor: **0.0132 ± 0.0022**.
>
> Rev.4 burada *"fark ≤ tohum bandı ise ayırt edilemez"* diyordu. Bu üç ayrı
> hatayı birden yapıyor: **(a)** örtüşen güven aralığı bir *eşdeğerlik testi
> değildir* — farkı reddedememek, eşdeğerliğin kanıtı değil; **(b)** marj
> veriyi gördükten sonra seçiliyor; **(c)** referansın kendi aralığı yokmuş gibi davranıyordu — oysa tahminler diskteymiş ve eşleştirilmiş test bugün mümkün (yukarıda).
>
> **Doğru kriter — önceden sabitlenmiş TOST:** non-regresyon ancak eşleştirilmiş
> farkın %90 GA üst sınırı, koşulardan *önce* ilan edilmiş Δ marjının altında
> kalırsa ilan edilir. Ve marj seçimi burada **belirleyici**: bugünkü üst sınır
> **0.0154**, yani Δ = 0.01 ile **eşdeğer değil**, Δ = 0.02 ile **eşdeğer**.
> İki sonuç arasındaki tek fark marjın kendisi — bu yüzden klinik ya da literatür
> gerekçesiyle savunulan bir Δ, sonuçlar görülmeden sabitlenmeli.
> **Bugünkü durum: Δ = 0.01'de kapı sağlanmıyor.**

`14 · zorunlu`

## Ablasyon merdiveni

Notlarda anılan çalışma, 3B göğüs BT'de **basit concatenation'ın cross-attention'ı
yendiğini** bulmuş ve bunu veri yetersizliğine bağlamış. Pediatrik eğitim setimiz
7.042 hacim (8.816'lık kohortun eğitim bölümü) — ve yetişkin yarısı bunu telafi
etmiyor, çünkü CT-RATE'te indication %49.8 mevcut ve medyanı 2 kelime.
Merdiven opsiyonel değil.

- **Yalnızca BT**taban · bugünkü C47 çizgisi
- **BT + concat(indication)**en ucuz füzyon · notlardaki çalışmada kazanan · bunu geçemezsek cross-attention gerekçesiz
- **+ cross-attention koşullandırma (C1, FiLM'siz)**uzamsal soru · token dizisini kullanan ilk basamak
- **+ FiLM kanal kapılaması (C1 tam)**"hangi öznitelik türü" · iki seviyeli koşullandırma tamamlanıyor
- **+ alaka ağırlıklandırma (C2)**w = 1 + βr · havuz ve kayıp birlikte
- **füzyon: concat / kapılı toplam**Z = W[Z_gen;Z_ind] vs Z_gen + g⊙Z_ind
- **izolasyon: maskeli / paylaşımlı self-attention**sızıntının doğrudan ölçümü · paylaşımlı sürüm daha iyi skor verirse sebebi sızıntıdır
- **yaş: bant / skaler / düz tablo / karıştırılmış**ordinal-kümülatif parametrelemenin gerekçesi

> **Faz 2'ye ertelenen kollar**
>
> C3 (öğrenilen λ kapısı) · C4 (serbest indication query'leri) · peds mask-free /
> anatomi query'siz · kırpma × kapı 2×2. Dördü de gerçek kol olarak duruyor, ama
> Faz 1'in teslim listesinde değil — gerekçeleri §07 ve §08'de.
>
> **Sıralama, iptal değil.** Dördü de Faz 1'in *ölçümü* kapandığı anda kuyruğa girebilir;
> GPU kısıt değil, kollar birbirinden bağımsız ve paralel koşulabilir. Faz 1 çıtayı
> geçerse C3/C4 onun checkpoint'inden sıcak başlatılır. Geçemezse aynı kollar kurtarma
> kolu olarak koşulur — pediatrik AUC'yi yükseltmeye en yakın aday C3, çünkü boş-bölge
> uçurumundan en çok etkilenen kohort peds (§07). Bu durumda maskesiz-çıkarım kaybı
> manşet metriğin yanında açıkça raporlanır.

### Dört değerlendirme koşulu

| girdi | amaç | önceden ilan edilen beklenti |
|---|---|---|
| yalnızca BT | taban | mevcut performans |
| BT + doğru indication | beklenen klinik kullanım | en yüksek |
| BT + indication yok | sağlamlık | tabana yakın, altında değil |
| BT + **uyumsuz** indication | **indication yanlılığı testi** | nodül tahmini **kaybolmamalı** — kaybolursa koşullandırma çok güçlü |

*Dördüncü satır mekanizmanın varlık sebebini test eder. Notlardaki örnek: BT'de pulmoner nodül var, indication *"rule out pneumonia"* diyor. Nodül tahmini ayakta kalmalı.*

> **rev.5 · istatistiksel zemin baştan yazıldı**
>
> **0.0028 bir karar eşiği olarak kullanılamaz.** Tek bir tohum çiftinden geliyor:
> σ̂ = |d|/√2 = 0.0020, **1 serbestlik derecesiyle**. 1 sd'lik bir varyans tahmininin
> çift-taraflı %95 aralığının üst ucu **0.063** (tek-taraflı %95 sınırı 0.032) — yani nokta tahminin 32 katı. Dahası, bandı
> raporlayan satırın kendisi onu çürütüyor: aynı sütunda V1 = 0.8005 ve V1-seed1 = 0.7933
> duruyor, okuyucu **0.0072** hesaplıyor. (İkisi farklı etiket setlerinde ölçülmüş —
> band 0.7961 vs 0.7933'ten geliyor — ama tabloda öyle görünmüyor.)
>
> **Ve asla sınıf bazında kullanılamaz.** Ölçülen sınıf-başına tohum bantları
> 0.0001–0.0500 arasında. Rev.4'ün §10'daki *"Lung nodule −0.0080, kötüleşti"*
> çıkarımı, o sınıfın **kendi bandı 0.0118** olduğu için zaten yorumlanamazdı.
>
> **Rev.5 kuralı:**
>
> - Ablasyon başlamadan önce **tek bir konfigürasyonda 3 tohum** koşulur ve σ kullanılabilir serbestlik derecesiyle ölçülür.
> - Her sayı **hasta-bazlı bootstrap GA**'sıyla verilir; sınıf-başına farklar **sınıf-başına GA**'ya karşı yargılanır, ortalama-seviyesi banda karşı değil.
> - Güç: Δ = 0.005'i α = 0.05'te %80 güçle çözmek σ = 0.002 ise kol başına **3 tohum**, σ = 0.005 ise **16 tohum** gerektiriyor; 33 makro karşılaştırmaya Bonferroni eklenince 6 ve 32. **Hangi σ'nın doğru olduğunu bilmiyoruz** — bu da 3-tohum ön ölçümünün gerekçesi.
> - **Çokluluk:** 8 basamak × 3 kohort = **24** makro karşılaştırma → Holm–Bonferroni; 8 × 27 = **216** sınıf karşılaştırması → Benjamini–Hochberg FDR (q = 0.10). Önceden kayıtlı tahminler ayrı ve düzeltmesiz analiz edilir.
> - **Bütçe:** 3 tohum × 10 konfigürasyon ≈ **450 GPU-saat** (5 tohumda 750). Rev.7'de C3/C4 Faz 2'ye alınınca 675'ten düştü. Rev.4 bunun için hiçbir şey ayırmıyordu — oysa "çok pahalı" diye ertelenen CT-RATE maskelerinin (1.740) **%26'sı** kadar. **Ama GPU bu projede kısıt değil** ve kollar çok-GPU paralel koşulabiliyor: 5 tohuma çıkmak duvar saatini değil yalnızca kuyruğu büyütür. Yani 3 tohum bir bütçe tavizi değil, σ ön ölçümünün sonucudur; σ = 0.005 çıkarsa 16 tohuma çıkmanın önünde kaynak engeli yok.
> - **%9.19 yanlış artefaktın rakamı:** o *bölge* cache'inde ölçüldü (88.170 hacim-bölge çifti), 27-sınıf patoloji etiketlerinde değil. Patoloji etiketlerinin koşudan koşuya kararsızlığı **hiç ölçülmedi** → Faz 0'a eklendi.

`15 · manşet görsel`

## Kilit deney: aynı BT, üç indication

Notlarda "much more interesting contribution" olarak işaretlenen deney. ARC-CT
paper'ı zaten Grad-CAM'in CT-CLIP'ten daha odaklı olduğunu gösteriyordu; bu onun
dinamik hâli.

> **[ŞEKİL 4]** **Şekil 4 — Önerilen deney (henüz koşulmadı).** Görüntü değişmiyor; yalnızca klinik soru değişiyor. Koşullu patoloji query'lerinin dikkat kütlesi soruya göre yeniden dağılıyor, ama koşulsuz temsil sabit kalıyor. Bu tek figür hem mekanizmanın çalıştığını hem de güvenli olduğunu aynı anda gösterir. `Z_gen`'in üç koşuldaki eşitliği burada bir *umut* değil, §04'teki izolasyonun yapısal sonucu — deney onu doğrulamıyor, **ihlal edilmediğini denetliyor**. Asıl ölçülen şey sınıf sıralamasının anlamlı biçimde değişmesi.

`16 · neyin ters gidebileceği`

## Riskler ve kontroller

> **notlardan · en ciddiye alınacak risk**
>
> > *The model could learn: indication says 'PE' → predict PE, rather than actually finding an embolus. This is probably the biggest danger of this approach.*

| risk | durum | kontrol |
|---|---|---|
| Indication kısayolu (multimodal shortcut) | `[en büyük tehlike]` | **Dört katmanlı:** %30 dropout · uyumsuz-indication testi · karşıolgusal tutarlılık kaybı (**tahmin logitleri üzerinde**, §12) · görüntüsüz indication-only taban. Dördü birden raporlanır. |
| Indication bir bulguyu *bastırması* | `[mimariyle engellendi]` | rezidüel koşullandırma (q̃ = q + Δ) · w ≥ 1 · koşulsuz banka her zaman açık |
| Indication → etiket sızıntısı (metin örtüşmesi) | `[ölçüldü]` 46 hacimde indication anahtar kelimesi (%0.52); `LEAKAGE_EXCLUDE.txt` güvenlik payıyla **54** satır (46 indication + 9 yaş, 1'i her ikisi) | 54 hacim **tüm kolların** validation'ından dışlanır (yalnızca indication kolundan değil — yoksa paydalar eşleşmez) |
| Indication → etiket sızıntısı (semantik) | `[açık]` — 27 tanım pediatrik rapor okunarak yazıldı | karıştırma testi · anahtar kelime taraması bunu göremez |
| Kohort bayrağı sızıntısı | `[tasarımla engellendi]` | modele asla site/kohort göstergesi verilmez — yalnızca yaş |
| Indication metninde PHI | `[açık]` — medyan 21 kelime, isim/tarih içerir | modele ulaşmadan temizlik + IRB · prosedür belgelenir |
| Etiket boru hattı deterministik değil | `[kısmen]` %9.19 **bölge cache'inde**; patoloji etiketleri ölçülmedi (Faz 0/9) | etiket anlık görüntüleri dondurulur · tohum kontrolü zorunlu |
| Çift banka maliyeti | `[kabul edilmiş]` ~1.82× Q-Former dikkati | çift banka yeni query parametresi getirmiyor (ağırlık paylaşımı); C4 + klinik-soru query'si 5 yeni satır · maliyet raporlanır |

`17 · sıra`

## Fazlı plan

Faz 0'ın tamamı model kodu yazılmadan yapılır. 1–2 "maskesiz peds" kararını,
3 C1'in nereye bağlanacağını belirliyor. **5. madde koşuldu ve tahmini çürüttü**
(§10); 8–9 istatistiksel zemini kuruyor.

| # | ölçüm | gereken | süre | neyi karara bağlar |
|---|---|---|---|---|
| 1 | Yetişkinde routing cezası | **3.000 maske zaten diskte** — eval betiğinde bir satır `/nonexistent` | ~30 dk | ceza pediatriğe mi özgü · **maskesiz peds kararı** |
| 2 | Ceza ↔ checkpoint adımı | her 400/800 adımda kayıtlı ara checkpoint'ler | ~1 sa | seçim yanlılığı elenir mi · **maskesiz peds kararı** |
| 3 | R2 okuma başlığı | mevcut 9 checkpoint | ~1 sa | bedava kazanç var mı · **C1 nereye bağlanacak** |
| 4 | Yaş × bölge survival kırılımı | 8.816 maske + yaş metadata | ~1 sa | C3'ün açılış figürü · λ tahmininin gerekçesi |
| 5 | Dolgu oranı × sınıf AUC | mevcut checkpoint'ler | bitti | **tahmin çürüdü** (§10) — kırpma Faz 2'de kaldı, beklenen kazanç düştü |
| 6 | Indication-only taban (görüntüsüz) | metin + yaş | ~4 sa | tüm indication kolunun anlamlılık zemini |
| 7 | Raporlarda "interval change" dili | tek çıkarım geçişi | ~1 gün | boylamsal kolu (4.813 çift) tek başına karara bağlar |
| 8 | 3 tohum, tek konfigürasyon | mevcut peds fine-tune reçetesi | ~45 sa GPU | **σ'yı kullanılabilir serbestlik derecesiyle ölç** — merdivenin kaç tohum gerektirdiği buna bağlı |
| 9 | Patoloji etiketi yeniden çıkarımı | aynı prompt, ikinci geçiş | ~3 sa GPU | 27-sınıf etiketlerin koşudan koşuya kararsızlığı (%9.19 bölge cache'inin rakamı, bunun değil) |

*1, 2 ve 3 birlikte ~3 saat ve "hard masking zarar veriyor" cümlesinin yazıya girip giremeyeceğini belirliyor.*

### Faz 1 · minimum yayınlanabilir paper

- **C1 + C2** tam (cross-attn + FiLM + alaka), çift banka, concat füzyon.
- **Boş-bölge uçurumu düzeltmesi** (hata düzeltmesi, C3'ten bağımsız): ρ override'dan önce hesaplanır ve boş-bölge oranı loglanır.
- Yaş bandı + cinsiyet, bağlam token'ı olarak; `L_ind` + `L_loc`.
- **Koşulsuz yolun izolasyonu**: self-attn grup maskesi, λ ayrımı, grup-kısıtlı havuzlama, füzyon W = [I ; 0] — ve eğitim öncesi bit-özdeşlik assertion'ı.
- **Üçüncü, hiç dokunulmamış test bölünmesi** (hasta-bazlı). Seçim ve tüm ablasyon sıralaması validation'da; manşet pediatrik ve yetişkin sayıları konfigürasyon donduktan sonra **bir kez**, test bölünmesinde. Bugün seçim metriği ile raporlanan metrik aynı — ~30–60 checkpoint (bağımsızlık varsayımıyla üst sınır) üzerinden maksimum alındığı için iyimserlik **~0.0065** mertebesinde, yani tartışılan yetişkin açığının yarısı.
- **Radyolog adjudikasyonu**: ~200 pediatrik validation hacmi, iki pediatrik radyolog, 27 sınıf. Bugün *hiçbir* etiket insan tarafından doğrulanmadı; üstelik etiketler ve kontrastif metin hedefi **aynı rapordan** türediği için gürültüleri korele ve AUC'ler bağımsız bir referansa göre iyimser.
- **Yaş-tabakalı raporlama**: her pediatrik sayı üç kez — havuzlanmış, <18 (n=1.280), ≥18 (n=492). Manşette <18 kullanılır. Yoksa "çocuklarda işe yarıyor" ile "çocuk hastanesinin yetişkin çeyreğinde işe yarıyor" ayrılamaz.
- Kohort-başına kayıp maskesi; parametre grupları; `min(adult18, peds23)` seçimi.
- Üç kohortlu değerlendirme + dört koşul + sekiz basamaklı merdiven.

### Faz 2 · güçlendirme

- **C3 — öğrenilen λ kapısı**, Faz 0/1–3 gerekçeyi doğrularsa ve çıkarımda maske sorunu çözülürse (§07).
- **C4 — 4 serbest indication query'si**, uyumsuz-indication testi C1+C2 üzerinde temiz geçerse (§08).
- **Boylamsal delta query'leri** — 4.813 çift, 1.390 hasta. Faz 0/7 desteklerse.
- **Yapılandırılmış indication kavramları** (semptom / klinik bağlam / hedef durum / anatomi) ayrı query token'ları olarak — notlardaki "multiple queries can independently interrogate the CT" fikri.
- **Gövde-normalize kırpma** (Dr. Kurugol direktifi, §10) — Faz 0/5 sırasını belirler, kararı değil. uygulama (B): kutu = 6 tamsayı/hacim, ~0 disk (§10).
- CT-RATE train maskeleri (~1.740 GPU-saat) — *ancak* Faz 0/1–2 haklı çıkarırsa.
- Pediatrik benchmark protokolünün yayımı: veri PHI, o yüzden şema + prompt + protokol + model yayımlanır.

`18 · karar`

## Açık kararlar

Notlar önceki sekiz sorunun beşini kapattı. Kalanlar ve yeni açılanlar:

- **Faz 0 önce mi?** Üç saatlik üç ölçüm, hem "maskesiz peds" kararını hem de paper'ın açılış iddiasını belirliyor. Güçlü tavsiyem: evet.
- **Çift banka + izolasyon maliyetini kabul ediyor muyuz?** (Faz 1'de 1.72×, C4 gelirse 1.82×.) Koşulsuz patoloji yolu, notlardaki "unconditional pathway" şartının en katı yorumu: **1.82×** Q-Former çapraz-dikkati, artı `QFormerBlock`'a self-attention maskesi parametresi ve λ'nın ikiye ayrılması. Daha ucuz alternatif tek banka + rezidüel koşullandırma + w ≥ 1 — ama o zaman koruma iki katmana iner ve `Z_gen`'in indication'dan bağımsızlığı *yapısal* olmaktan çıkar. Benim tercihim çift banka: §15'teki manşet deneyin ölçülebilir iddiası (`Z_gen` üç koşulda özdeş) tek bankayla kurulamaz, ve hakem ilk oraya bakar.
- **Pnömotoraks:** bölge-8 düzeltmesiyle kalsın ve C3'ün ilan edilmiş test vakası olsun mu? (Modelden çıkarmayı önermiyorum; gerekçe §11.)
- **Arterial wall calcification:** senin önerin doğru, karar değişti — peds'te kayıp maskeli, peds skorunda yok. Çocuktaki pozitifler (ligamentum arteriosum, kateter izi, kıkırdak, ven) yetişkindeki hastalık değil. Tek düzeltme beklentide: ölçtük, yetişkin kazancı **−0.0052** kadar, yani kapıyı bu kapatmayacak. Tanım düzeltmesinin 0.654→0.735 kazancı yine de raporlanır — o bir *etiket kalitesi* bulgusu, eğitim kararı değil.
- **Yetişkin kaybının sebebi ne, madem yeniden-init değil?** Bu soru rev.5'te *açıldı*, kapanmadı: 18 satır da bit-özdeş başladı ve yine de kaybedildi. Ölçülebilir kayıp taşıyan üç sınıf (Mosaic, Fibrotic sequela, Lung nodule) aynı anda hem "anlamı yaşa bağlı" hem de "etiketleyici eşiği uyuşmuyor" listesinde — ve etiket harmonizasyonu tek başına Mosaic'i 0.0287 (−0.0803'ün %32'si), Interlobular'ı 0.0759 oynatıyor. **Ayrıştıran deney:** {harmonize/yayınlanmış etiket} × {sabit adım bütçesi} 2×2, hücre başına ≥2 tohum, erken durdurma kapalı — 8 koşu, ~120 GPU-saat. Bunu Faz 1'den önce koşturuyor muyuz?
- **Pediatrik manşet metrik "27" mi "ölçülebilir 23" mü?** Dört sınıf düşünce (Emphysema, Coronary, Hiatal hernia, Arterial wall calc) 27 → 23 oluyor. V2 için: 27-sınıf 0.7974 · 23-sınıf **0.7944**. Fark bir tohum bandı kadar. **Karar kuralı sayı değil hastalık kimliği olmalı** ve sonuçlar görülmeden şimdi sabitlenmeli.
- **CT-RATE train maskeleri** (~1.740 GPU-saat) — Faz 0/1'in sonucuna bağlansın mı, yoksa şimdi taahhüt mü?
- **IRB + PHI temizliği** indication metni için — kim, ne zaman? C1'in önündeki tek gerçek dış bağımlılık.
- **Kaynak doğrulaması.** Notlarda **iki** çalışma anılıyor: Dia-LLaMA (2025, disease-aware attention) ve 3B göğüs BT'de concat > cross-attention bulan isimsiz çalışma. **Di Piazza ve ark. notlarda geçmiyor** — o kendi `CONTEXT_ROUTED_CT_PROPOSAL.md`'mizden geliyor ve rev.4'te yanlışlıkla notlara atfedilmişti.

Kaynaklar: `reports/Dr_Kurugols_notes.txt` (25 KB, 26 Ağustos) ·
`reports/IMPORTANT.md` · `reports/ARC_CT_PEDIATRIC_REPORT.md` ·
`reports/CONTEXT_ROUTED_CT_PROPOSAL.md` · `arc-ct@9c04648` +
yerel değişiklikler · `eval_matrix/` (V1–V5, C16, C47mix, C47harm) ·
25 Ağustos 2026 tarihli 8 değerlendirme koşusu. Kod okuması: `anatomy_qformer.py`,
`qformer.py`, `train_stage2.py`, `evaluate.py`, `schema.py`.

**Rev. 2** — Dr. Kurugol'un notları alındıktan sonra yeniden yazıldı: çift
patoloji bankası, token-dizisi cross-attention, alaka kapısı, serbest indication
query'leri, bölünmüş global query'ler, ablasyon merdiveni, çözünürlük bölümü.
Düşen: yaş-koşullu prevalans önselı (opsiyonele indirildi).

**Rev. 3** — tasarım denetiminde bulunan dört kusur kapatıldı:
(i) self-attention koşulsuz bankaya sızıyordu → grup maskesi;
(ii) λ kapısı koşulsuz query'lerin dikkat geometrisini indication'a bağlıyordu →
λ^gen / λ^ind ayrımı;
(iii) füzyon `W` rastgele başlıyordu → [I ; 0];
(iv) `r_c`'nin nokta-çarpımı biçimi donuk kulelerle eğitilemezdi → MLP.
Ayrıca havuz normalize edildi (Σw'ye bölme), β öğrenilen ve 0-init yapıldı, ve
karşıolgusal tutarlılık *kaybı* kaldırıldı — izolasyon onu gereksiz kılıyor.

**Rev. 4** — yetişkin açığı ilk kez sınıf bazında ölçüldü; *Arterial wall
calcification* pediatrik eğitimden ve skordan çıkarıldı; manşet metrik
"ölçülebilir 23" oldu.

**Rev. 5** — altı kollu bağımsız denetim (sayılar · kod iddiaları · mimari ·
iç tutarlılık · notlara sadakat · istatistik) sonrası. **İki iddia geri çekildi:**
(1) §13'ün "yeniden başlatılmış 5 query satırı" nedenselliği — checkpoint ölçümü
hiçbir patoloji satırının yeniden başlatılmadığını gösterdi; (2) §10'un
"Lung nodule kötüleşti" iddiası — işaret tersti, doğru değer +0.0021.
**Kapatılan tasarım hataları:** β'nın kaçak çözümü (softplus), boş-bölge
muhafızının λ'yı etkisizleştirmesi, ρ'nun ters okunması, **dördüncü sızıntı yolu
(havuzlama)**, H_dem boyut uyuşmazlığı, `L_loc`'un C1'e karşı çalışması,
`L_ind`'in hangi bankayı denetlediği, γ'nın query'yi silebilmesi.
**Geri getirilen:** karşıolgusal tutarlılık kaybı — ama `Z_gen`'e değil
tahmin logitlerine. **Eklenen:** TOST eşdeğerlik kriteri, raporlanabilirlik tabanı,
test bölünmesi, radyolog adjudikasyonu, yaş-tabakalı raporlama, güç/çokluluk/bütçe.

**Rev. 5 sonrası ikinci tur denetim** (sayılar + tutarlılık, rev.5 üzerinde).
Beş *retraction kalıntısı* kapatıldı — belge kendi geri çekmesini beş yerde
çürütüyordu. **Üçüncü bir düzeltme:** rev.5 sınıf-başına deltaları tek kolun
marjinal aralığıyla kıyaslamış ve "15/18 gürültünün içinde, pay raporlanamaz"
demişti; iki model aynı hacimlerde ölçüldüğü için doğru araç **eşleştirilmiş**
bootstrap ve orada **12/18 sıfırı dışlıyor**. Pay hesabı meşru: anlamı yaşa bağlı
üç sınıf brüt kaybın **%61'ini** taşıyor. Ayrıca: ARC-CT tahminlerinin
*zaten diskte olduğu* bulundu (Faz 0 maddesi kaldırıldı), eşleştirilmiş fark
**+0.0132 [0.0110, 0.0154]** ölçüldü, kırpma × sınıf-AUC testi koşuldu ve
**çürüdü**, ve ~12 sayısal düzeltme yapıldı.

**Rev. 6** — belge baştan sona okundu ve **24 tutarsızlık** kapatıldı. En
ciddileri: §13 hâlâ "referansın aralığı yok" diyordu (oysa iki paragraf üstünde
tahminlerin diskte olduğu yazıyor); §10'da çürütülen hipotezin eski paragrafı
çürütmenin *altında* duruyordu; §05 "maske kısıtı AYNEN duruyor" derken §07
onu yumuşatıyordu; §03'ün H2 hipotezi yanlıştı (seçim maskesiz değil,
oracle-maskeli); §02 tablosunun dört satırı gövdeyle çelişiyordu; bölge-survival
sayıları terk edilmiş 13-bölge şemasındandı (%10/%3 → **%5.6/%1.5**).

**Rev. 7** — kapsam daraltıldı. **C1 + C2 Faz 1'in tamamı**; C3 ve C4 opsiyonel
kola alındı. Gerekçe: C3'ün motivasyonu §03 Gerçek 2 ile karışmış bir ölçüme dayanıyor
*ve* ρ üzerinden çıkarımda maske gerektirerek ARC-CT'nin maskesiz-çıkarım
özelliğini kırma riski taşıyor; C4 ise tasarımdaki en büyük kısayol yüzeyi ve
"indication yoksa katkısı sıfır" iddiası mekanize değil. Boş-bölge uçurumunun
kapatılması C3'ten ayrılıp **Faz 1 hata düzeltmesi** oldu. Sonuç: merdiven
12 → **8**, makro karşılaştırma 36 → **24**, bütçe 675 → **~450 GPU-saat**,
query bankası 71 → **67 slot**, maliyet 1.82× → **1.72×**.
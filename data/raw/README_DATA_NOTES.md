# Veri Notları — MT IVANI Noon Report (Ocak–Ağustos 2021)

Bu klasördeki veriler, kullanıcı tarafından yüklenen 8 adet Noon Report görüntüsünden
(1.png … 8.png, aynı içeriğin PDF/Pages hali) **elle, dikkatle transkribe edilerek**
oluşturulmuştur. Hiçbir sayı veya olay uydurulmamıştır.

## Kapsam ve bilinen sınırlamalar

1. **`mt_ivani_noon_reports.csv`** — Her gün için ROB (FW/BBM/ME Oil/AE Oil) ve
   Consumption (FW/BBM/ME Oil/AE Oil) değerlerini içerir. Bu değerler tablonun
   açıkça basılı/yazılı hücrelerinden okunmuştur, yorum veya tahmin içermez.
   - Ağustos ayı 19'dan sonrası (20-31) görüntüde boş bırakılmış (kapı verisi yok) —
     dahil edilmemiştir.
   - Aylık TOTAL satırları (görüntülerdeki toplam satırları) dahil edilmemiştir;
     bu toplamlar gerektiğinde veritabanı sorgusuyla hesaplanabilir.

2. **`mt_ivani_events.csv`** — Serbest metin notlardan (LO değişimi, yağ değişimi,
   tahliye/yükleme, FW ikmali, mürettebat değişimi, karaya oturma, onarım, vb.)
   RAG için anlamlı olan olaylar transkribe edilmiştir.
   - **Bu liste TAM/eksiksiz değildir.** Görüntülerde her gün için tekrar eden,
     düşük bilgi değerli "before/after discharge" okumaları gibi rutin notların
     bir kısmı, kapsamı makul tutmak amacıyla atlanmıştır. Operasyonel açıdan
     anlamlı olaylar (bakım, arıza, mürettebat, sefer, önemli ikmal olayları)
     önceliklendirilmiştir.
   - Bazı notlar görüntünün sağ kenarında **kesilmiş** durumda (örn. saat bilgisi
     veya ROB detayları). Bu durumlarda not, görünen kısmıyla aktarılmış ve
     "(görüntüde kesik)" ifadesiyle işaretlenmiştir — eksik kısım UYDURULMAMIŞTIR.
   - Sarı ile vurgulanmış tek başına duran sayılar (örn. "3987", "4105") görüntüde
     hangi işlemi temsil ettiği açıkça yazmıyor; muhtemelen bir BBM sounding/mutabakat
     okuması olduğu değerlendirilmiş ancak **bu yorum kesin değildir** ve bu satırlar
     ingestion'a dahil edilmemiştir (yanlış bir anlam atfetmemek için). Gerçek anlamı
     netleşirse `mt_ivani_events.csv`'ye eklenebilir.

3. **`mt_ivani_monthly_fw_summary.csv`** — Her ay için mavi kutuda gösterilen
   "Pemakaian Fresh Water" (Fresh Water tüketim) özet kutusundan alınmıştır.

## Test/demo verisi DEĞİLDİR

Bu veriler MT IVANI gemisine ait gerçek Noon Report kayıtlarından alınmıştır — uydurma
veya sentetik veri değildir. Ancak elle transkripsiyon süreci hataya açık olabileceğinden,
özellikle kritik kararlar için orijinal görüntülerle çapraz kontrol önerilir.

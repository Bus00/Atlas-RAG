# ATLAS- RAG (V1)

MT IVANI gemisinin Noon Report kayıtları üzerinde çalışan, **tamamen lokal**
(kurulumdan sonra internetsiz) bir RAG (Retrieval-Augmented Generation)
sohbet asistanı.

```
Noon Report (CSV) → PostgreSQL → chunk'lama → embedding (bge-m3) → ChromaDB
                                                                        ↓
                              Streamlit  ←  Ollama (qwen2.5:7b)  ←  retrieval
```

Bu README, sıfırdan çalışan bir V1'e ulaşmak için gereken **tüm adımları**
sırasıyla anlatır. Adımlar Mac (Apple Silicon, macOS) için yazılmıştır.

---

## 0) Veri hakkında önemli not

`data/raw/` altındaki CSV dosyaları, yüklediğiniz 8 Noon Report görüntüsünden
elle transkribe edilmiş **gerçek veridir** — uydurma değildir. Kapsam ve
bilinen sınırlamalar için `data/raw/README_DATA_NOTES.md` dosyasına bakın
(özellikle: bazı rutin notlar atlanmıştır, bazı notlar görüntüde kesik).

---

## 1) Python ortamı

```
YAP:
cd maritime-rag
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

BEKLENEN:
Hatasız kurulum biter. (sentence-transformers ile birlikte torch da inecek,
birkaç GB indirme olabilir — internet gerekir, bu adım tek seferliktir.)
```

Eğer belirli sürüm numaraları sizin ortamınızda bulunamazsa, `requirements.txt`
içindeki `==X.Y.Z` kısımlarını silip tekrar deneyin (`pip install -r requirements.txt`).

---

## 2) PostgreSQL kurulumu

Zaten kurulu bir PostgreSQL'iniz yoksa:

```
YAP:
brew install postgresql@16
brew services start postgresql@16
createdb maritime_rag

BEKLENEN:
Hata almadan "maritime_rag" adında boş bir veritabanı oluşur.
```

`.env` dosyasını oluşturun ve gerekirse kullanıcı adı/şifreyi güncelleyin:

```
YAP:
cp .env.example .env
# .env içindeki PG_USER / PG_PASSWORD değerlerini kendi kurulumunuza göre düzenleyin
# (Homebrew ile kurulumda genellikle PG_USER=<mac kullanıcı adınız>, şifre boş olabilir)

BEKLENEN:
.env dosyası proje kök dizininde oluşur.
```

---

## 3) Veritabanı migration'larını çalıştırma

```
YAP:
python -m database.run_migrations

BEKLENEN:
"[OK] Migration çalıştırıldı: .../01_lookup_tables.sql"
"[OK] Migration çalıştırıldı: .../02_core_entities.sql"
"[OK] Migration çalıştırıldı: .../03_seed_reference_data.sql"
"Tüm migration'lar tamamlandı."
```

Doğrulama (opsiyonel):

```
YAP:
psql maritime_rag -c "SELECT name FROM vessel;"

BEKLENEN:
 name
----------
 MT IVANI
```

---

## 4) Noon Report verisini PostgreSQL'e yükleme (ingestion)

```
YAP:
python -m ingestion.load_noon_reports

BEKLENEN:
"[OK] 231 noon_report kaydı yazıldı/güncellendi."
"[OK] ~60 noon_report_event kaydı yazıldı (... atlandı)."
"[OK] 8 aylık özet chunk'ı yazıldı."
"[TAMAM] Ingestion (Postgres yazma aşaması) tamamlandı."
```

---

## 5) Ollama kurulumu ve model indirme

```
YAP:
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b-instruct

BEKLENEN:
Model indirilir (~4-5 GB, internet gerekir, tek seferlik).
"ollama list" komutuyla modelin listede göründüğünü doğrulayabilirsiniz.
```

Test:

```
YAP:
ollama run qwen2.5:7b-instruct "merhaba, çalışıyor musun?"

BEKLENEN:
Modelden kısa bir Türkçe/İngilizce cevap gelir.
```

> Not: 16 GB RAM'de model biraz yavaş yanıt verebilir; ilk birkaç saniye
> normal. Eğer belleğiniz zorlanırsa `ollama pull qwen2.5:3b-instruct`
> deneyip `.env` içindeki `OLLAMA_MODEL` değerini güncelleyebilirsiniz.

---

## 6) Embedding + ChromaDB indeksleme

Bu adım, veritabanındaki event kayıtlarından RAG chunk'ları oluşturur,
bge-m3 ile embed eder ve ChromaDB'ye yazar. **İlk çalıştırmada embedding
modeli HuggingFace'ten indirilir (~2 GB, internet gerekir, tek seferlik).**

```
YAP:
python -m rag.build_index

BEKLENEN:
"[OK] ~63 yeni event chunk'ı document_chunk tablosuna yazıldı."
"[embed] ~71 chunk embed edilecek..."
"[OK] Toplam ~71 chunk embed edilip ChromaDB'ye yazıldı."
```

Bu adımdan sonra `data/chroma_store/` klasörü oluşur — embedding'ler
buraya kalıcı olarak yazılır, tekrar hesaplanmasına gerek kalmaz.

---

## 7) Testleri çalıştırma

```
YAP:
pytest tests/ -v

BEKLENEN:
tests/test_csv_loader.py       -> hepsi PASSED (DB gerektirmez)
tests/test_chunker.py          -> hepsi PASSED (DB gerektirmez)
tests/test_structured_lookup.py -> hepsi PASSED (sadece psycopg kurulu olmalı, DB bağlantısı gerekmez)
tests/test_pipeline_safety.py  -> hepsi PASSED (mock'larla, gerçek Ollama/Chroma gerekmez)
```

> Bu testler, geliştirme sırasında (bu ortamda PostgreSQL/Ollama/ChromaDB
> çalıştırılamadığı için) doğrudan çalıştırılamadı; ancak `csv_loader` ve
> `chunker` modüllerindeki mantık, bağımsız Python script'leriyle elle
> doğrulandı (gerçek CSV verisiyle). `pytest` kurulumunuzdan sonra ilk iş
> olarak bu komutu çalıştırıp sonucu görmenizi öneririz.

---

## 8) Streamlit arayüzünü başlatma

```
YAP:
streamlit run app/streamlit_app.py

BEKLENEN:
Tarayıcıda http://localhost:8501 açılır. Sol panelde PostgreSQL/ChromaDB/
Ollama durumlarının hepsi ✅ olmalı. Soru kutusuna örnek bir soru yazıp
deneyebilirsiniz:

  "MT IVANI'de AE yağı değişimiyle ilgili ne olmuş?"
  "MT IVANI 2021-05-10 tarihinde ne olmuş?"
  "14.05.2021 tarihinde yakıt tüketimi ne kadardı?"
  "Ay uzaya gitti mi?"  (veride olmayan bir soru — sistem "bulunamadı" demeli)
```

---

## Proje yapısı

```
maritime-rag/
├── data/
│   ├── raw/                 # Transkribe edilmiş gerçek Noon Report CSV'leri
│   ├── processed/           # (V1'de kullanılmıyor, gelecek genişletmeler için)
│   └── chroma_store/        # ChromaDB persistent storage (ilk çalıştırmada oluşur)
├── database/
│   ├── migrations/          # 01_lookup_tables, 02_core_entities, 03_seed_reference_data
│   ├── db.py                 # Bağlantı yönetimi
│   └── run_migrations.py
├── ingestion/
│   ├── csv_loader.py         # Saf parsing/doğrulama (DB'siz test edilebilir)
│   └── load_noon_reports.py  # CSV -> PostgreSQL
├── embeddings/
│   └── embedder.py           # bge-m3 wrapper (sentence-transformers)
├── vectorstore/
│   └── chroma_store.py       # ChromaDB wrapper
├── retrieval/
│   ├── retriever.py          # semantic search + basit metadata filtre çıkarımı
│   └── structured_lookup.py  # tarih + ROB/consumption sorularını Postgres'ten kesin çeker
├── llm/
│   └── ollama_client.py      # Ollama HTTP API istemcisi
├── rag/
│   ├── chunker.py            # event -> chunk metni (saf, test edilebilir)
│   ├── build_index.py        # Postgres event -> document_chunk -> embed -> ChromaDB
│   └── pipeline.py           # uçtan uca RAG orkestrasyonu + güvenlik kuralı
├── app/
│   └── streamlit_app.py
├── tests/
├── config/
│   └── settings.py
├── requirements.txt
├── .env.example
└── README.md
```

## V1 kapsamı DIŞINDA olanlar (bilinçli olarak eklenmedi)

Tıbbi/sağlık modülleri, savaş/afet senaryoları, predictive maintenance,
gelişmiş arıza tahmini, mobil uygulama, production deployment, kullanıcı
authentication, mikroservis mimarisi. Bunlar V2/V3 konularıdır.

## Bilinen sınırlamalar (V1)

- Sadece MT IVANI, Ocak–Ağustos 2021 verisi var (çoklu gemi desteği şema
  düzeyinde hazır, ama şu an tek gemi verisi yüklü).
- `voyage` tablosu şemada var ama boş — kaynak verideki sefer notları net
  başlangıç/bitiş tarihi içermiyor.
- Event sınıflandırması (bakım/arıza türü eşleme) basit anahtar kelime
  kurallarıyla yapılıyor; yanlış sınıflandırma olabilir, ancak bu sadece
  metadata filtrelemeyi etkiler — chunk metninin kendisi (RAG'in asıl
  cevap kaynağı) her zaman ham, değiştirilmemiş event metnidir.
- Retrieval eşiği (`DISTANCE_THRESHOLD` in `rag/pipeline.py`) sabit kodlanmış
  bir başlangıç değeridir; gerçek kullanımda çok fazla "bulunamadı" cevabı
  alırsanız bu değeri biraz yükseltmeyi deneyebilirsiniz.

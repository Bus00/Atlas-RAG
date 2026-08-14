-- ====================================================================
-- 02_core_entities.sql
-- Maritime RAG V1 — Transactional (işlemsel) tablolar
-- 01_lookup_tables.sql üzerine inşa edilir; onu DEĞİŞTİRMEZ.
--
-- Bu migration'ın amacı:
--   1) Tüm tablolarda tekrar eden "updated_at" alanını otomatik güncelleyen
--      ortak bir trigger fonksiyonu eklemek (01'de bu eksikti — INSERT'te
--      dolduruluyordu ama UPDATE'te elle güncellenmesi gerekiyordu).
--   2) V1 için gerekli işlemsel tabloları eklemek:
--      vessel, voyage, document, equipment, noon_report,
--      noon_report_event, document_chunk
--
-- Tasarım kararları (gerekçeli):
--   - "voyage" tablosu şemada var ama V1 verisinde sefer başlangıç/bitiş
--     tarihleri net değil (notlar "Sail Priok-Pabelokan" gibi serbest metin).
--     Bu yüzden voyage V1'de çoğunlukla BOŞ kalacak; noon_report.voyage_id
--     nullable'dır. Gelecekte sefer sınırları netleşirse doldurulur.
--   - Ayrı bir "maintenance_record" / "failure_record" tablosu yerine TEK
--     bir "noon_report_event" tablosu kullanılıyor; event_type alanı ile
--     (maintenance/failure/bunker/voyage/other) ayrım yapılıyor ve ilgili
--     lookup tablolarına (maintenance_type, failure_mode) nullable FK ile
--     bağlanıyor. Bu, gereksiz tablo çoğaltmasını önler (kullanıcı isteği).
--   - "document_chunk" RAG'in Postgres tarafındaki "source of truth"udur:
--     chunk metni burada saklanır, embedding ise ChromaDB'de. chroma_id
--     alanı ikisi arasındaki referansı tutar (traceability için).
-- ====================================================================

-- --------------------------------------------------------------------
-- 0) Ortak updated_at trigger fonksiyonu
-- --------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 01_lookup_tables.sql'deki mevcut tablolara da trigger'ı ekliyoruz
-- (bu tablolara zarar vermiyoruz, sadece davranış ekliyoruz)
DO $$
DECLARE
    t TEXT;
BEGIN
    FOR t IN
        SELECT unnest(ARRAY['country','company','office','port','supplier',
                             'equipment_category','maintenance_type',
                             'failure_mode','document_type'])
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_set_updated_at ON %I;
             CREATE TRIGGER trg_set_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION set_updated_at();', t, t);
    END LOOP;
END $$;

-- --------------------------------------------------------------------
-- 1) vessel
-- --------------------------------------------------------------------
CREATE TABLE vessel (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    company_id INTEGER REFERENCES company(id),
    name VARCHAR(150) NOT NULL,
    imo_number VARCHAR(20) UNIQUE,
    vessel_type VARCHAR(100),
    flag_country_id INTEGER REFERENCES country(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_vessel_name UNIQUE (name)
);
CREATE TRIGGER trg_set_updated_at BEFORE UPDATE ON vessel
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- --------------------------------------------------------------------
-- 2) voyage (V1'de çoğunlukla boş — bkz. yukarıdaki not)
-- --------------------------------------------------------------------
CREATE TABLE voyage (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vessel_id INTEGER NOT NULL REFERENCES vessel(id) ON DELETE CASCADE,
    voyage_code VARCHAR(50),
    departure_port_id INTEGER REFERENCES port(id),
    arrival_port_id INTEGER REFERENCES port(id),
    start_date DATE,
    end_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_voyage_vessel_code UNIQUE (vessel_id, voyage_code)
);
CREATE TRIGGER trg_set_updated_at BEFORE UPDATE ON voyage
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- --------------------------------------------------------------------
-- 3) document (kaynak dosya: Noon Report görseli/PDF vb.)
-- --------------------------------------------------------------------
CREATE TABLE document (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_type_id INTEGER NOT NULL REFERENCES document_type(id),
    vessel_id INTEGER NOT NULL REFERENCES vessel(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    source_path VARCHAR(500),
    period_start DATE,
    period_end DATE,
    is_test_data BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_vessel_source UNIQUE (vessel_id, source_path)
);
CREATE TRIGGER trg_set_updated_at BEFORE UPDATE ON document
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- --------------------------------------------------------------------
-- 4) equipment
-- --------------------------------------------------------------------
CREATE TABLE equipment (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vessel_id INTEGER NOT NULL REFERENCES vessel(id) ON DELETE CASCADE,
    equipment_category_id INTEGER NOT NULL REFERENCES equipment_category(id),
    name VARCHAR(150) NOT NULL,
    code VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_equipment_vessel_name UNIQUE (vessel_id, name)
);
CREATE TRIGGER trg_set_updated_at BEFORE UPDATE ON equipment
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- --------------------------------------------------------------------
-- 5) noon_report — günlük yapılandırılmış kayıt (ROB / Consumption)
-- --------------------------------------------------------------------
CREATE TABLE noon_report (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vessel_id INTEGER NOT NULL REFERENCES vessel(id) ON DELETE CASCADE,
    voyage_id INTEGER REFERENCES voyage(id),
    document_id INTEGER REFERENCES document(id),
    report_date DATE NOT NULL,

    fw_rob NUMERIC(10,2),
    bbm_rob NUMERIC(12,2),
    me_oil_rob NUMERIC(10,2),
    ae_oil_rob NUMERIC(10,2),

    fw_consumption NUMERIC(10,2),
    bbm_consumption NUMERIC(12,2),
    me_oil_consumption NUMERIC(10,2),
    ae_oil_consumption NUMERIC(10,2),

    is_test_data BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_noon_report_vessel_date UNIQUE (vessel_id, report_date)
);
CREATE INDEX idx_noon_report_date ON noon_report(report_date);
CREATE TRIGGER trg_set_updated_at BEFORE UPDATE ON noon_report
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- --------------------------------------------------------------------
-- 6) noon_report_event — serbest metin olay/not kayıtları
--    (bakım, arıza, bunker, sefer notu, diğer)
-- --------------------------------------------------------------------
CREATE TABLE noon_report_event (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    noon_report_id INTEGER NOT NULL REFERENCES noon_report(id) ON DELETE CASCADE,
    equipment_id INTEGER REFERENCES equipment(id),
    maintenance_type_id INTEGER REFERENCES maintenance_type(id),
    failure_mode_id INTEGER REFERENCES failure_mode(id),

    event_type VARCHAR(30) NOT NULL
        CHECK (event_type IN ('maintenance','failure','bunker','voyage','other')),
    event_date DATE NOT NULL,
    event_text TEXT NOT NULL,
    equipment_hint VARCHAR(150),   -- ekipman net eşleşmediyse ham metin ipucu
    is_test_data BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_event_date ON noon_report_event(event_date);
CREATE INDEX idx_event_type ON noon_report_event(event_type);

-- --------------------------------------------------------------------
-- 7) document_chunk — RAG için hazırlanmış metin parçaları
--    (Postgres = source of truth metin + metadata, ChromaDB = embedding index)
-- --------------------------------------------------------------------
CREATE TABLE document_chunk (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    noon_report_id INTEGER REFERENCES noon_report(id) ON DELETE CASCADE,
    noon_report_event_id INTEGER REFERENCES noon_report_event(id) ON DELETE CASCADE,

    vessel_id INTEGER NOT NULL REFERENCES vessel(id),   -- denormalize: hızlı metadata filtre
    report_date DATE,                                    -- denormalize: hızlı metadata filtre

    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(30) NOT NULL
        CHECK (chunk_type IN ('event','monthly_summary')),

    chroma_id VARCHAR(100) UNIQUE,   -- ChromaDB'deki karşılık gelen id
    embedded_at TIMESTAMP,           -- embed edilip Chroma'ya yazıldığında doldurulur

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_chunk_document_index UNIQUE (document_id, chunk_index)
);
CREATE INDEX idx_chunk_vessel_date ON document_chunk(vessel_id, report_date);
CREATE INDEX idx_chunk_embedded ON document_chunk(embedded_at);

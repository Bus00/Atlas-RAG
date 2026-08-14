-- ====================================================================
-- 03_seed_reference_data.sql
-- V1 için gerekli minimum referans verisi.
--
-- ÖNEMLİ: Burada YALNIZCA Noon Report kaynaklarında gerçekten geçen
-- bilgiler kullanılıyor (gemi adı "MT IVANI", limanlar "Priok"/"Tanjung
-- Priok" ve "Pabelokan", ülke Endonezya — bu limanlar Endonezce sefer
-- notlarında defalarca geçiyor).
--
-- Şirket (company) bilgisi kaynak dokümanlarda YOK — bu yüzden uydurma
-- bir şirket kaydı OLUŞTURULMUYOR. vessel.company_id NULL bırakılıyor.
-- IMO numarası ve bayrak ülkesi de kaynakta yok, NULL bırakılıyor.
-- ====================================================================

-- Ülke
INSERT INTO country (name, iso_code) VALUES
    ('Indonesia', 'IDN')
ON CONFLICT (name) DO NOTHING;

-- Limanlar (Noon Report sefer notlarında geçen gerçek limanlar)
INSERT INTO port (country_id, name, unlocode)
SELECT id, 'Tanjung Priok', 'IDTPP' FROM country WHERE name = 'Indonesia'
ON CONFLICT (unlocode) DO NOTHING;

INSERT INTO port (country_id, name)
SELECT id, 'Pabelokan' FROM country WHERE name = 'Indonesia'
AND NOT EXISTS (SELECT 1 FROM port WHERE name = 'Pabelokan');

-- Gemi
INSERT INTO vessel (company_id, name, imo_number, vessel_type, flag_country_id)
VALUES (NULL, 'MT IVANI', NULL, 'Tanker (MT)', NULL)
ON CONFLICT (name) DO NOTHING;

-- Doküman türü
INSERT INTO document_type (name) VALUES ('Noon Report')
ON CONFLICT (name) DO NOTHING;

-- Ekipman kategorileri (Noon Report'ta izlenen ana sistemler)
INSERT INTO equipment_category (name, description) VALUES
    ('Main Engine', 'Ana makine (ME)'),
    ('Auxiliary Engine', 'Yardımcı makine (AE)'),
    ('Fresh Water System', 'Taze su sistemi (FW)'),
    ('Hull & Steering', 'Gövde ve dümen/manevra sistemi'),
    ('Cargo System', 'Yük tahliye/yükleme sistemi')
ON CONFLICT (name) DO NOTHING;

-- Bakım türleri (Noon Report notlarında geçen bakım işlemleri)
INSERT INTO maintenance_type (name, description) VALUES
    ('Oil Change', 'Yağ değişimi (ME/AE)'),
    ('Repair', 'Onarım (gövde, dümen, vb.)'),
    ('Supply / Replenishment', 'Sarf malzeme / yağ / zincir ikmali'),
    ('Inspection / Test', 'Denetim veya test (örn. AIS testi, PSA vetting)')
ON CONFLICT (name) DO NOTHING;

-- Arıza türleri (Noon Report notlarında geçen arıza/olay türleri)
INSERT INTO failure_mode (name, description) VALUES
    ('Grounding', 'Karaya oturma (kandas)'),
    ('Hull Damage', 'Gövde hasarı'),
    ('Steering / Maneuvering Fault', 'Dümen / manevra kolu arızası')
ON CONFLICT (name) DO NOTHING;

-- Kaynak doküman kaydı (8 Noon Report görseli/PDF)
INSERT INTO document (document_type_id, vessel_id, title, source_path, period_start, period_end, is_test_data)
SELECT dt.id, v.id, 'Noon Report - MT IVANI - ' || m.label, m.source_path, m.period_start::date, m.period_end::date, FALSE
FROM document_type dt, vessel v,
(VALUES
    ('Periode Januari 2021',  '1.png', '2021-01-01', '2021-01-31'),
    ('Periode February 2021', '2.png', '2021-02-01', '2021-02-28'),
    ('Periode Maret 2021',    '3.png', '2021-03-01', '2021-03-31'),
    ('Periode April 2021',    '4.png', '2021-04-01', '2021-04-30'),
    ('Periode Mei 2021',      '5.png', '2021-05-01', '2021-05-31'),
    ('Periode Juni 2021',     '6.png', '2021-06-01', '2021-06-30'),
    ('Periode Juli 2021',     '7.png', '2021-07-01', '2021-07-31'),
    ('Periode Aug 2021',      '8.png', '2021-08-01', '2021-08-19')
) AS m(label, source_path, period_start, period_end)
WHERE dt.name = 'Noon Report' AND v.name = 'MT IVANI'
ON CONFLICT (vessel_id, source_path) DO NOTHING;

"""
medical/medical_inventory.py
----------------------------------
Tıbbi Envanter (Medical Inventory) -- gemideki (onboard) ilaç ve ekipman
stokunun yapısal takibi.

BU MODÜL HİÇBİR KLİNİK KARAR VERMEZ. Hiçbir şekilde:
  - "bu ilacı kullan" / "şu ilacı hastaya ver" gibi bir öneri
  - doz / tedavi süresi
  - hastalık tedavisi
  - hastalık tanısı/olasılığı
ÜRETMEZ. Sadece "bu ilaçtan ne kadar var, son kullanma tarihi ne zaman,
bu ekipman şu an kullanılabilir mi" gibi SAF ENVANTER sorularına cevap
verir -- tamamen medical/next_best_measurement.py ve
medical/resource_availability.py'deki "kaynak durumu, klinik karar değildir"
ilkesiyle aynı ayrımı korur.

TASARIM KARARI -- NEDEN İLAÇ VE EKİPMAN AYRI DATACLASS'LAR:
İlaç kayıtları için son kullanma tarihi KESİNLİKLE zorunludur (proje
talimatı), ekipman kayıtları için ise yoktur (ekipman zaten "bakım tarihi"
kavramını kullanır, "son kullanma tarihi" değil). Bu iki kavram yapısal
olarak farklı olduğu için (biri zorunlu bir alan, diğeri o alana sahip
bile değil), tek bir ortak dataclass'a zorlamak "Optional ama aslında bazen
zorunlu" gibi belirsiz/gereksiz bir soyutlama yaratırdı. Bunun yerine,
`MedicationItem` ve `EquipmentItem` ayrı, açık dataclass'lar olarak
tanımlandı; ORTAK envanter yapısı ise tek bir `MedicalInventory` container
sınıfında (her ikisini de tutan) sağlanıyor -- proje talimatındaki "ortak
bir envanter yapısı altında tutmak mantıklıysa yap, ama gereksiz soyutlama
oluşturma" ilkesiyle tutarlı.

TASARIM KARARI -- KAYIT DESENİ:
medical/resource_availability.py'deki ResourceRegistry deseniyle
BİLİNÇLİ OLARAK tutarlı: inventory_id'ye göre anahtarlanmış bir sözlük,
register()/get() metodları, sorted() ile deterministik kimlik listesi.
Bu, mevcut mimariyle uyum için tercih edildi (proje talimatı: "Yeni
özellikleri mevcut mimariye uyumlu şekilde ekle").

DETERMİNİZM (proje talimatı -- Tasarım bölümü):
  - Sistem saati (wall-clock time) KULLANILMAZ. Tarihe göre filtreleme
    (son kullanma tarihi geçmiş/yaklaşan) gereken her fonksiyon, karşılaştırma
    tarihini (`as_of`) AÇIK bir parametre olarak alır -- asla kendi başına
    "bugün" varsaymaz.
  - "Yaklaşan son kullanma tarihi" sorgusu da, hangi zaman aralığının
    "yaklaşan" sayılacağını (`within`) AÇIK bir parametre olarak alır --
    sistem kendi başına bir eşik (örn. "30 gün") İCAT ETMEZ; bu, çağıran
    kodun/kullanıcının kararıdır.
  - known_medication_ids() / known_equipment_ids() ve tüm sorgu
    fonksiyonları, sorted() ile alfabetik/deterministik bir sırada döner
    (dict yineleme sırasına GÜVENİLMEZ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from medical.models.errors import MedicalDataValidationError


def _is_number(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


class EquipmentStatus(str, Enum):
    """Proje talimatındaki örnek durumlarla birebir eşleşir."""
    AVAILABLE = "available"        # mevcut
    UNAVAILABLE = "unavailable"    # kullanılamaz
    IN_MAINTENANCE = "in_maintenance"  # bakımda


@dataclass
class MedicationItem:
    """
    Tek bir ilaç kaydının yapısal temsili. `status`, mevcut repodaki
    serbest-metin durum alanı deseniyle tutarlıdır (bkz.
    LaboratoryResult.status, VitalSign.quality) -- kapalı bir enum'a
    zorlanmadı, çünkü ilaç durumu (örn. "aktif", "karantinada",
    "geri çağrıldı") kapalı, sabit bir küme değildir. Son kullanma
    tarihine/stok miktarına göre tespitler (süresi geçmiş, tükenmiş vb.)
    bu alandan DEĞİL, aşağıdaki ayrı sorgu fonksiyonlarından gelir --
    `status` sadece insan tarafından girilen serbest bir notasyondur.
    """
    inventory_id: str
    name: str
    quantity: float
    unit: str
    expiration_date: datetime  # KESİNLİKLE zorunlu (proje talimatı)
    status: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.inventory_id or not self.inventory_id.strip():
            raise MedicalDataValidationError("inventory_id boş olamaz.")
        if not self.name or not self.name.strip():
            raise MedicalDataValidationError("name boş olamaz.")
        if not _is_number(self.quantity):
            raise MedicalDataValidationError(f"quantity sayısal olmalı: {self.quantity!r}")
        if self.quantity < 0:
            raise MedicalDataValidationError(f"quantity negatif olamaz: {self.quantity!r}")
        if not self.unit or not self.unit.strip():
            raise MedicalDataValidationError("unit boş olamaz.")
        if not isinstance(self.expiration_date, datetime):
            raise MedicalDataValidationError(
                f"expiration_date bir datetime nesnesi olmalı (zorunlu): {self.expiration_date!r}"
            )
        if self.status is not None and not self.status.strip():
            raise MedicalDataValidationError("status verilmişse boş olamaz.")
        if self.note is not None and not self.note.strip():
            raise MedicalDataValidationError("note verilmişse boş olamaz.")


@dataclass
class EquipmentItem:
    """
    Tek bir ekipman kaydının yapısal temsili. `maintenance_date`
    OPSİYONELDİR (proje talimatı: ekipman için son kullanma tarihi
    zorunlu değildir -- bu modelde ekipmanın zaten bir "son kullanma
    tarihi" alanı YOKTUR, sadece opsiyonel bir bakım tarihi vardır).
    """
    inventory_id: str
    name: str
    quantity: float
    unit: str
    status: EquipmentStatus
    maintenance_date: Optional[datetime] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.inventory_id or not self.inventory_id.strip():
            raise MedicalDataValidationError("inventory_id boş olamaz.")
        if not self.name or not self.name.strip():
            raise MedicalDataValidationError("name boş olamaz.")
        if not _is_number(self.quantity):
            raise MedicalDataValidationError(f"quantity sayısal olmalı: {self.quantity!r}")
        if self.quantity < 0:
            raise MedicalDataValidationError(f"quantity negatif olamaz: {self.quantity!r}")
        if not self.unit or not self.unit.strip():
            raise MedicalDataValidationError("unit boş olamaz.")
        if not isinstance(self.status, EquipmentStatus):
            raise MedicalDataValidationError(f"status geçerli bir EquipmentStatus olmalı: {self.status!r}")
        if self.maintenance_date is not None and not isinstance(self.maintenance_date, datetime):
            raise MedicalDataValidationError(
                f"maintenance_date verilmişse bir datetime nesnesi olmalı: {self.maintenance_date!r}"
            )
        if self.note is not None and not self.note.strip():
            raise MedicalDataValidationError("note verilmişse boş olamaz.")


@dataclass
class MedicalInventory:
    """
    İlaç ve ekipman kayıtlarının ORTAK envanter container'ı --
    medical/resource_availability.py::ResourceRegistry ile aynı desen
    (inventory_id'ye göre anahtarlanmış sözlük, register()/get(),
    deterministik sıralı kimlik listesi).
    """
    medications: dict[str, MedicationItem] = field(default_factory=dict)
    equipment: dict[str, EquipmentItem] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.medications, dict):
            raise MedicalDataValidationError(f"medications bir sözlük olmalı: {self.medications!r}")
        if not isinstance(self.equipment, dict):
            raise MedicalDataValidationError(f"equipment bir sözlük olmalı: {self.equipment!r}")
        for inventory_id, item in self.medications.items():
            if not isinstance(item, MedicationItem):
                raise MedicalDataValidationError(f"medications içindeki her değer bir MedicationItem olmalı: {item!r}")
            if item.inventory_id != inventory_id:
                raise MedicalDataValidationError(
                    f"medications sözlüğündeki anahtar ('{inventory_id}') MedicationItem.inventory_id "
                    f"('{item.inventory_id}') ile eşleşmiyor."
                )
        for inventory_id, item in self.equipment.items():
            if not isinstance(item, EquipmentItem):
                raise MedicalDataValidationError(f"equipment içindeki her değer bir EquipmentItem olmalı: {item!r}")
            if item.inventory_id != inventory_id:
                raise MedicalDataValidationError(
                    f"equipment sözlüğündeki anahtar ('{inventory_id}') EquipmentItem.inventory_id "
                    f"('{item.inventory_id}') ile eşleşmiyor."
                )

    def register_medication(self, item: MedicationItem) -> None:
        if not isinstance(item, MedicationItem):
            raise MedicalDataValidationError(f"item bir MedicationItem olmalı: {item!r}")
        self.medications[item.inventory_id] = item

    def register_equipment(self, item: EquipmentItem) -> None:
        if not isinstance(item, EquipmentItem):
            raise MedicalDataValidationError(f"item bir EquipmentItem olmalı: {item!r}")
        self.equipment[item.inventory_id] = item

    def get_medication(self, inventory_id: str) -> Optional[MedicationItem]:
        return self.medications.get(inventory_id)

    def get_equipment(self, inventory_id: str) -> Optional[EquipmentItem]:
        return self.equipment.get(inventory_id)

    def known_medication_ids(self) -> list[str]:
        return sorted(self.medications.keys())

    def known_equipment_ids(self) -> list[str]:
        return sorted(self.equipment.keys())


def build_medical_inventory(
    medications: Optional[list[MedicationItem]] = None,
    equipment: Optional[list[EquipmentItem]] = None,
) -> MedicalInventory:
    """MedicationItem/EquipmentItem listelerinden bir MedicalInventory oluşturur."""
    medications = medications or []
    equipment = equipment or []
    if not isinstance(medications, list):
        raise MedicalDataValidationError(f"medications bir liste olmalı: {medications!r}")
    if not isinstance(equipment, list):
        raise MedicalDataValidationError(f"equipment bir liste olmalı: {equipment!r}")

    inventory = MedicalInventory()
    for item in medications:
        inventory.register_medication(item)
    for item in equipment:
        inventory.register_equipment(item)
    return inventory


# ---------------------------------------------------------------------
# Deterministik sorgu fonksiyonları -- HİÇBİRİ sistem saatini kullanmaz;
# karşılaştırma tarihi (as_of) her zaman AÇIK bir parametredir.
# ---------------------------------------------------------------------

def find_expired_medications(inventory: MedicalInventory, as_of: datetime) -> list[MedicationItem]:
    """
    expiration_date < as_of olan ilaçları, inventory_id'ye göre alfabetik
    deterministik bir sırada döner. `as_of` çağıran kod tarafından AÇIKÇA
    verilir -- sistem kendi başına "bugün" varsaymaz.
    """
    if not isinstance(inventory, MedicalInventory):
        raise MedicalDataValidationError(f"inventory bir MedicalInventory olmalı: {inventory!r}")
    if not isinstance(as_of, datetime):
        raise MedicalDataValidationError(f"as_of bir datetime nesnesi olmalı: {as_of!r}")
    return [
        inventory.medications[mid] for mid in inventory.known_medication_ids()
        if inventory.medications[mid].expiration_date < as_of
    ]


def find_expiring_soon_medications(
    inventory: MedicalInventory, as_of: datetime, within: timedelta,
) -> list[MedicationItem]:
    """
    Henüz süresi geçmemiş (expiration_date >= as_of) AMA
    (as_of + within) içinde sona erecek ilaçları döner. `within`,
    "yaklaşan" sayılacak zaman penceresini çağıran kodun AÇIKÇA
    belirlemesi gerektiğini ifade eder -- sistem hiçbir sabit eşik
    (örn. "30 gün") İCAT ETMEZ.
    """
    if not isinstance(inventory, MedicalInventory):
        raise MedicalDataValidationError(f"inventory bir MedicalInventory olmalı: {inventory!r}")
    if not isinstance(as_of, datetime):
        raise MedicalDataValidationError(f"as_of bir datetime nesnesi olmalı: {as_of!r}")
    if not isinstance(within, timedelta):
        raise MedicalDataValidationError(f"within bir timedelta nesnesi olmalı: {within!r}")
    deadline = as_of + within
    return [
        inventory.medications[mid] for mid in inventory.known_medication_ids()
        if as_of <= inventory.medications[mid].expiration_date <= deadline
    ]


def find_zero_stock_medications(inventory: MedicalInventory) -> list[MedicationItem]:
    """quantity == 0 olan ilaçları, inventory_id'ye göre deterministik sırada döner."""
    if not isinstance(inventory, MedicalInventory):
        raise MedicalDataValidationError(f"inventory bir MedicalInventory olmalı: {inventory!r}")
    return [
        inventory.medications[mid] for mid in inventory.known_medication_ids()
        if inventory.medications[mid].quantity == 0
    ]


def find_equipment_by_status(inventory: MedicalInventory, status: EquipmentStatus) -> list[EquipmentItem]:
    """Belirli bir duruma sahip ekipmanları, inventory_id'ye göre deterministik sırada döner."""
    if not isinstance(inventory, MedicalInventory):
        raise MedicalDataValidationError(f"inventory bir MedicalInventory olmalı: {inventory!r}")
    if not isinstance(status, EquipmentStatus):
        raise MedicalDataValidationError(f"status geçerli bir EquipmentStatus olmalı: {status!r}")
    return [
        inventory.equipment[eid] for eid in inventory.known_equipment_ids()
        if inventory.equipment[eid].status == status
    ]

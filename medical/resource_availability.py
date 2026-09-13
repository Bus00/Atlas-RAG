"""
medical/resource_availability.py
----------------------------------------
V2 Faz (Next Best Measurement & Resource-Aware Decision Support), Bölüm B:
Kaynak Farkındalıklı Karar Desteği (Resource-Aware Decision Support).

BU MODÜL TIBBİ İÇERİK ÜRETMEZ. Sadece geminin/aracın (onboard) yapısal
kaynak durumunu (cihaz, laboratuvar kapasitesi, internet, iletişim,
eğitimli personel vb.) temsil eder: HANGİ kaynağın bilindiğini, mevcut mu
mevcut değil mi yoksa BİLİNMİYOR mu olduğunu.

KRİTİK KURAL: Bu modül HİÇBİR kaynağın var olduğunu VARSAYMAZ. Kayıtlı
olmayan (registry'de bulunmayan) herhangi bir resource_id için durum HER
ZAMAN ResourceState.UNKNOWN döner -- ASLA UNAVAILABLE ya da AVAILABLE
varsayılmaz. "Bilinmiyor" hiçbir zaman "mevcut değil" ya da "mevcut" olarak
YORUMLANMAZ (proje talimatı: "UNKNOWN must remain UNKNOWN").

BU MODÜL:
  - ilaç envanteri/seçimi/dozaj İÇERMEZ
  - tedavi kararı İÇERMEZ
  - hangi kaynağın "gerekli" olduğuna karar VERMEZ (bu,
    medical/next_best_measurement.py'nin işidir -- bu modül sadece
    "bu kaynak_id'nin durumu nedir?" sorusuna cevap verir)

Bu modül, medical/next_best_measurement.py'ye BAĞIMLI DEĞİLDİR (tek yönlü
bağımlılık: next_best_measurement.py bu modülü kullanır, tersi değil) --
böylece "kaynak durumu" kavramı, ölçüm önerisi kavramından TAMAMEN
BAĞIMSIZ test edilebilir ve yeniden kullanılabilir kalır.

DETERMİNİZM:
  - Sistem saati (wall-clock time) veya rastgele sayı üretimi KULLANILMAZ.
  - ResourceRegistry.known_resource_ids(), sorted() ile alfabetik
    deterministik bir sırada döner (dict yineleme sırasına GÜVENİLMEZ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from medical.models.errors import MedicalDataValidationError


class ResourceCategory(str, Enum):
    """
    Onboard kaynak kategorileri. Proje talimatındaki örnek kategorilerle
    sınırlıdır -- burada HİÇBİR spesifik cihaz markası/modeli veya gemiye
    özgü envanter İCAT EDİLMEZ, sadece yapısal kategori etiketleridir.
    """
    MEASUREMENT_DEVICE = "measurement_device"
    LABORATORY_CAPABILITY = "laboratory_capability"
    INTERNET_CONNECTIVITY = "internet_connectivity"
    COMMUNICATION_CAPABILITY = "communication_capability"
    TRAINED_PERSONNEL = "trained_personnel"
    OTHER = "other"


class ResourceState(str, Enum):
    """
    Bir kaynağın durumu -- ASLA bir klinik durum DEĞİLDİR, sadece
    "bu kaynak şu an kullanılabilir mi" sorusunun yapısal cevabıdır.
    """
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class Resource:
    """
    Tek bir onboard kaynağın yapısal temsili.

    resource_id: deterministik, sabit bir string (örn. "device:temperature",
    "laboratory_capability", "internet_connectivity"). Rastgele/otomatik
    üretilen bir kimlik DEĞİLDİR -- çağıran kod (veya next_best_measurement.py)
    tarafından açıkça belirlenir.

    note: opsiyonel, serbest metin bağlam (örn. "son envanter kontrolünde
    bildirildi") -- KLİNİK bir alan DEĞİLDİR, sadece kaynağın durumuna dair
    insan tarafından girilmiş bir açıklamadır.
    """
    resource_id: str
    category: ResourceCategory
    state: ResourceState
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise MedicalDataValidationError(f"resource_id boş olmayan bir string olmalı: {self.resource_id!r}")
        if not isinstance(self.category, ResourceCategory):
            raise MedicalDataValidationError(f"category geçerli bir ResourceCategory olmalı: {self.category!r}")
        if not isinstance(self.state, ResourceState):
            raise MedicalDataValidationError(f"state geçerli bir ResourceState olmalı: {self.state!r}")


@dataclass
class ResourceRegistry:
    """
    Bilinen kaynakların, resource_id'ye göre anahtarlanmış bir koleksiyonu.

    KRİTİK: get_state(), kayıtlı olmayan HERHANGİ bir resource_id için
    ResourceState.UNKNOWN döner -- kayıtlı olmama durumu ASLA
    ResourceState.UNAVAILABLE'a dönüştürülmez (bu, "kaynağın mevcut olmadığını
    VARSAYMAK" anlamına gelirdi, ki bu proje talimatınca yasaktır).
    """
    resources: dict[str, Resource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, dict):
            raise MedicalDataValidationError(f"resources bir sözlük olmalı: {self.resources!r}")
        for resource_id, resource in self.resources.items():
            if not isinstance(resource, Resource):
                raise MedicalDataValidationError(f"resources içindeki her değer bir Resource olmalı: {resource!r}")
            if resource.resource_id != resource_id:
                raise MedicalDataValidationError(
                    f"resources sözlüğündeki anahtar ('{resource_id}') Resource.resource_id "
                    f"('{resource.resource_id}') ile eşleşmiyor."
                )

    def register(self, resource: Resource) -> None:
        """Bir kaynağı ekler ya da (aynı resource_id ile) günceller."""
        if not isinstance(resource, Resource):
            raise MedicalDataValidationError(f"resource bir Resource nesnesi olmalı: {resource!r}")
        self.resources[resource.resource_id] = resource

    def get(self, resource_id: str) -> Optional[Resource]:
        return self.resources.get(resource_id)

    def get_state(self, resource_id: str) -> ResourceState:
        """
        Kayıtlı olmayan bir resource_id için HER ZAMAN UNKNOWN döner --
        var olduğu ya da olmadığı ASLA varsayılmaz.
        """
        resource = self.resources.get(resource_id)
        if resource is None:
            return ResourceState.UNKNOWN
        return resource.state

    def known_resource_ids(self) -> list[str]:
        """Alfabetik, deterministik bir sırada bilinen (kayıtlı) resource_id'ler."""
        return sorted(self.resources.keys())


def build_resource_registry(resources: list[Resource]) -> ResourceRegistry:
    """
    Bir Resource listesinden ResourceRegistry oluşturur. Listedeki sıra
    çıktı sırasını ETKİLEMEZ -- ResourceRegistry.known_resource_ids() zaten
    alfabetik olarak deterministiktir.
    """
    if not isinstance(resources, list):
        raise MedicalDataValidationError(f"resources bir liste olmalı: {resources!r}")
    registry = ResourceRegistry()
    for resource in resources:
        registry.register(resource)
    return registry

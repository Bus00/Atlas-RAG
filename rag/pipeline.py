"""
rag/pipeline.py
------------------
Uçtan uca akış, soru tipine göre iki ayrı yoldan biri işletilir:

  a) Structured query (tarih + ROB/consumption gibi kesin bir soru) ve
     PostgreSQL'de karşılığı varsa: ChromaDB'ye ve LLM'e HİÇ gidilmez,
     cevap doğrudan noon_report kaydından deterministik üretilir
     (format_structured_answer). Bu değerler zaten kesin olduğundan LLM'e
     sormak ekstra risk/gecikme dışında bir şey katmaz.

  b) Aksi halde semantic akış işler:
     soru -> embedding -> ChromaDB retrieval -> ilgili chunk'lar
           -> prompt -> Ollama -> cevap + kaynaklar

Güvenlik prensibi (proje talimatı gereği): LLM SADECE verilen context'e
dayanarak cevap vermeli; context'te yoksa "bu bilgi mevcut veri
kaynaklarında bulunamadı" demeli. Bunu iki katmanda sağlıyoruz:
  1) System prompt açıkça bunu talep ediyor.
  2) Hiç ilgili context bulunamazsa LLM'e hiç sormadan güvenli cevabı
     doğrudan döndürüyoruz (LLM'in "context yok ama yine de bir şeyler
     uydurma" riskini tamamen ortadan kaldırmak için).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm.ollama_client import OllamaClient, OllamaConnectionError
from retrieval.retriever import Retriever, RetrievedChunk
from retrieval.structured_lookup import (
    StructuredFact,
    lookup_monthly_consumption,
    lookup_noon_report_by_date,
    looks_like_monthly_structured_query,
    looks_like_structured_query,
)
from routing.domain_router import Domain, classify_domain

NO_INFO_MESSAGE = "Bu bilgi mevcut veri kaynaklarında bulunamadı."

# V2 Faz 2: tıbbi karar destek modülü (klinik değerlendirme, ilaç güvenliği vb.)
# henüz implemente edilmedi (bkz. Faz 3+). Bu yüzden MEDICAL/BOTH olarak
# yönlendirilen sorularda mevcut denizcilik RAG akışını (yanlış/anlamsız bir
# cevap üretebilecek) hiç çalıştırmıyoruz -- bunun yerine açıkça "henüz
# kullanılamıyor" diyoruz. "No fake implementation" ilkesi gereği.
MEDICAL_NOT_YET_AVAILABLE_MESSAGE = (
    "Bu soru tıbbi bir konuyla ilgili görünüyor. ATLAS'ın tıbbi karar destek "
    "modülü (klinik değerlendirme, ilaç güvenliği vb.) şu anda geliştirme "
    "aşamasındadır ve bu sürümde kullanılamamaktadır."
)

# Benzerlik mesafesi (cosine distance) bu eşiğin üzerindeyse, chunk "alakasız"
# kabul edilir ve context'e dahil edilmez. bge-m3 + cosine için 0.55 makul bir
# başlangıç eşiği (daha küçük mesafe = daha alakalı). Gerekirse .env üzerinden
# ayarlanabilir hale getirilebilir; V1'de sabit kod yeterli.
DISTANCE_THRESHOLD = 0.60

SYSTEM_PROMPT = """Sen bir denizcilik (maritime) operasyon asistanısın. Görevin, sana verilen
Noon Report kayıtlarına (context) dayanarak Türkçe, kısa ve net cevaplar vermek.

KURALLAR (kesinlikle uy):

1. SADECE aşağıda sana verilen context'teki bilgiyi kullan. Context dışında hiçbir bilgi
   uydurma, tahmin etme veya genel denizcilik bilgini ekleme.

2. Sorunun cevabının bir KISMI context'te mevcutsa, mevcut bilgiyi MUTLAKA cevapla.
   Bilginin başka bir kısmı eksikse sadece o eksik kısmı "belirtilmediğini" veya
   "mevcut veride bulunmadığını" belirt.

3. Context'te soruyla ilgili en az bir doğrudan bilgi varsa, cevabı
   "Bu bilgi mevcut veri kaynaklarında bulunamadı." şeklinde reddetme.
   Mevcut bilgiyi kullan ve yalnızca gerçekten eksik olan kısmı belirt.

4. "Bu bilgi mevcut veri kaynaklarında bulunamadı." ifadesini SADECE context'te
   soruyla ilgili hiçbir bilgi bulunmadığında kullan.

5. Bir kaydın "olay türü: ikmal/bunker" veya event_type=bunker olarak etiketlendiğini
   görürsen, bunun bir bunker/ikmal olayı olduğunu kabul et.
   Bunker ile ilgili bilgileri diğer kayıt bilgileriyle birbirine karıştırma.
   Bir kayıttaki eksik veya kesik bilgi, diğer bilgileri yok saymanı GEREKTİRMEZ.
   Bu etikete dayanarak "bunker yapılmadı", "bunker gerçekleşmedi" veya
   "bunker belirtilmemiş" gibi bir sonuç çıkarma.

6. Context içinde bir değer açıkça yazılıysa bu değeri doğru anlamıyla kullan.
   Örneğin "bunker öncesi 4.035 L" ifadesini "bunker sonrası 4.035 L" olarak
   değiştirme. Context'te "bunker sonrası" değeri eksik veya görüntüde kesikse,
   bunu açıkça belirt ve mevcut "bunker öncesi" değerini yine kullan.

7. "Yapılmadı", "gerçekleşmedi", "değişiklik yok" gibi OLUMSUZ bir sonucu SADECE
   context bunu açıkça ve doğrudan söylüyorsa kullan.
   Bir değerin eksik veya kesik olması, olayın gerçekleşmediği anlamına gelmez.

8. Emin olmadığın bir noktada kesin bir sonuç çıkarma. Context'te ne yazıyorsa
   onu aktar ve belirsizliği açıkça belirt.

9. Cevabın sonunda kaynak veya referans listesi oluşturma.
   "Kaynaklar:", "Kayıtlar:", "Dayanak kayıt:" gibi ayrı bir bölüm yazma.
   Kaynaklar uygulama tarafından kullanıcıya ayrıca gösterilecektir.

10. Kullanıcının sorusuyla ilgisi olmayan context bilgilerini cevaba dahil etme.

11. Cevabı kısa, doğrudan ve doğal Türkçe ile ver. Aynı bilgiyi veya
    "bulunamadı" mesajını tekrar etme.

12. Sorunun cevabı context'teki bir veya daha fazla kayıtta açıkça bulunuyorsa,
    önce bu bilgiyi doğrudan cevapla. Eksik bilgiler varsa yalnızca onları belirt."""

@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    used_structured_lookup: bool = False
    llm_was_called: bool = True
    structured_source: dict | None = None
    domain: Domain | None = None  # V2 Faz 2: izlenebilirlik -- hangi domain'e yönlendirildiği


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Kayıt - {c.vessel}, {c.report_date}, tür: {c.event_type or c.chunk_type}] {c.chunk_text}")

    context_block = "\n".join(context_parts) if context_parts else "(context bulunamadı)"

    return f"""CONTEXT:
{context_block}

SORU: {question}

Yukarıdaki CONTEXT'e dayanarak soruyu cevapla. Context'te cevap yoksa bunu açıkça belirt."""


def format_structured_answer(fact: StructuredFact) -> str:
    """
    PostgreSQL'den gelen kesin bir noon_report kaydını, LLM'e hiç sormadan,
    deterministik bir cümleye çevirir.

    Neden LLM'e sormuyoruz: Bu değerler zaten %100 kesin ve yapılandırılmış;
    LLM'in yaptığı tek şey sayıyı cümleye sarmak olurdu, bu da hiçbir katma
    değer sağlamadan hallucination/tutarsızlık riski ekler (bkz. pipeline
    tasarım notu).
    """
    return (
        f"{fact.vessel} - {fact.report_date.isoformat()} tarihli Noon Report kaydına göre:\n"
        f"- FW ROB: {fact.fw_rob} ton, Consumption: {fact.fw_consumption} ton\n"
        f"- BBM ROB: {fact.bbm_rob} L, Consumption: {fact.bbm_consumption} L\n"
        f"- ME Oil ROB: {fact.me_oil_rob} L, Consumption: {fact.me_oil_consumption} L\n"
        f"- AE Oil ROB: {fact.ae_oil_rob} L, Consumption: {fact.ae_oil_consumption} L\n"
        f"(Kaynak: PostgreSQL noon_report kaydı, {fact.vessel} - {fact.report_date.isoformat()})"
    )


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None, llm_client: OllamaClient | None = None):
        self._retriever = retriever or Retriever()
        self._llm = llm_client or OllamaClient()

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        domain = classify_domain(question)

        # V2 Faz 2: MEDICAL/BOTH sorularını, henüz var olmayan bir tıbbi
        # cevap üretmeye çalışmak yerine (ki bu ya NO_INFO_MESSAGE'a düşer
        # ya da denizcilik context'iyle anlamsız bir cevaba yol açabilir),
        # burada açıkça durduruyoruz. Mevcut MARITIME/GENERAL akışı bu
        # kapının altında hiç değişmeden devam ediyor.
        if domain in (Domain.MEDICAL, Domain.BOTH):
            return RagAnswer(
                answer=MEDICAL_NOT_YET_AVAILABLE_MESSAGE,
                sources=[],
                used_structured_lookup=False,
                llm_was_called=False,
                domain=domain,
            )

        if looks_like_monthly_structured_query(question):
            monthly_fact = lookup_monthly_consumption(question)

            if monthly_fact:
                return RagAnswer(
                    answer=(
                        f"MT IVANI'nin {monthly_fact['year']} yılı "
                        f"{monthly_fact['month']}. ayındaki yakıt (BBM) tüketimi "
                        f"{monthly_fact['bbm_consumption']} L'dir."
                    ),
                    sources=[],
                    used_structured_lookup=True,
                    llm_was_called=False,
                    structured_source=monthly_fact,
                    domain=domain,
                )

        if looks_like_structured_query(question):
            fact = lookup_noon_report_by_date(question)
            if fact:
                # Kesin veri bulundu: ChromaDB'ye ve LLM'e hiç gitmeden,
                # deterministik cevabı doğrudan döndür.
                return RagAnswer(
                    answer=format_structured_answer(fact),
                    sources=[],
                    used_structured_lookup=True,
                    llm_was_called=False,
                    domain=domain,
                )

        chunks = self._retriever.retrieve(question, top_k=top_k)
        relevant_chunks = [c for c in chunks if c.distance <= DISTANCE_THRESHOLD]

        if not relevant_chunks:
            return RagAnswer(
                answer=NO_INFO_MESSAGE,
                sources=[],
                used_structured_lookup=False,
                llm_was_called=False,
                domain=domain,
            )

        prompt = _build_prompt(question, relevant_chunks)

        try:
            llm_answer = self._llm.generate(prompt=prompt, system=SYSTEM_PROMPT)
        except OllamaConnectionError as exc:
            return RagAnswer(
                answer=f"[LLM HATASI] {exc}",
                sources=relevant_chunks,
                used_structured_lookup=False,
                llm_was_called=False,
                domain=domain,
            )

        if not llm_answer:
            llm_answer = NO_INFO_MESSAGE

        return RagAnswer(
            answer=llm_answer,
            sources=relevant_chunks,
            used_structured_lookup=False,
            llm_was_called=True,
            domain=domain,
        )
        

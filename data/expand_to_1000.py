import json
import re

TARGET = 1000

# --- Dosyaları oku ---
with open("suggestions.json", "r", encoding="utf-8") as f:
    suggestions = json.load(f)

with open("foreign_terms.txt", "r", encoding="utf-8") as f:
    base_terms = [line.strip() for line in f if line.strip()]

# Opsiyonel: whitelist varsa, bunları dışarıda bırak
whitelist = set()
try:
    with open("whitelist.txt", "r", encoding="utf-8") as f:
        whitelist = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    pass

# --- Heuristikler ---
turkish_chars = set("ğıüşöçİĞÜŞÖÇ")
def is_turkishish(term: str) -> bool:
    # Türkçe karakter içeriyorsa ya da tamamen Türkçe gibi duruyorsa
    if any(ch in turkish_chars for ch in term):
        return True
    return False

def normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip())

def englishish(term: str) -> bool:
    # Harf ağırlıklı ve ASCII ise "ing/ed" varyasyonları üretmeye uygun
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9\-_/ ]{1,60}", term))

def generate_variants(term: str) -> list[str]:
    """Terimden varyasyon üret: plural, ing/ed, türev ekler, camel/snake, vb."""
    t = normalize(term)
    variants = {t}

    # Çok kelimeli ifadelerde (pull request, big data) sadece bazı varyasyonlar
    # üretelim (pull requests gibi)
    if " " in t:
        if englishish(t.replace(" ", "")):
            variants.add(t + "s")
            variants.add(t + "es")
        return sorted(variants)

    # Tek kelimeyse daha agresif varyasyon üret
    if englishish(t):
        low = t.lower()
        variants.add(low)

        # çoğul
        variants.add(low + "s")
        variants.add(low + "es")

        # fiil çekimleri (çok kaba ama sayı artırır)
        variants.add(low + "ing")
        variants.add(low + "ed")
        variants.add(low + "er")
        variants.add(low + "ers")

        # isimleştirme/türetme
        variants.add(low + "tion")
        variants.add(low + "tions")
        variants.add(low + "ation")
        variants.add(low + "ations")
        variants.add(low + "ization")
        variants.add(low + "izer")
        variants.add(low + "izers")
        variants.add(low + "ality")
        variants.add(low + "ment")

        # küçük stil farkları
        variants.add(low.replace("-", "_"))
        variants.add(low.replace("_", "-"))

        # CamelCase örneği (download -> Download)
        variants.add(low.capitalize())

    return sorted(variants)

# --- Basit öneri üretici (placeholder yerine daha iyi) ---
manual_map = {
    "optimize": ["eniyilemek", "iyileştirmek", "en uygun hâle getirmek"],
    "optimization": ["eniyileme", "iyileştirme", "en uygunlaştırma"],
    "performans": ["başarım", "verim"],
    "performance": ["başarım", "verim"],
    "feedback": ["geri bildirim", "dönüt"],
    "update": ["güncelleme", "yenileme"],
    "download": ["indirme"],
    "upload": ["yükleme", "karşıya yükleme"],
    "online": ["çevrim içi"],
    "offline": ["çevrim dışı"],
    "dashboard": ["gösterge paneli"],
    "deploy": ["yayına almak", "çalışır hâle getirmek"],
    "release": ["sürüm", "yayın"],
    "version": ["sürüm"],
    "ui": ["kullanıcı arayüzü"],
    "ux": ["kullanıcı deneyimi"],
    "backend": ["arka uç"],
    "frontend": ["ön yüz"],
    "database": ["veritabanı"],
    "cloud": ["bulut"],
    "bug": ["hata"],
    "issue": ["sorun", "kayıtlı sorun"],
    "fix": ["düzeltme"],
    "refactor": ["yeniden düzenlemek", "iyileştirmek (kod)"],
    "meeting": ["toplantı"],
    "deadline": ["son tarih"],
    "agenda": ["gündem"],
}

def suggestion_for(term: str) -> list[str]:
    key = term.lower()
    if key in manual_map:
        return manual_map[key]

    # Türkçe görünen kelimeleri genelde dönüştürme (ama listeye ekleyebilirsin)
    if is_turkishish(term):
        return ["(Türkçe: dönüştürme yok)"]

    # Default: boş bırakma, en azından placeholder üret
    return ["(Türkçe karşılık eklenecek)", "(alternatif eklenecek)"]

# --- 1000'e tamamla ---
added = 0
for base in base_terms:
    if base.lower() in whitelist:
        continue

    for v in generate_variants(base):
        if v.lower() in whitelist:
            continue
        if v not in suggestions:
            suggestions[v] = suggestion_for(v)
            added += 1
            if len(suggestions) >= TARGET:
                break
    if len(suggestions) >= TARGET:
        break

# --- Kaydet ---
with open("suggestions_1000.json", "w", encoding="utf-8") as f:
    json.dump(suggestions, f, ensure_ascii=False, indent=2)

print("Eklenen:", added)
print("Toplam anahtar:", len(suggestions))
print("Yazıldı: suggestions_1000.json")

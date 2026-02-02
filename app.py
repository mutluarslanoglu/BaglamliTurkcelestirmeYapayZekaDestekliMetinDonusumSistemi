from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# CORS
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "user_profile.db")

# -----------------------------
# Dosya okuma
# -----------------------------
def load_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]

def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_token(tok: str) -> str:
    return tok.lower()

def preserve_casing(original: str, suggestion: str) -> str:
    # Tamamı büyük (kısaltma) zaten koruyoruz ama yine de:
    if original.isupper():
        return suggestion.upper()
    # Baş harf büyükse öneriyi de baş harf büyük yap
    if original[:1].isupper():
        return suggestion[:1].upper() + suggestion[1:]
    return suggestion

def is_all_caps(token: str) -> bool:
    return token.isupper() and len(token) >= 2

def looks_like_proper_noun(original_token: str, token_index_in_sentence: int) -> bool:
    # Cümle başındaki büyük harf normal olabilir
    if token_index_in_sentence == 0:
        return False
    return original_token[:1].isupper()

# -----------------------------
# Basit TR ek ayrıştırma (heuristik)
# Amaç: "optimizasyonun", "performansı", "konfigürasyonu" gibi hallerde kökü yakalamak.
# -----------------------------
TR_SUFFIX_RE = re.compile(
    r"(?P<root>[a-zçğıöşü]+)"
    r"(?P<suffix>(?:"
    r"(?:lar|ler)"
    r"|(?:ım|im|um|üm|m)"
    r"|(?:ın|in|un|ün|n)"
    r"|(?:ı|i|u|ü)"
    r"|(?:a|e)"
    r"|(?:da|de|ta|te)"
    r"|(?:dan|den|tan|ten)"
    r"|(?:ya|ye)"
    r"|(?:ki)"
    r"|(?:dır|dir|dur|dür|tır|tir|tur|tür)"
    r"|(?:mış|miş|muş|müş)"
    r"|(?:acak|ecek)"
    r"|(?:yı|yi|yu|yü)"
    r"|(?:yla|yle)"
    r")*)$",
    re.IGNORECASE
)

def split_root_suffix(norm: str) -> Tuple[str, str]:
    m = TR_SUFFIX_RE.match(norm)
    if not m:
        return norm, ""
    return m.group("root"), m.group("suffix") or ""

# -----------------------------
# SQLite öğrenme
# -----------------------------
def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prefs(
            user_id TEXT NOT NULL,
            foreign_term TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            context_tag TEXT NOT NULL,
            score INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(user_id, foreign_term, suggestion, context_tag)
        )
    """)
    con.commit()
    con.close()

def db_add_score(user_id: str, foreign_term: str, suggestion: str, context_tag: str, delta: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO prefs(user_id, foreign_term, suggestion, context_tag, score)
        VALUES(?,?,?,?,?)
        ON CONFLICT(user_id, foreign_term, suggestion, context_tag)
        DO UPDATE SET score = score + excluded.score
    """, (user_id, foreign_term, suggestion, context_tag, delta))
    con.commit()
    con.close()

def db_get_scores(user_id: str, foreign_term: str, context_tag: str) -> Dict[str, int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT suggestion, score FROM prefs
        WHERE user_id=? AND foreign_term=? AND context_tag=?
    """, (user_id, foreign_term, context_tag))
    rows = cur.fetchall()
    con.close()
    return {s: int(sc) for s, sc in rows}

# -----------------------------
# Uygulama verileri
# -----------------------------
SUGGESTIONS: Dict[str, List[str]] = load_json(os.path.join(DATA_DIR, "suggestions_1000.json"))
FOREIGN_TERMS = set(load_lines(os.path.join(DATA_DIR, "foreign_terms.txt")))
WHITELIST = set(load_lines(os.path.join(DATA_DIR, "whitelist.txt")))

# Çok kelimeli terimleri yakalamak için foreign_terms içinden phrase çıkar
PHRASES = sorted([t for t in FOREIGN_TERMS if " " in t], key=len, reverse=True)

# "Seviye" ayarı: yerleşik kelimeleri azaltmak için bir kırpma listesi
# light: sadece bariz yabancı/teknik İngilizce kelimeleri yakala
# balanced: karışık
# strict: listedeki her şeyi yakala
COMMON_LOANWORDS = {
    "proje", "rapor", "analiz", "model", "metod", "metodoloji", "test", "grafik", "tablo", "form", "format", "sistem"
}

# Koruma: kod bloğu / URL / mail
CODE_LIKE_RE = re.compile(r"```.*?```", re.DOTALL)
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")

# Cümle bölme
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Token: kelime + TR harfleri + apostrof/tire
TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+(?:['’\-][A-Za-zÇĞİÖŞÜçğıöşü]+)?")

@dataclass
class Candidate:
    id: str
    original: str
    foreign_norm: str
    start: int
    end: int
    context: str
    root: str
    suffix: str

def protect_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in CODE_LIKE_RE.finditer(text):
        spans.append((m.start(), m.end()))
    for m in URL_RE.finditer(text):
        spans.append((m.start(), m.end()))
    for m in EMAIL_RE.finditer(text):
        spans.append((m.start(), m.end()))
    spans.sort()
    return spans

def in_protected(a: int, b: int, protected: List[Tuple[int, int]]) -> bool:
    for x, y in protected:
        if a >= x and b <= y:
            return True
    return False

def level_allows(term_norm: str, level: str) -> bool:
    if level == "strict":
        return True
    if level == "balanced":
        # çok yerleşik kelimeleri hariç tut
        return term_norm not in COMMON_LOANWORDS
    # light: çok bariz yabancı/ingilizce veya suggestions sözlüğünde olanlar
    is_ascii = term_norm.isascii()
    return (term_norm in SUGGESTIONS) or (is_ascii and len(term_norm) >= 4)

def detect_candidates(text: str, level: str) -> List[Candidate]:
    protected = protect_spans(text)
    cands: List[Candidate] = []

    # 1) Phrase yakala (çok kelimeli ifadeler)
    lower_text = text.lower()
    for ph in PHRASES:
        if not level_allows(ph, level):
            continue
        # phrase spanlarını ara
        for m in re.finditer(re.escape(ph), lower_text):
            a, b = m.start(), m.end()
            if in_protected(a, b, protected):
                continue
            # basit whitelist kontrolü: phrase içindeki kelimelerden biri whitelist ise dokunma
            if any(w in WHITELIST for w in ph.split()):
                continue
            cid = f"ph:{a}:{b}"
            cands.append(Candidate(
                id=cid,
                original=text[a:b],
                foreign_norm=ph,
                start=a,
                end=b,
                context=get_sentence_context(text, a),
                root=ph,
                suffix=""
            ))

    # 2) Tek kelime yakala
    sentences = SENT_SPLIT.split(text.strip()) if text.strip() else []
    offsets = []
    pos = 0
    for s in sentences:
        start = text.find(s, pos)
        if start == -1:
            start = pos
        offsets.append(start)
        pos = start + len(s)

    for si, s in enumerate(sentences):
        base = offsets[si]
        tokens = list(TOKEN_RE.finditer(s))
        for ti, m in enumerate(tokens):
            original = m.group(0)
            a, b = base + m.start(), base + m.end()
            if in_protected(a, b, protected):
                continue

            # whitelist/kısaltma/özel ad koru
            if original in WHITELIST or normalize_token(original) in {normalize_token(w) for w in WHITELIST}:
                continue
            if is_all_caps(original):
                continue
            if looks_like_proper_noun(original, ti):
                continue

            norm = normalize_token(original)
            root, suffix = split_root_suffix(norm)

            # foreign eşleşmesi: önce kökü, sonra tamamı
            hit = None
            if norm in FOREIGN_TERMS:
                hit = norm
            elif root in FOREIGN_TERMS:
                hit = root

            if not hit:
                continue
            if not level_allows(hit, level):
                continue

            cid = f"w:{a}:{b}"
            cands.append(Candidate(
                id=cid,
                original=original,
                foreign_norm=hit,
                start=a,
                end=b,
                context=s,
                root=root,
                suffix=suffix
            ))

    # üst üste binenleri ele (phrase > kelime öncelikli)
    cands.sort(key=lambda c: (c.start, -(c.end - c.start)))
    filtered: List[Candidate] = []
    last_end = -1
    for c in cands:
        if c.start < last_end:
            continue
        filtered.append(c)
        last_end = c.end
    return filtered

def get_sentence_context(text: str, idx: int) -> str:
    # idx'nin geçtiği cümleyi bul (basit)
    parts = SENT_SPLIT.split(text)
    pos = 0
    for p in parts:
        end = pos + len(p)
        if pos <= idx <= end:
            return p.strip()
        pos = end + 1
    return text.strip()

def rank_suggestions(user_id: str, foreign_norm: str, base_suggestions: List[str], context_tag: str) -> List[Dict]:
    scores = db_get_scores(user_id, foreign_norm, context_tag)
    items = [{"suggestion": s, "score": scores.get(s, 0)} for s in base_suggestions]
    items.sort(key=lambda x: x["score"], reverse=True)
    return items

def apply_replacements(text: str, replacements: List[Dict]) -> str:
    reps = sorted(replacements, key=lambda r: r["start"], reverse=True)
    out = text
    for r in reps:
        out = out[:r["start"]] + r["new"] + out[r["end"]:]
    return out

# -----------------------------
# API Şemaları
# -----------------------------
class AnalyzeRequest(BaseModel):
    user_id: str = "default"
    text: str
    context_tag: str = "akademik"     # akademik / egitsel / kurumsal
    level: str = "balanced"           # light / balanced / strict

class Choice(BaseModel):
    candidate_id: str
    chosen: Optional[str] = None
    rejected: List[str] = []

class ApplyRequest(BaseModel):
    user_id: str = "default"
    text: str
    context_tag: str = "akademik"
    level: str = "balanced"
    choices: List[Choice]

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Bağlam Duyarlı Türkçeleştirme Asistanı")

# CORS: web/index.html rahat çalışsın
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_init()

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    cands = detect_candidates(req.text, req.level)
    results = []

    for c in cands:
        base = SUGGESTIONS.get(c.foreign_norm, [])

        ranked = rank_suggestions(req.user_id, c.foreign_norm, base, req.context_tag)

        results.append({
            "id": c.id,
            "original": c.original,
            "foreign_norm": c.foreign_norm,
            "start": c.start,
            "end": c.end,
            "context": c.context,
            "suggestions": ranked
        })

    report = {
        "candidates_found": len(results),
        "unique_foreign_terms": len(set(r["foreign_norm"] for r in results)),
        "level": req.level
    }
    return {"items": results, "report": report}

@app.post("/apply")
def apply(req: ApplyRequest):
    analyzed = analyze(AnalyzeRequest(user_id=req.user_id, text=req.text, context_tag=req.context_tag, level=req.level))
    items = {it["id"]: it for it in analyzed["items"]}

    replacements = []

    for ch in req.choices:
        it = items.get(ch.candidate_id)
        if not it:
            continue

        foreign = it["foreign_norm"]
        original = it["original"]

        # reddedilenleri -1 puan
        for r in ch.rejected:
            if r:
                db_add_score(req.user_id, foreign, r, req.context_tag, -1)

        # seçilen +2 puan ve uygula
        if ch.chosen:
            db_add_score(req.user_id, foreign, ch.chosen, req.context_tag, +2)
            new_word = preserve_casing(original, ch.chosen)
            replacements.append({"start": it["start"], "end": it["end"], "new": new_word})

    new_text = apply_replacements(req.text, replacements)

    return {
        "new_text": new_text,
        "applied_count": len(replacements),
        "report": analyzed["report"]
    }

@app.get("/")
def root():
    return {"ok": True, "message": "API çalışıyor. /analyze ve /apply kullan."}

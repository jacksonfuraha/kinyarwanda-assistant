"""AI engine: translation (MarianMT), TTS + ASR (Meta MMS), phoneme + cosine scoring, progress."""
import csv, json, os, re, time, hashlib
from pathlib import Path

ROOT = Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text())
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
REF = ROOT / "references"; REF.mkdir(exist_ok=True)
def ref_path(text): return REF / (re.sub(r"\s+", "_", norm(text)) + ".wav")
PROGRESS = ROOT / "progress.json"
SR = 16000

def norm(t): return re.sub(r"[^a-z' ]", "", t.lower()).strip()

def load_lexicon():
    """words.csv: first column = English, second = Kinyarwanda (header optional)."""
    d, p = {}, ROOT / "words.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in csv.reader(fh) if len(r) >= 2 and r[0].strip() and r[1].strip()]
        if rows and rows[0][0].strip().lower() in ("english", "en"): rows = rows[1:]
        for r in rows: d[norm(r[0])] = r[1].strip()
    return d or {"hello": "Muraho", "thank you": "Murakoze", "water": "Amazi", "friend": "Inshuti", "peace": "Amahoro"}

LEXICON = load_lexicon()
REVERSE = {v.lower(): k for k, v in LEXICON.items()}
_M = {}  # lazy model cache


# ---------- Phoneme analysis (Kinyarwanda orthography is largely phonemic) ----------
UNITS = sorted(["nshy", "shy", "cyw", "ryw", "pfw", "nsh", "ny", "sh", "cy", "jy", "ts", "pf", "mb",
                "nd", "ng", "nk", "mp", "nt", "nj", "ns", "nz", "bw", "mw", "kw", "gw", "rw", "tw",
                "dw", "fw", "sw", "zw", "my", "by", "ty", "py", "ky"], key=len, reverse=True)
def phonemes(text):
    s, i, out = norm(text).replace(" ", ""), 0, []
    while i < len(s):
        if s[i] in "aeiou" and i + 1 < len(s) and s[i + 1] == s[i]:
            out.append(s[i] + ":"); i += 2; continue
        u = next((u for u in UNITS if s.startswith(u, i)), s[i])
        out.append(u); i += len(u)
    return out

def syllables(text):  # pronunciation guide, e.g. murakoze -> mu-ra-ko-ze
    return " ".join("-".join(re.findall(r"[^aeiou]*[aeiou]+|[^aeiou]+$", w)) for w in norm(text).split())

def edit_sim(a, b):
    if not a and not b: return 1.0
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return 1 - d[-1] / max(len(a), len(b))

# ---------- Translation ----------
def model_translate(text, direction="en_rw"):
    key = "mt_" + direction
    if key not in _M:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        n = CFG["translation_" + direction]
        _M[key] = (AutoTokenizer.from_pretrained(n), AutoModelForSeq2SeqLM.from_pretrained(n))
    tok, m = _M[key]
    out = m.generate(**tok(text, return_tensors="pt"), num_beams=CFG["num_beams"], max_new_tokens=CFG["max_new_tokens"])
    return tok.decode(out[0], skip_special_tokens=True)

def translate(text):
    """Returns (kinyarwanda_text, english_text, source). Auto-detects direction."""
    k = norm(text)
    if k in LEXICON: return LEXICON[k], k, "lexicon"
    if k in REVERSE: return text.strip(), REVERSE[k], "lexicon"
    try:
        out = model_translate(text, "en_rw")
        return out, text.strip(), "model"
    except Exception as e:
        return "", text, f"unavailable ({e.__class__.__name__})"

# ---------- Audio ----------
def _tts_model():
    if "tts" not in _M:
        from transformers import VitsModel, AutoTokenizer
        _M["tts"] = (VitsModel.from_pretrained(CFG["tts_model"]), AutoTokenizer.from_pretrained(CFG["tts_model"]))
    return _M["tts"]

def tts(text, voice="Recorded"):
    import soundfile as sf, torch, librosa
    if voice in ("Recorded", "Female"):
        if ref_path(text).exists(): return str(ref_path(text))   # your own recording first
        voice = "Female"                                          # no recording yet -> AI voice
    f = CACHE / f"{hashlib.md5((norm(text) + voice).encode()).hexdigest()}.wav"
    if f.exists(): return str(f)
    model, tok = _tts_model()
    with torch.no_grad():
        y = model(**tok(norm(text), return_tensors="pt")).waveform[0].numpy()
    sr = model.config.sampling_rate
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if voice == "Male":  # MMS has one speaker -> male voice is a pitch-shifted rendering
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=CFG["male_pitch_steps"])
    sf.write(f, y, SR); return str(f)

def cultural_sound(kind):
    """Synthesised Intore-style drum pattern / dance rhythm (no copyrighted audio)."""
    import numpy as np, soundfile as sf
    f = CACHE / f"{kind}.wav"
    bpm, pattern = (110, [1, 0, .6, .6, 1, 0, .6, 0]) if "Drum" in kind else (128, [1, .5, .6, .5, 1, .5, .6, .8])
    step = int(SR * 60 / bpm / 2); y = np.zeros(step * len(pattern) * 4)
    t = np.arange(int(SR * .35)) / SR
    rng = np.random.default_rng(0)
    for n in range(len(pattern) * 4):
        a = pattern[n % len(pattern)]
        if a:
            f0 = 90 if a == 1 else 150
            hit = np.sin(2 * np.pi * (f0 + 60 * np.exp(-30 * t)) * t) * np.exp(-9 * t) + .15 * rng.standard_normal(len(t)) * np.exp(-40 * t)
            y[n * step:n * step + len(t)] += a * hit[:len(y) - n * step]
    sf.write(f, .8 * y / np.abs(y).max(), SR); return str(f)

def _load(path):
    import librosa
    return librosa.load(path, sr=SR, mono=True)[0]

def _asr():
    import torch
    if "asr" not in _M:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        proc = AutoProcessor.from_pretrained(CFG["asr_model"], target_lang=CFG["asr_lang"])
        model = Wav2Vec2ForCTC.from_pretrained(CFG["asr_model"], target_lang=CFG["asr_lang"], ignore_mismatched_sizes=True)
        model.eval(); torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1)); _M["asr"] = (proc, model)
    return _M["asr"]

def transcribe(path):
    import torch, librosa
    proc, model = _asr()
    y, _ = librosa.effects.trim(_load(path), top_db=25)      # less silence = faster
    inp = proc(y, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode(): ids = torch.argmax(model(**inp).logits, -1)[0]
    return proc.decode(ids)

def warm_up():
    """Load the heavy speech model in the background so the first Record click is not slow."""
    import threading
    def run():
        try:
            if CFG["use_asr"]: _asr()
        except Exception: pass
    threading.Thread(target=run, daemon=True).start()

# ---------- Scoring ----------
def acoustic_similarity(ref_path, usr_path):
    """MFCC (CMVN) + DTW alignment; cosine similarity averaged along the warping path -> 0..1."""
    import numpy as np, librosa
    def feat(p):
        y, _ = librosa.effects.trim(_load(p), top_db=25)
        m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)[1:]
        m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)
        return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-8)
    X, Y = feat(ref_path), feat(usr_path)
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
    cos = float(np.mean([X[:, i] @ Y[:, j] for i, j in wp]))
    lo, hi = CFG["calibration"]["floor"], CFG["calibration"]["ceil"]
    return float(np.clip((cos - lo) / (hi - lo), 0, 1)), cos

def feedback(score):
    return ("Excellent!", 3) if score >= 90 else ("Great!", 2) if score >= 75 else ("Good", 1) if score >= 60 else ("Try again", 0)

def model_confidence(target, usr_path):
    """Probability from the CNN trained on YOUR voice that the recording is the target word (None if no model)."""
    import torch
    from wordmodel import WordCNN, feats
    d = ROOT / "model"
    if not (d / "word_cnn.pt").exists(): return None
    if "cnn" not in _M:
        labels = json.loads((d / "labels.json").read_text())
        m = WordCNN(len(labels)); m.load_state_dict(torch.load(d / "word_cnn.pt", map_location="cpu")); m.eval()
        _M["cnn"] = (m, labels)
    m, labels = _M["cnn"]; k = norm(target)
    if k not in labels: return None
    x = torch.tensor(feats(_load(usr_path)))[None, None]
    with torch.no_grad(): p = torch.softmax(m(x), 1)[0]
    return float(p[labels.index(k)])

def score_pronunciation(target, usr_path, voice="Recorded"):
    ref = tts(target, "Recorded")
    ac, cos = acoustic_similarity(ref, usr_path)
    parts, heard = {"acoustic": ac}, ""
    if CFG["use_asr"]:
        try:
            heard = transcribe(usr_path); parts["phoneme"] = edit_sim(phonemes(target), phonemes(heard))
        except Exception as e:
            heard = f"(ASR unavailable: {e.__class__.__name__})"
    mc = model_confidence(target, usr_path)
    if mc is not None: parts["model"] = mc
    w = CFG["weights"]
    total = sum(w[k] * v for k, v in parts.items()) / sum(w[k] for k in parts)
    s = round(100 * total); fb, stars = feedback(s)
    return {"score": s, "acoustic": round(100 * ac), "cosine": round(cos, 3), "heard": heard,
            "phoneme": None if "phoneme" not in parts else round(100 * parts["phoneme"]),
            "model": None if mc is None else round(100 * mc), "feedback": fb, "stars": stars}

# ---------- Progress ----------
def load_progress():
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}

def save_attempt(word, score):
    p = load_progress(); w = norm(word)
    p[w] = max(p.get(w, 0), score); PROGRESS.write_text(json.dumps(p, indent=1))

def progress_summary():
    p = load_progress(); n = sum(1 for k, v in p.items() if k in {norm(x) for x in LEXICON.values()} and v >= CFG["mastered_threshold"])
    tot = max(len(LEXICON), 1)
    lvl = "Advanced Speaker" if n >= 0.7 * tot else "Intermediate Speaker" if n >= 0.35 * tot else "Beginner"
    return n, tot, lvl

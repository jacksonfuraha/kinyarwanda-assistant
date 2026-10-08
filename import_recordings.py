"""Import recordings you saved elsewhere into this project.
Usage:  python import_recordings.py "C:\\path\\to\\your\\recordings"
Works when files are named after the word (murakoze.wav, murakoze_2.mp3, Murakaza neza (3).wav,
"thank you 1.wav") OR sit inside a folder named after the word. Searches sub-folders too."""
import re, sys, shutil
from pathlib import Path
import librosa, soundfile as sf
import engine as E

if len(sys.argv) < 2: raise SystemExit('Give the folder, e.g. python import_recordings.py "D:\\my_sounds"')
src = Path(sys.argv[1]); DATA = E.ROOT / "dataset"; REF = E.ROOT / "references"; REF.mkdir(exist_ok=True)
slug = lambda w: re.sub(r"\s+", "_", E.norm(w))
valid = {slug(v): v for v in E.LEXICON.values()}
valid.update({slug(k): v for k, v in E.LEXICON.items()})      # English names work too
def clean(s): return slug(re.sub(r"[\d_\-()\[\]]+", " ", s))
def word_of(f):
    for c in (f.stem, f.parent.name):
        if clean(c) in valid: return valid[clean(c)]
    for c, ok in ((f.stem, True), (f.parent.name, f.parent != src)):   # not in words.csv -> use the name itself
        w = re.sub(r"\s+", " ", re.sub(r"[\d_\-()\[\]]+", " ", c)).strip()
        if ok and w and w.lower() != "take": return w.capitalize()
done, skipped = {}, []
for f in sorted(p for p in src.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}):
    w = word_of(f)
    if not w: skipped.append(f.name); continue
    try: y, _ = librosa.load(f, sr=E.SR, mono=True)
    except Exception: skipped.append(f.name + " (cannot read)"); continue
    y, _ = librosa.effects.trim(y, top_db=30)
    d = DATA / slug(w); d.mkdir(parents=True, exist_ok=True)
    out = d / f"take_{len(list(d.glob('take_*.wav'))) + 1:02d}.wav"
    sf.write(out, y / (abs(y).max() + 1e-9) * 0.9, E.SR)
    if not (REF / f"{slug(w)}.wav").exists(): shutil.copy(out, REF / f"{slug(w)}.wav")
    done[w] = done.get(w, 0) + 1
print(f"Imported {sum(done.values())} files for {len(done)} words:")
for w, n in sorted(done.items()): print(f"  {w}: {n}")
if skipped: print(f"\nSkipped {len(skipped)} (name did not match a word):", ", ".join(skipped[:15]))

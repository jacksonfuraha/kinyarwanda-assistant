"""Shows which words will play YOUR recording. Run: python check_recordings.py"""
import soundfile as sf
import engine as E
ok = 0
for w in E.LEXICON.values():
    p = E.ref_path(w)
    if p.exists():
        i = sf.info(p); ok += 1; print(f"OK    {w:<18} {i.duration:4.1f}s  {p.name}")
    else:
        print(f"MISSING {w:<16} (will use the AI voice)  expected: {p.name}")
print(f"\n{ok} of {len(E.LEXICON)} words have your recording.")

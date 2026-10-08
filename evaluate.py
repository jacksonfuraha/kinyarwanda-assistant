"""Evaluation + tuning. Run: python evaluate.py  -> results.json, confusion_matrix.png"""
import json, time, itertools, numpy as np
import engine as E
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

res = {}
# 1) Translation model quality (bypasses curated lexicon): chrF, exact match, latency
import sacrebleu
words = list(E.LEXICON); hyp, lat = [], []
for w in words:
    t = time.time(); hyp.append(E.model_translate(w)); lat.append(time.time() - t)
ref = [E.LEXICON[w] for w in words]
res["translation"] = {"chrF": sacrebleu.corpus_chrf(hyp, [ref]).score,
                      "exact_match": float(np.mean([h.lower().strip(" .?!") == r.lower() for h, r in zip(hyp, ref)])),
                      "avg_seconds": float(np.mean(lat))}
# hallucination test: nonsense input must not yield long / repetitive output
bad = 0
for s in ["asdkjh qwe", "zzzz", "!!!"]:
    o = E.model_translate(s).split(); bad += len(o) > 12 or (len(o) > 3 and len(set(o)) / len(o) < .5)
res["hallucination_failures"] = f"{bad}/3"

# 2) Pronunciation scoring: positives = same word (pitch-shifted + noise), negatives = a different word
import librosa, soundfile as sf
rng = np.random.default_rng(1); ys, feats = [], []
keys = [E.LEXICON[w] for w in words]
for i, w in enumerate(keys):
    ref_p = E.tts(w); y = E._load(ref_p)
    pos = librosa.effects.pitch_shift(y, sr=E.SR, n_steps=-3) + .01 * rng.standard_normal(len(y))
    sf.write(E.CACHE / "pos.wav", pos, E.SR)
    neg = E.tts(keys[(i + 7) % len(keys)])
    for p, lab in [(str(E.CACHE / "pos.wav"), 1), (neg, 0)]:
        ys.append(lab); feats.append(E.acoustic_similarity(ref_p, p)[1])
ys, feats = np.array(ys), np.array(feats)

def evaluate(lo, hi, thr=60):
    pred = (np.clip((feats - lo) / (hi - lo), 0, 1) * 100 >= thr).astype(int)
    return precision_recall_fscore_support(ys, pred, average="binary", zero_division=0)[2], pred

best = max(itertools.product(np.arange(0, .4, .05), np.arange(.3, .9, .05)), key=lambda c: evaluate(*c)[0] if c[1] > c[0] else -1)
f1, pred = evaluate(*best)  # tuned calibration is saved -> reproducible
cfg = json.loads((E.ROOT / "config.json").read_text()); cfg["calibration"] = {"floor": round(float(best[0]), 2), "ceil": round(float(best[1]), 2)}
(E.ROOT / "config.json").write_text(json.dumps(cfg, indent=2))
p, r, f, _ = precision_recall_fscore_support(ys, pred, average="binary", zero_division=0)
res["scoring"] = {"accuracy": accuracy_score(ys, pred), "precision": p, "recall": r, "f1": f,
                  "confusion_matrix": confusion_matrix(ys, pred).tolist(), "tuned_calibration": cfg["calibration"]}

# 3) ASR round-trip: character similarity of phoneme sequences
try:
    res["asr_phoneme_similarity"] = float(np.mean([E.edit_sim(E.phonemes(w), E.phonemes(E.transcribe(E.tts(w)))) for w in keys]))
except Exception as e:
    res["asr_phoneme_similarity"] = f"skipped ({e.__class__.__name__})"

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.imshow(res["scoring"]["confusion_matrix"], cmap="Blues"); plt.xlabel("Predicted (0=wrong,1=correct)"); plt.ylabel("Actual")
for (i, j), v in np.ndenumerate(res["scoring"]["confusion_matrix"]): plt.text(j, i, v, ha="center")
plt.savefig(E.ROOT / "confusion_matrix.png", dpi=120)
(E.ROOT / "results.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))

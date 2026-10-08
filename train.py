"""Train a CNN on YOUR recorded words. Run: python train.py
Data: dataset/<word>/take_01.wav ... (made by record_words.py) + references/<word>.wav"""
import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from wordmodel import WordCNN, feats, augment, SR

ROOT = Path(__file__).parent; OUT = ROOT / "model"; OUT.mkdir(exist_ok=True)
CFG = {"epochs": 40, "batch": 32, "lr": 1e-3, "aug_per_file": 8, "min_takes": 8, "seed": 42}
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"]); rng = np.random.default_rng(CFG["seed"])

files, folders = [], set()
for d in (ROOT / "dataset").glob("*"):
    if d.is_dir(): folders.add(d.name); files += [(f, d.name) for f in d.glob("*.wav")]
files += [(f, f.stem) for f in (ROOT / "references").glob("*.wav") if f.stem not in folders]
count = {}
for _, w in files: count[w] = count.get(w, 0) + 1
keep = sorted(w for w, n in count.items() if n >= CFG["min_takes"])
for w, n in sorted(count.items()):
    if n < CFG["min_takes"]: print(f"skipped '{w}': only {n} recordings (need {CFG['min_takes']}+)")
if len(keep) < 2: raise SystemExit("Record at least 8 takes for 2+ words first (python record_words.py).")
idx = {w: i for i, w in enumerate(keep)}
data = [(librosa.load(f, sr=SR, mono=True)[0], idx[w]) for f, w in files if w in idx]
y = [c for _, c in data]
tr, tmp = train_test_split(np.arange(len(data)), test_size=.3, stratify=y, random_state=CFG["seed"])
va, te = train_test_split(tmp, test_size=.5, stratify=[y[i] for i in tmp], random_state=CFG["seed"])

def build(ids, aug):
    X, Y = [], []
    for i in ids:
        w, c = data[i]; X.append(feats(w)); Y.append(c)
        for _ in range(CFG["aug_per_file"] if aug else 0): X.append(feats(augment(w, rng))); Y.append(c)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(Y)
(Xtr, Ytr), (Xva, Yva), (Xte, Yte) = build(tr, True), build(va, False), build(te, False)
print(f"{len(keep)} words | train {len(Ytr)} (augmented) | val {len(Yva)} | test {len(Yte)}")

model = WordCNN(len(keep)); opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"]); lossf = nn.CrossEntropyLoss()
best, state = -1, None
for ep in range(CFG["epochs"]):
    model.train(); perm = torch.randperm(len(Ytr))
    for b in range(0, len(perm), CFG["batch"]):
        ids = perm[b:b + CFG["batch"]]; opt.zero_grad(); lossf(model(Xtr[ids]), Ytr[ids]).backward(); opt.step()
    model.eval()
    with torch.no_grad(): acc = (model(Xva).argmax(1) == Yva).float().mean().item()
    if acc >= best: best, state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    if ep % 5 == 4: print(f"epoch {ep + 1}: val accuracy {acc:.2%}")
model.load_state_dict(state); model.eval()
with torch.no_grad(): pred = model(Xte).argmax(1).numpy()
p, r, f, _ = precision_recall_fscore_support(Yte, pred, average="macro", zero_division=0)
res = {"words": len(keep), "test_accuracy": accuracy_score(Yte, pred), "precision": p, "recall": r, "f1": f, "best_val_accuracy": best}
torch.save(model.state_dict(), OUT / "word_cnn.pt")
(OUT / "labels.json").write_text(json.dumps([w.replace("_", " ") for w in keep], indent=1))
(OUT / "train_config.json").write_text(json.dumps({**CFG, **res}, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(9, 8)); plt.imshow(confusion_matrix(Yte, pred, labels=range(len(keep))), cmap="Blues")
plt.xlabel("Predicted word"); plt.ylabel("Actual word"); plt.colorbar(); plt.savefig(OUT / "confusion_matrix.png", dpi=110)
print(json.dumps(res, indent=1)); print("Saved model/word_cnn.pt, labels.json, train_config.json, confusion_matrix.png")

"""Shared pieces for the word-recognition CNN trained on YOUR recordings."""
import numpy as np, torch, torch.nn as nn, librosa
SR, FRAMES, NMELS = 16000, 126, 40   # ~2 s of log-mel features

def feats(y):
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < SR // 10: y = np.pad(y, (0, SR // 10 - len(y)))
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=NMELS, hop_length=256))
    m = (m - m.mean()) / (m.std() + 1e-6)
    out = np.zeros((NMELS, FRAMES), np.float32); n = min(FRAMES, m.shape[1]); out[:, :n] = m[:, :n]
    return out

def augment(y, rng):  # gain + (pitch shift | time stretch) + light noise
    y = y * rng.uniform(.6, 1.2); r = rng.random()
    if r < .35: y = librosa.effects.pitch_shift(y, sr=SR, n_steps=float(rng.uniform(-2, 2)))
    elif r < .7: y = librosa.effects.time_stretch(y, rate=float(rng.uniform(.9, 1.1)))
    return (y + rng.uniform(0, .01) * rng.standard_normal(len(y))).astype(np.float32)

class WordCNN(nn.Module):
    def __init__(s, n):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.3))
        s.c = nn.Linear(64, n)
    def forward(s, x): return s.c(s.f(x))

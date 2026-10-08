"""Record MANY takes of every word (training data). Run: python record_words.py
Saves dataset/<word>/take_NN.wav and keeps take 1 as the reference voice (references/<word>.wav)."""
import re, shutil, gradio as gr, librosa, soundfile as sf
import engine as E

TARGET = 10
WORDS = list(E.LEXICON.values())
DATA = E.ROOT / "dataset"
slug = lambda w: re.sub(r"\s+", "_", E.norm(w))
def takes(w): return len(list((DATA / slug(w)).glob("take_*.wav")))
def status():
    done = sum(takes(w) >= TARGET for w in WORDS)
    todo = [f"{w} ({takes(w)})" for w in WORDS if takes(w) < TARGET][:8]
    return f"Words with {TARGET}+ takes: {done} / {len(WORDS)}. Next: " + (", ".join(todo) or "all done!")
def show(w): return f"## {w}\nGuide: **{E.syllables(w)}**  |  takes recorded: **{takes(w)} / {TARGET}**"

def save(word, audio):
    if not audio: return "Record the word first.", status(), word, None, show(word)
    y, _ = librosa.load(audio, sr=E.SR, mono=True); y, _ = librosa.effects.trim(y, top_db=30)
    d = DATA / slug(word); d.mkdir(parents=True, exist_ok=True)
    f = d / f"take_{takes(word) + 1:02d}.wav"; sf.write(f, y / (abs(y).max() + 1e-9) * 0.9, E.SR)
    if not E.ref_path(word).exists(): shutil.copy(f, E.ref_path(word))
    nxt = word if takes(word) < TARGET else WORDS[(WORDS.index(word) + 1) % len(WORDS)]
    return f"Saved take {takes(word)} of '{word}'.", status(), nxt, None, show(nxt)

with gr.Blocks(title="Record your voice dataset") as demo:
    gr.Markdown(f"# Record your voice dataset\nSay each word {TARGET} times, a little differently each time (speed, distance), in a quiet room.")
    dd = gr.Dropdown(WORDS, value=WORDS[0], label="Word"); big = gr.Markdown(show(WORDS[0]))
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Record")
    btn = gr.Button("Save take", variant="primary"); msg, stat = gr.Markdown(), gr.Markdown(status())
    dd.change(show, dd, big); btn.click(save, [dd, mic], [msg, stat, dd, mic, big])
if __name__ == "__main__": demo.launch()

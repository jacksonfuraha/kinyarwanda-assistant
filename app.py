"""Kinyarwanda Pronunciation Assistant - Gradio UI. Run: python app.py"""
import gradio as gr
import re
import engine as E

def recorded_path(word):
    """Your own recording of this word (references/ first, then dataset/), or None."""
    slug = re.sub(r"\s+", "_", E.norm(word))
    for p in (E.ROOT / "references" / f"{slug}.wav", E.ROOT / "dataset" / slug / "take_01.wav"):
        if p.exists(): return p
    d = E.ROOT / "dataset" / slug
    t = sorted(d.glob("*.wav")) if d.exists() else []
    return t[0] if t else None

_orig_tts = E.tts
def _tts(text, voice="Recorded"):
    if voice != "Male":
        p = recorded_path(text)
        if p: return str(p)
    return _orig_tts(text, "Female" if voice == "Recorded" else voice)
E.tts = _tts      # engine.score_pronunciation also calls tts(), so scoring compares with YOUR recording too

CSS = """
:root,.dark,body.dark,.gradio-container,.dark .gradio-container{--body-background-fill:#f1e9db;--background-fill-primary:#f1e9db;--background-fill-secondary:#f7f0e0;
--block-background-fill:transparent;--block-border-width:0px;--block-shadow:none;--body-text-color:#302b25;--block-label-text-color:#302b25;--block-title-text-color:#302b25;
--input-background-fill:#fffdf9;--input-border-color:#b9ad9a;--border-color-primary:#c9b998;--panel-background-fill:#f7f0e0;--neutral-200:#d8ccb2;color-scheme:light}
body,.dark body{background:#f1e9db!important;color:#302b25}
#txtbox{background:#f7f0e0!important;border:1px solid #d8ccb2!important;border-radius:6px;padding:6px 8px}
#txtbox span,#txtbox label{color:#302b25!important;background:transparent!important;font-weight:700}
#txtbox textarea,#txtbox input{background:#fffdf9!important;color:#302b25!important;border:1px solid #b9ad9a!important;border-radius:4px}
.gradio-container .label-wrap,.gradio-container .label-wrap span{color:#6b6252!important}
#player,#cplayer{max-width:100%}
.gradio-container{max-width:1100px!important;margin:auto!important;font-family:'Trebuchet MS','Segoe UI',sans-serif!important;background:#f1e9db!important}
#banner{height:235px;position:relative;overflow:hidden;text-align:center;color:#fff;border-radius:0 0 4px 4px;background:linear-gradient(180deg,#328ac6 0%,#8fc3e0 58%,#f3d6a5 100%)}
#banner h1{position:relative;z-index:1;width:calc(100% - 200px);margin:0 auto;padding-top:26px;font-size:2.2em;text-shadow:0 2px 4px #263747a6}
#banner p{position:relative;z-index:1;margin:6px 0 0;font-size:1.1em;text-shadow:0 1px 3px #263747a6}
#banner svg{position:absolute;bottom:0;left:0;width:100%;height:150px}
.rw-flag{position:absolute;z-index:2;right:20px;top:18px;width:84px;height:48px;transform:skewY(-4deg);box-shadow:0 2px 5px #23394466}
.rw-flag i{display:block;position:relative}.rw-flag i:nth-child(1){height:50%;background:#20a0d8}.rw-flag i:nth-child(2){height:25%;background:#f5d02a}.rw-flag i:nth-child(3){height:25%;background:#2a9a45}
.rw-flag b{position:absolute;right:8px;top:-12px;width:14px;height:14px;border-radius:50%;background:#f5d02a}
#inputrow{max-width:900px;margin:-30px auto 0!important;position:relative;z-index:3;align-items:end}
#translate{background:linear-gradient(#e0322a,#b3160f)!important;color:#fff!important;font-weight:700;border-radius:6px!important;height:44px}
#panels{border:1px solid #c9b998!important;border-radius:14px;margin:22px 0 0!important;padding:10px;background:#f1e9db}
#right{border-left:1px solid #c9b998!important;padding-left:14px}
.ptitle{text-align:center;font-weight:700;font-size:1.05em;border-bottom:1px solid #d8ccb2;padding-bottom:4px;margin-bottom:6px}
.word{font-size:2.5em;font-weight:800;text-align:center;color:#302b25;border-top:1px solid #c9b998;border-bottom:1px solid #c9b998;padding:2px 0}
.guide{text-align:center;font-size:.9em;color:#6b6252;margin-top:4px}
.hear{text-align:center;font-weight:700;margin:6px 0}
#listen{background:linear-gradient(#ff9a2e,#e86a00)!important;color:#fff!important;font-weight:700;font-size:1.1em;border-radius:6px!important}
#rec_btn{flex:0 0 140px!important;width:140px!important;height:140px!important;min-width:140px!important;max-width:140px!important;aspect-ratio:1/1;align-self:center!important;border-radius:50%!important;padding:0!important;color:#fff!important;font-weight:700;font-size:1.05em;
 background:radial-gradient(circle at 40% 30%,#ee2a24,#b3100c)!important;border:5px solid #f0b050!important;box-shadow:0 0 22px #f0b05088;display:flex!important;flex-direction:column;align-items:center;justify-content:center}
#rec_btn::before{content:"";display:block;width:44px;height:52px;margin-bottom:6px;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.9V21h2v-3.1A7 7 0 0 0 19 11z'/%3E%3C/svg%3E") no-repeat center/contain}
#rec_btn.recording{box-shadow:0 0 0 8px #ee2a2455,0 0 30px #ee2a24!important;transform:scale(1.05)}
#rec_btn{touch-action:none;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;cursor:pointer}
#recrow,#avrow{flex-wrap:nowrap!important;align-items:center!important}
#avrow{align-items:flex-end!important}
.offscreen{position:absolute!important;left:-9999px!important;top:0;width:320px;opacity:0;pointer-events:none}
.score-box{text-align:center}.score-t{font-weight:700;line-height:1.2}
.score-n{font-size:3em;font-weight:800;line-height:1.1}.stars{font-size:.45em;color:#f2b01e;margin-left:4px}.fbk{font-weight:700;font-size:1.1em}.score-box small{font-size:.7em;color:#6b6252}
.prog{margin:16px 20px 0}.prog .bar{height:18px;border-radius:9px;border:1px solid #7d7463;background:#fffaf0;overflow:hidden}
.prog .bar>div{height:100%;background:linear-gradient(#2d6fd0,#0b3a8c)}
.medals{position:relative;height:70px;margin-top:-26px}.medals div{position:absolute;transform:translateX(-50%);text-align:center;font-weight:700;font-size:.85em}
.medals span{display:block;width:34px;height:34px;line-height:32px;margin:0 auto 4px;border-radius:50%;border:3px solid #8a6a1e;background:#e3a93a;color:#fff;font-size:1.1em}
.lvl{text-align:center;font-weight:700;margin-top:4px}
"""
BANNER = """<div id="banner"><div class="rw-flag"><i></i><i><b></b></i><i></i></div>
<h1>Kinyarwanda Pronunciation Assistant</h1><p>Learn to Speak Kinyarwanda Like a Native!</p>
<svg viewBox="0 0 1000 150" preserveAspectRatio="none">
<path d="M0 150V100L120 72 240 104 380 78 500 108 620 90 700 100 740 150Z" fill="#b4c9c0" opacity=".9"/>
<path d="M600 150L790 20 990 150Z" fill="#8fb0a4"/><path d="M790 20L760 62 778 54 790 68 804 54 822 62Z" fill="#f4f7f5"/>
<path d="M0 150V112C150 92 300 122 450 106S750 96 1000 116V150Z" fill="#7f9a6a"/>
<path d="M0 150V128C200 118 400 136 600 127S850 121 1000 131V150Z" fill="#5a7f43"/>
<path d="M0 150V140C250 132 500 144 1000 136V150Z" fill="#c9a93a"/>
<path d="M92 150L96 108 98 78 106 78 108 108 112 150Z" fill="#5a3d28"/>
<ellipse cx="102" cy="72" rx="88" ry="13" fill="#3f6b34"/><ellipse cx="62" cy="86" rx="46" ry="9" fill="#4b7a3a"/><ellipse cx="150" cy="82" rx="44" ry="9" fill="#4b7a3a"/></svg></div>"""
AVATAR = """<svg viewBox="0 0 120 130" width="105"><path d="M10 130C12 100 35 92 60 92S108 100 110 130Z" fill="#b5452a"/>
<path d="M38 108Q60 120 82 108" stroke="#f2c33d" stroke-width="4" fill="none" stroke-dasharray="1 5" stroke-linecap="round"/>
<ellipse cx="60" cy="52" rx="30" ry="34" fill="#c27a3a"/><path d="M28 50C26 15 94 15 92 50C86 30 34 30 28 50Z" fill="#1b1410"/>
<path d="M26 56V44C26 6 94 6 94 44V56" stroke="#9aa0a6" stroke-width="6" fill="none"/><rect x="18" y="46" width="14" height="24" rx="6" fill="#4a4f55"/>
<rect x="88" y="46" width="14" height="24" rx="6" fill="#4a4f55"/><circle cx="49" cy="52" r="3" fill="#222"/><circle cx="71" cy="52" r="3" fill="#222"/>
<path d="M47 68Q60 80 73 68" stroke="#fff" stroke-width="4" fill="#fff" stroke-linecap="round"/></svg>"""

def progress_html():
    n, tot, lvl = E.progress_summary(); pct = 100 * n / tot
    return (f"<div class='prog'><b>Your Fluency Progress:</b><div class='bar'><div style='width:{pct:.0f}%'></div></div>"
            "<div class='medals'><div style='left:14%'><span>★</span>Beginner</div><div style='left:50%'><span style='background:#9a9a9a;border-color:#666'>▲</span>Intermediate Speaker</div>"
            "<div style='left:84%'><span>★</span>Advanced Speaker</div></div>"
            f"<div class='lvl'>Fluency Level: {lvl} &nbsp;|&nbsp; Words Mastered: {n} / {tot}</div></div>")

def score_html(r=None):
    if r is None: return "<div class='score-box'><div class='score-t'>Your Pronunciation<br>Score:</div><div class='score-n'>--%</div><div class='fbk'>Press Record</div></div>"
    col = "#2c8a3a" if r["score"] >= 75 else "#d98a00" if r["score"] >= 60 else "#c8201a"
    ph = "n/a" if r["phoneme"] is None else f"{r['phoneme']}%"; mdl = "n/a" if r["model"] is None else f"{r['model']}%"
    stars = "★" * r["stars"] + "☆" * (3 - r["stars"])
    return (f"<div class='score-box'><div class='score-t'>Your Pronunciation<br>Score:</div><div class='score-n' style='color:{col}'>{r['score']}%<span class='stars'>{stars}</span></div>"
            f"<div class='fbk'>{r['feedback']}</div><small>Sound match {r['acoustic']}% · Phoneme match {ph} · Your-voice model {mdl}</small></div>")

def do_translate(text, voice):
    if not text.strip(): return "<div class='word'>&nbsp;</div>", "", None, ""
    rw, en, src = E.translate(text)
    if not rw: return "<div class='word'>Not available</div>", f"<div class='guide'>{src}</div>", None, ""
    note = "your recording" if (voice != "Male" and recorded_path(rw)) else "AI voice (no recording found)"
    return (f"<div class='word'>{rw} 🔊</div>", f"<div class='guide'>{E.syllables(rw)} · source: {src} · audio: {note}</div>", E.tts(rw, voice), rw)

def do_listen(rw, voice): return E.tts(rw, voice) if rw else None

def do_score(rw, rec, voice):
    if not rw: return "<div class='score-box'><div class='fbk'>Translate a word first</div></div>", progress_html()
    if not rec: return score_html(), progress_html()
    r = E.score_pronunciation(rw, rec, voice); E.save_attempt(rw, r["score"])
    return score_html(r), progress_html()

def do_score_b64(rw, payload, voice):
    """Receives the recording made by the hold-to-record button (WAV, base64 text) and scores it."""
    import base64
    if not payload or "," not in payload: return score_html(), progress_html()
    f = E.CACHE / "last_recording.wav"; f.write_bytes(base64.b64decode(payload.split(",", 1)[1]))
    try:
        return do_score(rw, str(f), voice)
    except Exception as ex:
        import traceback; traceback.print_exc()
        return f"<div class='score-box'><div class='fbk'>Could not score</div><small>{ex.__class__.__name__}: {ex}</small></div>", progress_html()

HOLD_JS = """() => {
 if (window.__vr) return; window.__vr = true;
 const S = {ctx:null, stream:null, ready:false, rec:false, holding:false, chunks:[]};
 const btn = () => document.querySelector('#rec_btn');
 const label = (t, on) => { const b = btn(); if (b) { b.textContent = t; b.classList.toggle('recording', !!on); } };
 const setup = async () => {
   if (S.ready) return;
   S.stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:false, noiseSuppression:false, autoGainControl:true}});
   S.ctx = new (window.AudioContext || window.webkitAudioContext)();
   const src = S.ctx.createMediaStreamSource(S.stream);
   const proc = S.ctx.createScriptProcessor(4096, 1, 1);
   proc.onaudioprocess = e => { if (S.rec) S.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
   src.connect(proc); proc.connect(S.ctx.destination);
   S.ready = true;
 };
 const toWav = (chunks, sr) => {
   let n = chunks.reduce((a, c) => a + c.length, 0), x = new Float32Array(n), o = 0;
   chunks.forEach(c => { x.set(c, o); o += c.length; });
   const r = sr / 16000, m = Math.floor(n / r), pcm = new Int16Array(m);
   for (let i = 0; i < m; i++) {
     const s = Math.floor(i * r), e = Math.max(s + 1, Math.floor((i + 1) * r)); let a = 0;
     for (let j = s; j < e && j < n; j++) a += x[j];
     pcm[i] = Math.max(-1, Math.min(1, a / (e - s))) * 32767;
   }
   const buf = new ArrayBuffer(44 + m * 2), v = new DataView(buf);
   const w = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
   w(0, 'RIFF'); v.setUint32(4, 36 + m * 2, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true);
   v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
   v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, m * 2, true);
   new Int16Array(buf, 44).set(pcm);
   return buf;
 };
 const send = buf => {
   const fr = new FileReader();
   fr.onload = () => {
     const el = document.querySelector('#audio_b64 textarea') || document.querySelector('#audio_b64 input');
     if (!el) { console.error('audio_b64 box not found'); return; }
     const proto = el.tagName === 'INPUT' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
     Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, Date.now() + ',' + fr.result.split(',')[1]);
     el.dispatchEvent(new Event('input', {bubbles: true}));
   };
   fr.readAsDataURL(new Blob([buf], {type: 'audio/wav'}));
 };
 const start = async e => {
   if (!e.target.closest || !e.target.closest('#rec_btn') || S.holding) return;
   e.preventDefault(); S.holding = true; label('Recording...', true);
   try { await setup(); if (S.ctx.state === 'suspended') await S.ctx.resume(); }
   catch (err) { S.holding = false; label('Record', false); alert('Please allow microphone access in your browser, then hold the button again.'); return; }
   if (!S.holding) { label('Record', false); return; }       // released while the microphone was still starting
   S.chunks = []; S.rec = true;
 };
 const end = () => {
   if (!S.holding) return; S.holding = false; label('Record', false);
   if (!S.rec) return; S.rec = false;
   const n = S.chunks.reduce((a, c) => a + c.length, 0);
   if (n < S.ctx.sampleRate * 0.3) { label('Hold longer', false); setTimeout(() => label('Record', false), 1200); return; }
   send(toWav(S.chunks, S.ctx.sampleRate));
 };
 document.addEventListener('pointerdown', start, true);
 document.addEventListener('pointerup', end, true);
 document.addEventListener('pointercancel', end, true);
 window.addEventListener('blur', end);
 document.addEventListener('contextmenu', e => { if (e.target.closest && e.target.closest('#rec_btn')) e.preventDefault(); }, true);
}"""

with gr.Blocks(title="Kinyarwanda Pronunciation Assistant") as demo:
    gr.HTML(BANNER); target = gr.State("")
    with gr.Row(elem_id="inputrow"):
        txt = gr.Textbox(label="Enter an English word or a Kinyarwanda word:", value="Thank you", scale=5, lines=1, max_lines=1, elem_id="txtbox")
        btn = gr.Button("Translate", elem_id="translate", scale=1, min_width=120)
    with gr.Row(elem_id="panels", equal_height=False):
        with gr.Column(elem_id="left", min_width=0):
            gr.HTML("<div class='ptitle'>Kinyarwanda Translation & Audio:</div>")
            out = gr.HTML("<div class='word'>&nbsp;</div>"); guide = gr.HTML()
            with gr.Row(elem_id="avrow"):
                with gr.Column(scale=0, min_width=115): gr.HTML(AVATAR)
                with gr.Column(min_width=0):
                    gr.HTML("<div class='hear'>Hear the Pronunciation</div>")
                    listen = gr.Button("🔊 Listen", elem_id="listen")
                    player = gr.Audio(autoplay=True, interactive=False, show_label=False, elem_id="player")
        with gr.Column(elem_id="right", min_width=0):
            gr.HTML("<div class='ptitle'>Practice Your Pronunciation:</div><div style='font-weight:700'>Press to Speak:</div>")
            with gr.Row(elem_id="recrow"):
                with gr.Column(scale=0, min_width=160): rec = gr.Button("Record", elem_id="rec_btn")
                with gr.Column(min_width=0): sc = gr.HTML(score_html())
    prog = gr.HTML(progress_html())
    with gr.Accordion("Audio options", open=False):
        voice = gr.Radio(["Recorded", "Female", "Male"], value="Recorded", label="Voice")
        cult = gr.Dropdown(["Intore Drums", "Dance Rhythm"], label="Cultural sounds", value=None)
        cplayer = gr.Audio(autoplay=True, interactive=False, label="Cultural sound", elem_id="cplayer")
        mic2 = gr.Audio(sources=["microphone"], type="filepath", label="Backup recorder (use this if the round Record button does nothing)")
    with gr.Column(elem_classes="offscreen"):
        audio_b64 = gr.Textbox(elem_id="audio_b64", show_label=False, lines=2, max_lines=2)
    btn.click(do_translate, [txt, voice], [out, guide, player, target])
    txt.submit(do_translate, [txt, voice], [out, guide, player, target])
    listen.click(do_listen, [target, voice], player)
    cult.change(lambda k: E.cultural_sound(k) if k else None, cult, cplayer)
    demo.load(None, None, None, js=HOLD_JS)
    audio_b64.change(do_score_b64, [target, audio_b64, voice], [sc, prog])
    mic2.stop_recording(do_score, [target, mic2, voice], [sc, prog])

if __name__ == "__main__":
    E.warm_up()
    demo.launch(css=CSS)

# Kinyarwanda Pronunciation Assistant (ITLPA701 - Python & Fundamentals of AI)

Approach: pretrained models (translation, TTS, ASR) + a CNN trained on YOUR recorded voice. Domain: NLP + speech.

## Folder layout (everything in ONE folder)
app.py, engine.py, wordmodel.py, train.py, record_words.py, import_recordings.py, evaluate.py,
config.json, words.csv (english,kinyarwanda), requirements.txt
Created automatically: references/ (one reference voice per word), dataset/ (many takes per word), model/ (trained CNN), cache/

## Run order
1. pip install -r requirements.txt
2. python import_recordings.py "K:\sounds"     (import recordings you already have)  -- or --  python record_words.py
3. python train.py        (needs 8+ takes per word; skip if you only have 1 per word)
4. python app.py          (the application)
5. python evaluate.py     (metrics, tuning, results.json, confusion_matrix.png)

## Architecture
Text -> words.csv lookup (else MarianMT opus-mt-en-rw) -> Kinyarwanda text + syllable guide
-> playback: your recording (references/) or MMS-TTS fallback (female / pitch-shifted male)
Microphone -> score = weighted mix of:
 * acoustic: MFCC + DTW cosine similarity to the reference voice
 * phoneme: ASR (MMS wav2vec 2.0, Kinyarwanda adapter) phonemes vs target phonemes
 * model: CNN confidence (trained in train.py on your recordings; used only if model/ exists)
Weights are in config.json. Progress is saved in progress.json.

## Responsible use
Low-resource translation/ASR can be wrong; scores can be unfair to accents/noise; voice data is personal;
the CNN knows only recorded words and mostly the recorded voices. Mitigations: reviewed word list first,
show the source of each translation, record several speakers, process audio locally, never use for high-stakes decisions.

import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

RATE = 16000
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
out = sys.argv[2] if len(sys.argv) > 2 else '/data/audio/diag.wav'

print(f'Grabando {seconds:.0f} s...')
audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype='float32')
sd.wait()
audio = audio[:, 0]
sf.write(out, audio, RATE)

print(f'Guardado en {out}')
print(f'pico {np.max(np.abs(audio)):.3f}   rms {np.sqrt(np.mean(audio ** 2)):.4f}')
model = load_silero_vad()
for silence in (750, 300):
    spans = get_speech_timestamps(torch.as_tensor(audio), model, sampling_rate=RATE,
                                  min_silence_duration_ms=silence, speech_pad_ms=200)
    lengths = [(s['end'] - s['start']) / RATE for s in spans]
    useful = sum(1 for d in lengths if d >= 1.0)
    print(f'silencio {silence} ms: {len(spans)} tramos, {useful} de >= 1 s, '
          f'duraciones {" ".join(f"{d:.1f}" for d in lengths)}')

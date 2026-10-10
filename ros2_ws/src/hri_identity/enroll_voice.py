#!/usr/bin/env python3
"""Enrola la voz de una persona promediando embeddings ECAPA de sus frases, tomadas de bags o del microfono."""
import argparse
import pathlib
import sys

import numpy as np

from hri_identity.voice import (SAMPLE_RATE, VOICE_DIR, VoiceEncoder, read_bag_audio,
                                speech_spans)

BAGS_DIR = '/data/bags'
OUTLIER_MARGIN = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bags', nargs='+', help='Nombres de bags dentro de /data/bags')
    source.add_argument('--record', type=float, metavar='SEGUNDOS',
                        help='Graba del microfono durante estos segundos')
    source.add_argument('--files', nargs='+',
                        help='Archivos de audio (wav, ogg, flac), por ejemplo una nota de voz')
    parser.add_argument('--output-dir', default=VOICE_DIR)
    return parser.parse_args()


def record(seconds):
    import sounddevice as sd
    print(f'Grabando {seconds:.0f} s. Habla con naturalidad, con pausas entre frases...')
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype='float32')
    sd.wait()
    return audio[:, 0]


def read_file(path):
    import soundfile as sf
    import torch
    import torchaudio
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    mono = torch.from_numpy(audio.mean(axis=1))
    if rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, rate, SAMPLE_RATE)
    return mono.numpy()


def normalized_mean(vectors):
    mean = np.mean(vectors, axis=0)
    return mean / np.linalg.norm(mean)


def drop_outliers(vectors):
    scores = [float(np.dot(v, normalized_mean(vectors[:i] + vectors[i + 1:])))
              for i, v in enumerate(vectors)]
    cutoff = float(np.median(scores)) - OUTLIER_MARGIN
    kept = [v for v, s in zip(vectors, scores) if s >= cutoff]
    return kept, scores, cutoff


def main():
    args = parse_args()
    if args.bags:
        audios = [read_bag_audio(f'{BAGS_DIR}/{bag}') for bag in args.bags]
    elif args.files:
        audios = [read_file(path) for path in args.files]
    else:
        audios = [record(args.record)]

    encoder = VoiceEncoder()
    vectors = [encoder.embed(audio[a:b]) for audio in audios for a, b in speech_spans(audio)]
    if len(vectors) < 3:
        print(f'Solo {len(vectors)} frase(s) de al menos 1 s; hacen falta 3.', file=sys.stderr)
        return 1

    kept, scores, cutoff = drop_outliers(vectors)
    print(f'{len(vectors)} frases, consistencia: {" ".join(f"{s:.2f}" for s in scores)}')
    print(f'Se descartan {len(vectors) - len(kept)} por debajo de {cutoff:.2f} '
          f'(ruido o voz ajena).')

    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f'{args.name}.npy', normalized_mean(kept))
    print(f'Guardado {output / args.name}.npy con {len(kept)} frases.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

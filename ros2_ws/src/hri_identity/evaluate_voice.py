#!/usr/bin/env python3
"""Mide la separacion de ECAPA entre la voz de una persona y voces de impostores, con validacion dejando un segmento fuera."""
import argparse
import itertools
import sys

import numpy as np

from hri_identity.voice import (SAMPLE_RATE, VoiceEncoder, load_voices, read_bag_audio,
                                speech_spans)

BAGS_DIR = '/data/bags'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person', required=True)
    parser.add_argument('--person-bags', nargs='+', required=True)
    parser.add_argument('--impostor-bags', nargs='+', required=True)
    parser.add_argument('--use-enrolled', action='store_true',
                        help='Compara contra la voz ya enrolada en vez de dejar un segmento fuera')
    return parser.parse_args()


def embed_bag(encoder, bag):
    audio = read_bag_audio(f'{BAGS_DIR}/{bag}')
    return [(a / SAMPLE_RATE, b / SAMPLE_RATE, encoder.embed(audio[a:b]))
            for a, b in speech_spans(audio)]


def describe(span, score):
    start, end, _ = span
    return f'{start:5.1f}-{end:5.1f} s -> {score:.2f}'


def normalized_mean(vectors):
    mean = np.mean(vectors, axis=0)
    return mean / np.linalg.norm(mean)


def main():
    args = parse_args()
    encoder = VoiceEncoder()

    genuine = list(itertools.chain.from_iterable(embed_bag(encoder, b) for b in args.person_bags))
    impostors = {b: embed_bag(encoder, b) for b in args.impostor_bags}
    if len(genuine) < 2 and not args.use_enrolled:
        print(f'Solo {len(genuine)} segmento(s) de {args.person}; hacen falta al menos 2.',
              file=sys.stderr)
        return 1

    vectors = [span[2] for span in genuine]
    genuine_scores = []
    if args.use_enrolled:
        template = load_voices()[args.person]
        print(f'{args.person}: {len(genuine)} segmentos contra la voz enrolada')
    else:
        template = normalized_mean(vectors)
        print(f'{args.person}: {len(genuine)} segmentos de voz (dejando uno fuera)')
    for i, span in enumerate(genuine):
        reference = template if args.use_enrolled else normalized_mean(vectors[:i] + vectors[i + 1:])
        score = float(np.dot(span[2], reference))
        genuine_scores.append(score)
        print(f'  {describe(span, score)}')
    impostor_scores = []
    for bag, spans in impostors.items():
        print(f'{bag}: {len(spans)} segmentos')
        for span in spans:
            score = float(np.dot(span[2], template))
            impostor_scores.append(score)
            print(f'  {describe(span, score)}')

    worst_genuine = min(genuine_scores)
    best_impostor = max(impostor_scores)
    print(f'\n  Peor genuino {worst_genuine:.3f}   mejor impostor {best_impostor:.3f}   '
          f'margen {worst_genuine - best_impostor:+.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

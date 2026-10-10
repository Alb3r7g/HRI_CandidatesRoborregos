#!/usr/bin/env python3
"""Grafica FAR/FRR por umbral y la distribucion de puntajes a partir de fusion_attempts.csv."""
import argparse
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GENUINE = '#2a78d6'
IMPOSTOR = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='/data/eval/fusion_attempts.csv')
    parser.add_argument('--out', default='/data/eval/far_frr.png')
    parser.add_argument('--high', type=float, default=None)
    parser.add_argument('--low', type=float, default=None)
    return parser.parse_args()


def kind(row):
    if row['real'] == 'genuino':
        return 'Genuino'
    if row['bag'].startswith('Ataque_'):
        return row['bag'].replace('Ataque_', 'Ataque ').lower().capitalize()
    return 'Otra persona'


def main():
    args = parse_args()
    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8'))
            if r['real'] in ('genuino', 'impostor')]
    genuine = np.array([float(r['puntaje']) for r in rows if r['real'] == 'genuino'])
    impostor = np.array([float(r['puntaje']) for r in rows if r['real'] == 'impostor'])
    thresholds = np.linspace(0, 1, 501)
    far = np.array([np.mean(impostor >= t) for t in thresholds]) * 100
    frr = np.array([np.mean(genuine < t) for t in thresholds]) * 100

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (curve, strip) = plt.subplots(1, 2, figsize=(12, 4.6),
                                       gridspec_kw={'width_ratios': [1.1, 1]})

    curve.step(thresholds, frr, where='post', color=GENUINE, lw=2,
               label=f'FRR (genuinos rechazados, n={len(genuine)})')
    curve.step(thresholds, far, where='post', color=IMPOSTOR, lw=2,
               label=f'FAR (impostores aceptados, n={len(impostor)})')
    for value, name in ((args.high, 'T_alto'), (args.low, 'T_bajo')):
        if value is not None:
            curve.axvline(value, color=INK, lw=1, ls='--')
            curve.text(value + 0.01, 96, name, color=INK, va='top')
    curve.set_xlim(0, 1)
    curve.set_ylim(0, 100)
    curve.set_xlabel('Umbral sobre el puntaje fusionado')
    curve.set_ylabel('Tasa de error (%)')
    curve.set_title('FAR y FRR por umbral', loc='left', color=INK)
    curve.grid(color=GRID, lw=0.8)
    curve.legend(frameon=False, loc='center left')
    for side in ('top', 'right'):
        curve.spines[side].set_visible(False)

    groups = ['Genuino', 'Otra persona', 'Ataque foto', 'Ataque audio', 'Ataque video']
    rng = np.random.default_rng(0)
    for i, group in enumerate(groups):
        scores = np.array([float(r['puntaje']) for r in rows if kind(r) == group])
        jitter = rng.uniform(-0.18, 0.18, len(scores))
        strip.scatter(scores, np.full(len(scores), i) + jitter, s=46,
                      color=GENUINE if group == 'Genuino' else IMPOSTOR,
                      edgecolor='white', linewidth=1.2, zorder=3)
    for value in (args.high, args.low):
        if value is not None:
            strip.axvline(value, color=INK, lw=1, ls='--')
    strip.set_yticks(range(len(groups)), groups)
    strip.invert_yaxis()
    strip.set_xlim(-0.03, 1.03)
    strip.set_xlabel('Puntaje fusionado (0 si no pasa la vivacidad)')
    strip.set_title('Cada intento', loc='left', color=INK)
    strip.grid(axis='x', color=GRID, lw=0.8)
    for side in ('top', 'right', 'left'):
        strip.spines[side].set_visible(False)
    strip.tick_params(axis='y', length=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f'{args.out}  ({len(genuine)} genuinos, {len(impostor)} impostores)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

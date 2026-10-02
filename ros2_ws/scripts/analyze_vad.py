#!/usr/bin/env python3
"""Analisis de /audio/voice_activity para calibrar los umbrales del VAD.

Recorre las probabilidades publicadas y reconstruye los disparos que produciria
cada umbral, sin necesidad de provocar el caso limite frente al microfono. Sirve
tanto en vivo como sobre un bag reproducido con ros2 bag play.

Uso (dentro del contenedor, con audio_capture_node y vad_node corriendo):
    python3 /ros2_ws/scripts/analyze_vad.py --seconds 30
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from hri_interfaces.msg import VoiceActivity

BLOCK_MS = 32.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=30.0,
                        help='Duracion de la captura (default: 30)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Umbral de entrada a evaluar (default: 0.5)')
    parser.add_argument('--hysteresis', type=float, default=0.15,
                        help='Margen de salida bajo el umbral (default: 0.15)')
    parser.add_argument('--min-speech-ms', type=float, default=200.0,
                        help='Duracion minima de voz configurada en el nodo (default: 200)')
    return parser.parse_args()


def collect(seconds):
    rclpy.init()
    node = Node('analyze_vad')
    samples = []
    node.create_subscription(
        VoiceActivity, '/audio/voice_activity',
        lambda m: samples.append(m.speech_probability), 200)

    print(f'Escuchando /audio/voice_activity durante {seconds:.0f} s...')
    deadline = node.get_clock().now().nanoseconds + int(seconds * 1e9)
    while node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    return np.asarray(samples, dtype=np.float32)


def find_bursts(probabilities, threshold, hysteresis):
    """Reconstruye los disparos aplicando la misma histeresis que el nodo."""
    bursts = []
    triggered = False
    start = 0
    for i, probability in enumerate(probabilities):
        if not triggered and probability >= threshold:
            triggered = True
            start = i
        elif triggered and probability < threshold - hysteresis:
            triggered = False
            bursts.append((start, i))
    if triggered:
        bursts.append((start, len(probabilities)))
    return bursts


def main():
    args = parse_args()
    probabilities = collect(args.seconds)
    if probabilities.size == 0:
        print('No llego ningun mensaje. Esta corriendo vad_node?', file=sys.stderr)
        sys.exit(1)

    print(f'\nBloques recibidos: {probabilities.size} '
          f'({probabilities.size * BLOCK_MS / 1000:.1f} s de audio)')

    # El piso de ruido indica cuanto margen real hay entre silencio y umbral: si
    # el percentil 99 ya roza el umbral, el entorno disparara falsos positivos.
    print('\n--- Distribucion de probabilidad')
    for percentile in (50, 90, 99, 100):
        print(f'  p{percentile:<3} {np.percentile(probabilities, percentile):.4f}')

    bursts = find_bursts(probabilities, args.threshold, args.hysteresis)
    print(f'\n--- Disparos con umbral {args.threshold:.2f} '
          f'(salida {args.threshold - args.hysteresis:.2f})')
    if not bursts:
        print('  ninguno')
    short = []
    for index, (start, end) in enumerate(bursts, start=1):
        duration_ms = (end - start) * BLOCK_MS
        peak = float(probabilities[start:end].max())
        flag = 'DESCARTADO' if duration_ms < args.min_speech_ms else 'aceptado'
        if duration_ms < args.min_speech_ms:
            short.append((duration_ms, peak))
        print(f'  #{index:<3} {duration_ms:7.0f} ms  prob. max {peak:.3f}  {flag}')

    print(f'\n--- Resumen con min_speech_ms={args.min_speech_ms:.0f}')
    print(f'  disparos totales:     {len(bursts)}')
    print(f'  bajo el minimo:       {len(short)}')
    if short:
        peaks = np.array([p for _, p in short])
        print(f'  su prob. max va de {peaks.min():.3f} a {peaks.max():.3f}')
        # Si los disparos cortos alcanzan probabilidades altas, subir el umbral no
        # los elimina y el filtro por duracion es el unico que los ataja.
        print('  subir el umbral por encima de '
              f'{peaks.max():.3f} los eliminaria sin necesidad del filtro')


if __name__ == '__main__':
    main()

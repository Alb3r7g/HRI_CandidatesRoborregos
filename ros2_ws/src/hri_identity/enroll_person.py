#!/usr/bin/env python3
"""Script standalone (no es un nodo ROS) para enrolar una persona.

Uso:
    python3 enroll_person.py --name albert --device 0
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

DEFAULT_OUTPUT_DIR = '/data/enrolled'
DEFAULT_NUM_SAMPLES = 20
MAX_ATTEMPTS = DEFAULT_NUM_SAMPLES * 20
CAPTURE_INTERVAL_S = 0.3


def parse_args():
    parser = argparse.ArgumentParser(description='Enrola una persona capturando su embedding facial promedio.')
    parser.add_argument('--name', required=True, help='Nombre de la persona (define el archivo <name>.npy)')
    parser.add_argument('--device', type=int, default=0, help='Índice de la cámara (default: 0)')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
                         help=f'Cantidad de frames con cara detectada a promediar (default: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                         help=f'Directorio de salida (default: {DEFAULT_OUTPUT_DIR})')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Cargando InsightFace (buffalo_l)...')
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f'No se pudo abrir la cámara (device={args.device})', file=sys.stderr)
        sys.exit(1)

    embeddings = []
    print(f'Capturando hasta {args.num_samples} muestras. Mira a la cámara...')

    # Sin ventana (cv2.imshow requiere X11, no disponible en el contenedor):
    # el feedback es por terminal, con un límite de intentos para no
    # colgarse indefinidamente si nunca se detecta una cara.
    attempts = 0
    while len(embeddings) < args.num_samples and attempts < MAX_ATTEMPTS:
        attempts += 1
        ret, frame = capture.read()
        if not ret:
            print('Frame no recibido de la cámara', file=sys.stderr)
            continue

        faces = app.get(frame)
        if not faces:
            if attempts % 10 == 0:
                print('  sin cara detectada, sigue mirando a la cámara...')
        else:
            # si hay varias caras en el frame, toma la de mayor área de bbox
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(face.normed_embedding)
            print(f'  muestra {len(embeddings)}/{args.num_samples} capturada')
            time.sleep(CAPTURE_INTERVAL_S)

    capture.release()

    if not embeddings:
        print('No se capturó ninguna muestra válida.', file=sys.stderr)
        sys.exit(1)

    if len(embeddings) < args.num_samples:
        print(f'Aviso: solo se lograron {len(embeddings)}/{args.num_samples} muestras tras {attempts} intentos.')

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.name}.npy')
    np.save(output_path, mean_embedding)
    print(f'Embedding de "{args.name}" guardado en {output_path} ({len(embeddings)} muestras promediadas)')


if __name__ == '__main__':
    main()

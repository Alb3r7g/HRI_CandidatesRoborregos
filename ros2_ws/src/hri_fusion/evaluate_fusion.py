#!/usr/bin/env python3
"""Reproduce bags por cara, boca, voz y transcripcion, aplica la fusion y mide FAR/FRR por umbral."""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from hri_fusion.fusion import Utterance, pick_speaker

BAGS_DIR = '/data/bags'
CACHE = '/data/eval/fusion_features.json'
CSV_OUT = '/data/eval/fusion_attempts.csv'
FRAME_STRIDE = 2
TRACK_GATE = 0.6
VAD = {'threshold': 0.5, 'min_silence_duration_ms': 750,
       'speech_pad_ms': 200, 'min_speech_duration_ms': 200}
GROUP_QUESTION = {'Grupo_01': 15.0}
MANUAL_LABELS = {
    ('Albert_02', 0.0): 'ruido',
    ('Grupo_01', 18.9): 'genuino',
}
DEFAULT_BAGS = ['Albert_02', 'Albert_Tablet_01', 'Albert_Tablet_02', 'Desconocido_01',
                'Desconocido_02', 'Grupo_01', 'Ataque_Foto', 'Ataque_Video', 'Ataque_Audio']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bags', nargs='+', default=DEFAULT_BAGS)
    parser.add_argument('--person', default='Albert')
    parser.add_argument('--refresh', action='store_true',
                        help='Recalcula las senales aunque existan en cache')
    return parser.parse_args()


def label(bag, start, end, person):
    manual = MANUAL_LABELS.get((bag, round(start, 1)))
    if manual:
        return manual
    if bag in GROUP_QUESTION:
        return 'genuino' if start <= GROUP_QUESTION[bag] <= end else 'impostor'
    return 'genuino' if bag.startswith(f'{person}_') else 'impostor'


def read_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from hri_interfaces.msg import AudioChunk
    from sensor_msgs.msg import Image
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path), rosbag2_py.ConverterOptions('', ''))
    start, audio, frames, count = None, [], [], 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        start = stamp if start is None else start
        if topic == '/audio/raw':
            audio.append(np.asarray(deserialize_message(data, AudioChunk).samples,
                                    dtype=np.float32))
        elif topic == '/camera/image_raw':
            count += 1
            if count % FRAME_STRIDE:
                continue
            msg = deserialize_message(data, Image)
            pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            image = pixels[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                image = image[:, :, ::-1]
            frames.append(((stamp - start) / 1e9, np.ascontiguousarray(image)))
    return frames, np.concatenate(audio)


class Tracker:

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def assign(self, box):
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        width = box[2] - box[0]
        best, best_distance = None, TRACK_GATE * width
        for track, previous in self.tracks.items():
            distance = np.linalg.norm(center - previous)
            if distance < best_distance:
                best, best_distance = track, distance
        if best is None:
            best, self.next_id = self.next_id, self.next_id + 1
        self.tracks[best] = center
        return best


def extract(bag, models):
    face_app, encoder, whisper, vad = models
    from hri_identity.voice import SAMPLE_RATE
    from silero_vad import get_speech_timestamps
    import torch
    frames, audio = read_bag(f'{BAGS_DIR}/{bag}')
    enrolled = {os.path.basename(p)[:-4]: np.load(p)
                for p in glob.glob('/data/enrolled/*.npy')}
    tracker = Tracker()
    observations = []
    for t, image in frames:
        for face in face_app.get(image):
            lm = face.landmark_3d_68
            width = np.linalg.norm(lm[60, :2] - lm[64, :2])
            observations.append({
                't': t, 'track': tracker.assign(face.bbox),
                'sims': {k: float(np.dot(face.normed_embedding, v / np.linalg.norm(v)))
                         for k, v in enrolled.items()},
                'mouth': float(np.linalg.norm(lm[62, :2] - lm[66, :2]) / max(width, 1e-6))})
    spans = get_speech_timestamps(torch.as_tensor(audio), vad, sampling_rate=SAMPLE_RATE,
                                  **VAD)
    utterances = []
    for span in spans:
        samples = audio[span['start']:span['end']]
        segments, _ = whisper.transcribe(samples, language='es', beam_size=5,
                                         condition_on_previous_text=False)
        text = ' '.join(s.text.strip() for s in segments).strip()
        voice = encoder.embed(samples).tolist() if len(samples) >= SAMPLE_RATE else None
        utterances.append({'start': span['start'] / SAMPLE_RATE,
                           'end': span['end'] / SAMPLE_RATE, 'text': text, 'voice': voice})
    return {'observations': observations, 'utterances': utterances}


def load_models():
    from faster_whisper import WhisperModel
    from insightface.app import FaceAnalysis
    from silero_vad import load_silero_vad
    from hri_identity.voice import VoiceEncoder
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'],
                            allowed_modules=['detection', 'recognition', 'landmark_3d_68'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    whisper = WhisperModel('small', device='cuda', compute_type='int8_float16',
                           download_root='/data/models/whisper')
    return face_app, VoiceEncoder(), whisper, load_silero_vad()


def attempts(bag, features, voices, person):
    rows = []
    for utt in features['utterances']:
        if not utt['text']:
            continue
        window = [o for o in features['observations']
                  if utt['start'] - 0.2 <= o['t'] <= utt['end'] + 0.2]
        by_track = {}
        for o in window:
            by_track.setdefault(o['track'], []).append(o)
        speaker, activity = pick_speaker({t: [o['mouth'] for o in obs]
                                          for t, obs in by_track.items()})
        identity, face, voice = '', None, None
        if speaker is not None:
            obs = by_track[speaker]
            names = obs[0]['sims'].keys()
            medians = {n: float(np.median([o['sims'][n] for o in obs])) for n in names}
            identity = max(medians, key=medians.get)
            face = medians[identity]
            if utt['voice'] is not None and identity in voices:
                voice = float(np.dot(np.asarray(utt['voice']), voices[identity]))
        u = Utterance(identity, face, voice, activity if speaker is not None else None)
        rows.append({'bag': bag, 'inicio': round(utt['start'], 2), 'fin': round(utt['end'], 2),
                     'real': label(bag, utt['start'], utt['end'], person),
                     'texto': utt['text'], 'caras_en_cuadro': len(by_track),
                     'identidad': identity,
                     'cara': None if face is None else round(face, 3),
                     'voz': None if voice is None else round(voice, 3),
                     'boca': round(activity, 4),
                     'vivo': u.live,
                     'confianza': round(u.confidence if identity else 0.0, 3),
                     'puntaje': round(u.score if identity else 0.0, 3)})
    return rows


def main():
    args = parse_args()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    missing = [b for b in args.bags
               if (args.refresh or b not in cache) and os.path.isdir(f'{BAGS_DIR}/{b}')]
    if missing:
        models = load_models()
        for bag in missing:
            print(f'Extrayendo {bag}...', file=sys.stderr)
            cache[bag] = extract(bag, models)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(cache, open(CACHE, 'w'))

    from hri_identity.voice import load_voices
    voices = load_voices()
    rows = [r for b in args.bags if b in cache for r in attempts(b, cache[b], voices, args.person)]
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['bag']:<17}{r['inicio']:6.1f}-{r['fin']:5.1f} {r['real']:<9}"
              f"caras {r['caras_en_cuadro']}  cara {r['cara']!s:<6} voz {r['voz']!s:<6} "
              f"boca {r['boca']:<7} {'vivo' if r['vivo'] else 'QUIETO'} C {r['confianza']:.2f}  \"{r['texto'][:40]}\"")
    genuine = np.array([r['puntaje'] for r in rows if r['real'] == 'genuino'])
    impostor = np.array([r['puntaje'] for r in rows if r['real'] == 'impostor'])
    print(f'\nIntentos: {len(genuine)} genuinos, {len(impostor)} impostores   -> {CSV_OUT}')
    print('Umbral  FRR      FAR')
    for t in np.arange(0.30, 0.91, 0.05):
        print(f'  {t:.2f}  {np.mean(genuine < t):6.1%}  {np.mean(impostor >= t):6.1%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

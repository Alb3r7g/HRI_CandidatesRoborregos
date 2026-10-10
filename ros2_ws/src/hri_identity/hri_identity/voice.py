import pathlib

import numpy as np
import torch

SAMPLE_RATE = 16000
ECAPA_SOURCE = 'speechbrain/spkrec-ecapa-voxceleb'
ECAPA_DIR = '/data/models/ecapa'
VOICE_DIR = '/data/enrolled/voice'
MIN_SPEECH_S = 1.0


class VoiceEncoder:

    def __init__(self, device='cuda:0'):
        from speechbrain.inference.speaker import EncoderClassifier
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, savedir=ECAPA_DIR, run_opts={'device': device})

    def embed(self, samples):
        wave = torch.as_tensor(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            vector = self.model.encode_batch(wave.to(self.device)).squeeze().cpu().numpy()
        return vector / np.linalg.norm(vector)


def load_voices(directory=VOICE_DIR):
    return {p.stem: np.load(p) for p in sorted(pathlib.Path(directory).glob('*.npy'))}


def best_match(embedding, voices):
    if not voices:
        return '', 0.0
    name, score = max(((n, float(np.dot(embedding, v))) for n, v in voices.items()),
                      key=lambda item: item[1])
    return name, score


def speech_spans(samples, min_silence_ms=750, speech_pad_ms=200):
    from silero_vad import get_speech_timestamps, load_silero_vad
    spans = get_speech_timestamps(
        torch.as_tensor(samples), load_silero_vad(), sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
    return [(s['start'], s['end']) for s in spans
            if (s['end'] - s['start']) / SAMPLE_RATE >= MIN_SPEECH_S]


def read_bag_audio(path, topic='/audio/raw'):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from hri_interfaces.msg import AudioChunk
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path)), rosbag2_py.ConverterOptions('', ''))
    blocks = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == topic:
            blocks.append(np.asarray(deserialize_message(data, AudioChunk).samples,
                                     dtype=np.float32))
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

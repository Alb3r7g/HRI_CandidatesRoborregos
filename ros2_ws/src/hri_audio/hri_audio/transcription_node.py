import math
import pathlib

import numpy as np
import rclpy
from faster_whisper import WhisperModel
from rclpy.node import Node

from hri_interfaces.msg import SpeechSegment, Transcript

WHISPER_SAMPLE_RATE = 16000
MODELS_DIR = '/data/models/whisper'
DEFAULT_PROMPT = 'Conversación en español de México entre personas y un robot de servicio.'


class TranscriptionNode(Node):

    def __init__(self):
        super().__init__('transcription_node')

        self.declare_parameter('model_size', 'small')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('compute_type', 'int8_float16')
        self.declare_parameter('language', 'es')
        self.declare_parameter('beam_size', 5)
        self.declare_parameter('initial_prompt', DEFAULT_PROMPT)
        self.declare_parameter('enrolled_dir', '/data/enrolled')

        self.language = self.get_parameter('language').value
        self.beam_size = self.get_parameter('beam_size').value
        self.prompt = self.build_prompt(
            self.get_parameter('initial_prompt').value,
            self.get_parameter('enrolled_dir').value)
        self.model = self.load_model(
            self.get_parameter('model_size').value,
            self.get_parameter('device').value,
            self.get_parameter('compute_type').value)

        self.publisher_ = self.create_publisher(Transcript, '/audio/transcript', 10)
        self.create_subscription(
            SpeechSegment, '/audio/speech_segment', self.segment_callback, 10)
        self.get_logger().info(f'Prompt de contexto: "{self.prompt}"')

    def build_prompt(self, base, enrolled_dir):
        names = sorted(p.stem for p in pathlib.Path(enrolled_dir).glob('*.npy'))
        if not names:
            return base
        return f'{base} Nombres: {", ".join(names)}.'

    def load_model(self, size, device, compute_type):
        try:
            model = WhisperModel(size, device=device, compute_type=compute_type,
                                 download_root=MODELS_DIR)
            self.get_logger().info(f'Whisper {size} en {device} ({compute_type})')
            return model
        except (RuntimeError, ValueError) as error:
            self.get_logger().warn(f'Whisper no pudo cargar en {device}: {error}. Usando CPU.')
            return WhisperModel(size, device='cpu', compute_type='int8',
                                download_root=MODELS_DIR)

    def segment_callback(self, msg):
        if msg.sample_rate != WHISPER_SAMPLE_RATE or msg.channels != 1:
            self.get_logger().warn(
                f'Segmento descartado: {msg.sample_rate} Hz, {msg.channels} canales; '
                f'se espera {WHISPER_SAMPLE_RATE} Hz mono.')
            return

        audio = np.asarray(msg.samples, dtype=np.float32)
        segments, info = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam_size,
            initial_prompt=self.prompt, condition_on_previous_text=False)
        segments = list(segments)
        text = ' '.join(s.text.strip() for s in segments).strip()
        if not text:
            return

        transcript = Transcript()
        transcript.header = msg.header
        transcript.duration = len(audio) / WHISPER_SAMPLE_RATE
        transcript.text = text
        transcript.language = info.language
        transcript.confidence = math.exp(
            sum(s.avg_logprob for s in segments) / len(segments))
        self.publisher_.publish(transcript)
        self.get_logger().info(
            f'[{transcript.duration:.1f} s, conf {transcript.confidence:.2f}] {text}')


def main(args=None):
    rclpy.init(args=args)
    node = TranscriptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

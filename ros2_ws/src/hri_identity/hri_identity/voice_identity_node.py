import numpy as np
import rclpy
from rclpy.node import Node

from hri_identity.voice import (MIN_SPEECH_S, SAMPLE_RATE, VOICE_DIR, VoiceEncoder,
                                best_match, load_voices)
from hri_interfaces.msg import SpeechSegment, VoiceIdentity


class VoiceIdentityNode(Node):

    def __init__(self):
        super().__init__('voice_identity_node')

        self.declare_parameter('similarity_threshold', 0.45)
        self.declare_parameter('voice_dir', VOICE_DIR)
        self.declare_parameter('device', 'cuda:0')

        self.voices = load_voices(self.get_parameter('voice_dir').value)
        self.encoder = VoiceEncoder(self.get_parameter('device').value)

        self.publisher_ = self.create_publisher(VoiceIdentity, '/audio/voice_identity', 10)
        self.create_subscription(
            SpeechSegment, '/audio/speech_segment', self.segment_callback, 10)
        self.get_logger().info(
            f'Voces enroladas: {", ".join(self.voices) or "ninguna"}   umbral PROVISIONAL: '
            f'{self.threshold():.2f}   dispositivo: {self.encoder.device}')

    def threshold(self):
        return self.get_parameter('similarity_threshold').value

    def segment_callback(self, msg):
        if msg.sample_rate != SAMPLE_RATE or msg.channels != 1:
            self.get_logger().warn(
                f'Segmento descartado: {msg.sample_rate} Hz, {msg.channels} canales.')
            return
        duration = len(msg.samples) / SAMPLE_RATE
        if duration < MIN_SPEECH_S:
            return

        embedding = self.encoder.embed(np.asarray(msg.samples, dtype=np.float32))
        name, similarity = best_match(embedding, self.voices)
        known = bool(name) and similarity >= self.threshold()

        result = VoiceIdentity()
        result.header = msg.header
        result.duration = duration
        result.identity_name = name if known else 'desconocido'
        result.similarity = similarity
        result.is_known = known
        self.publisher_.publish(result)
        self.get_logger().info(
            f'[{duration:.1f} s] {result.identity_name} '
            f'(mas parecido: {name or "-"} {similarity:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceIdentityNode()
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

import collections

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from silero_vad import load_silero_vad

from hri_interfaces.msg import AudioChunk, SpeechSegment, VoiceActivity

# Silero v5 solo acepta ventanas de 512 muestras a 16 kHz.
SILERO_WINDOW_SAMPLES = 512
SILERO_SAMPLE_RATE = 16000


class VadNode(Node):

    def __init__(self):
        super().__init__('vad_node')

        self.declare_parameter('threshold', 0.5)
        # Silero baja el umbral 0.15 para salir del estado de habla; se replica ese
        # margen para que una probabilidad oscilando alrededor del umbral no
        # produzca decenas de inicios y finales por segundo.
        self.declare_parameter('hysteresis', 0.15)
        # 750 ms tolera las pausas de una frase hablada con naturalidad. El valor
        # por defecto de Silero (100 ms) corta al respirar a media pregunta.
        self.declare_parameter('min_silence_ms', 750)
        # Descarta disparos de una tos o un golpe en la mesa antes de lanzar una
        # transcripción sobre ruido.
        self.declare_parameter('min_speech_ms', 200)
        # El habla ya empezó cuando la probabilidad cruza el umbral: sin este
        # margen previo el segmento pierde el primer fonema y la transcripción
        # se come el inicio de la palabra. También se añade al final.
        self.declare_parameter('speech_pad_ms', 200)

        self.threshold = self.get_parameter('threshold').value
        self.hysteresis = self.get_parameter('hysteresis').value
        min_silence_ms = self.get_parameter('min_silence_ms').value
        min_speech_ms = self.get_parameter('min_speech_ms').value
        speech_pad_ms = self.get_parameter('speech_pad_ms').value

        self.min_silence_samples = SILERO_SAMPLE_RATE * min_silence_ms // 1000
        self.min_speech_samples = SILERO_SAMPLE_RATE * min_speech_ms // 1000
        pad_samples = SILERO_SAMPLE_RATE * speech_pad_ms // 1000
        pad_blocks = max(1, pad_samples // SILERO_WINDOW_SAMPLES)

        # El modelo corre en CPU a propósito: son 31 ventanas por segundo sobre una
        # red diminuta, y la VRAM de la GPU queda libre para YOLO, InsightFace,
        # Whisper y el modelo de lenguaje, que sí la necesitan.
        self.model = load_silero_vad()

        # Cola circular con los últimos bloques, para anteponerlos al segmento
        # cuando el habla se detecta con retraso.
        self.pre_roll = collections.deque(maxlen=pad_blocks)
        self.pad_blocks = pad_blocks

        self.triggered = False
        self.buffer = []
        self.segment_stamp = None
        self.silence_run = 0
        self.speech_samples = 0
        self.trailing_blocks = 0
        self.peak_probability = 0.0

        self.activity_pub = self.create_publisher(VoiceActivity, '/audio/voice_activity', 10)
        self.segment_pub = self.create_publisher(SpeechSegment, '/audio/speech_segment', 10)
        self.create_subscription(AudioChunk, '/audio/raw', self.audio_callback, 50)

        self.get_logger().info(
            f'VAD listo: umbral {self.threshold:.2f} (salida {self.threshold - self.hysteresis:.2f}), '
            f'silencio {min_silence_ms} ms, voz mínima {min_speech_ms} ms, margen {speech_pad_ms} ms')

    def audio_callback(self, msg):
        if msg.sample_rate != SILERO_SAMPLE_RATE:
            self.get_logger().warn(
                f'Se esperaban {SILERO_SAMPLE_RATE} Hz y llegaron {msg.sample_rate}; bloque descartado',
                throttle_duration_sec=10.0)
            return

        block = np.asarray(msg.samples, dtype=np.float32)
        if block.size != SILERO_WINDOW_SAMPLES:
            self.get_logger().warn(
                f'Silero exige ventanas de {SILERO_WINDOW_SAMPLES} muestras y llegaron {block.size}; '
                'ajusta block_size en audio_capture_node',
                throttle_duration_sec=10.0)
            return

        # El modelo es recurrente: cada llamada avanza su estado interno, así que
        # se invoca una sola vez por bloque y la probabilidad se reutiliza para
        # publicar y para decidir.
        with torch.no_grad():
            probability = float(self.model(torch.from_numpy(block), SILERO_SAMPLE_RATE).item())

        activity = VoiceActivity()
        activity.header = msg.header
        activity.speech_probability = probability
        activity.is_speech = self.triggered or probability >= self.threshold
        self.activity_pub.publish(activity)

        self._update_segmentation(msg, block, probability)

    def _update_segmentation(self, msg, block, probability):
        if not self.triggered:
            self.pre_roll.append((msg.header, block))
            if probability >= self.threshold:
                self.triggered = True
                # El segmento arranca en el bloque más viejo del pre-roll, no en el
                # actual, y por eso hereda también su timestamp.
                self.segment_stamp = self.pre_roll[0][0].stamp
                self.buffer = [b for _, b in self.pre_roll]
                self.speech_samples = block.size
                self.silence_run = 0
                self.trailing_blocks = 0
                self.peak_probability = probability
                self.pre_roll.clear()
            return

        self.buffer.append(block)
        self.peak_probability = max(self.peak_probability, probability)

        if probability >= self.threshold - self.hysteresis:
            self.speech_samples += block.size
            self.silence_run = 0
            self.trailing_blocks = 0
            return

        self.silence_run += block.size
        self.trailing_blocks += 1
        if self.silence_run < self.min_silence_samples:
            return

        self._close_segment(msg)

    def _close_segment(self, msg):
        # Se recorta el silencio final salvo el margen: dejarlo entero alarga el
        # audio que recibe la transcripción sin aportar contenido.
        keep = max(0, len(self.buffer) - self.trailing_blocks + self.pad_blocks)
        samples = np.concatenate(self.buffer[:keep]) if keep else np.empty(0, dtype=np.float32)

        if self.speech_samples >= self.min_speech_samples and samples.size:
            segment = SpeechSegment()
            segment.header.stamp = self.segment_stamp
            segment.header.frame_id = msg.header.frame_id
            segment.samples = samples.tolist()
            segment.sample_rate = SILERO_SAMPLE_RATE
            segment.channels = msg.channels
            self.segment_pub.publish(segment)
            self.get_logger().info(
                f'Segmento de voz: {samples.size / SILERO_SAMPLE_RATE:.2f} s '
                f'(prob. máx {self.peak_probability:.2f})')
        else:
            # A nivel info y no debug: cuántos disparos se descartan, y con qué
            # probabilidad pico, es el dato que permite recalibrar min_speech_ms
            # y el umbral sin volver a grabar.
            self.get_logger().info(
                f'Disparo descartado: {self.speech_samples / SILERO_SAMPLE_RATE:.2f} s de voz '
                f'(mínimo {self.min_speech_samples / SILERO_SAMPLE_RATE:.2f} s, '
                f'prob. máx {self.peak_probability:.2f})')

        self.triggered = False
        self.buffer = []
        self.segment_stamp = None
        self.silence_run = 0
        self.speech_samples = 0
        self.trailing_blocks = 0
        self.peak_probability = 0.0
        self.pre_roll.clear()


def main(args=None):
    rclpy.init(args=args)
    node = VadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy instala su propio manejador de SIGINT y puede haber cerrado ya el
        # contexto: un segundo shutdown() lanza RCLError y el proceso termina con
        # codigo 1, lo que haria fallar a cualquier script que compruebe la salida.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

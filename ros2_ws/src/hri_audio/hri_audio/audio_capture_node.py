import queue

import rclpy
import sounddevice as sd
from rclpy.node import Node

from hri_interfaces.msg import AudioChunk


class AudioCaptureNode(Node):

    def __init__(self):
        super().__init__('audio_capture_node')

        # 16 kHz mono es el formato que esperan Silero VAD, faster-whisper y
        # ECAPA-TDNN; publicar en otro formato obligaria a remuestrear en cada
        # nodo consumidor. PortAudio remuestrea desde los 48 kHz del micrófono.
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        # 512 muestras a 16 kHz son 32 ms: el tamaño de ventana que Silero VAD
        # exige, y aproximadamente un frame de cámara a 30 fps, lo que simplifica
        # asociar voz con cara en la fusión.
        self.declare_parameter('block_size', 512)
        self.declare_parameter('device', -1)

        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.block_size = self.get_parameter('block_size').value
        device = self.get_parameter('device').value

        self.publisher_ = self.create_publisher(AudioChunk, '/audio/raw', 10)

        # El callback de PortAudio corre en un hilo de tiempo real: si se bloquea
        # se pierden muestras. Solo encola; la publicación ocurre en el timer.
        self.chunk_queue = queue.Queue()
        self.dropped_blocks = 0

        self.stream = None
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype='float32',
                device=None if device < 0 else device,
                callback=self._audio_callback)
            self.stream.start()
            self.get_logger().info(
                f'Capturando a {self.sample_rate} Hz, {self.channels} canal(es), '
                f'bloques de {self.block_size} muestras '
                f'({1000.0 * self.block_size / self.sample_rate:.0f} ms)')
        except Exception as exc:
            self.get_logger().error(f'No se pudo abrir el dispositivo de audio: {exc}')

        # Se drena la cola completa en cada tick, a mayor frecuencia que la de
        # llegada de bloques, para que la latencia no dependa del periodo del timer.
        block_period = self.block_size / float(self.sample_rate)
        self.timer = self.create_timer(block_period / 2.0, self.timer_callback)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.dropped_blocks += 1
        # stamp del inicio del bloque: el callback se dispara cuando el bloque ya
        # está completo, así que el sonido ocurrió block_size/sample_rate antes.
        now = self.get_clock().now()
        block_duration_ns = int(1e9 * frames / self.sample_rate)
        start_stamp = rclpy.time.Time(nanoseconds=now.nanoseconds - block_duration_ns)
        self.chunk_queue.put((start_stamp, indata.copy()))

    def timer_callback(self):
        while True:
            try:
                start_stamp, block = self.chunk_queue.get_nowait()
            except queue.Empty:
                return

            msg = AudioChunk()
            msg.header.stamp = start_stamp.to_msg()
            msg.header.frame_id = 'microphone'
            msg.samples = block.reshape(-1).tolist()
            msg.sample_rate = self.sample_rate
            msg.channels = self.channels
            self.publisher_.publish(msg)

    def destroy_node(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.dropped_blocks:
            self.get_logger().warn(
                f'PortAudio reportó {self.dropped_blocks} bloque(s) con underrun/overflow')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
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

import queue

import numpy as np
import rclpy
import sounddevice as sd
from hri_interfaces.msg import AudioChunk

RATE = 16000
buffer = queue.Queue()


def on_chunk(msg):
    buffer.put(np.asarray(msg.samples, dtype=np.float32))


def callback(outdata, frames, time_info, status):
    out = np.zeros(frames, dtype=np.float32)
    filled = 0
    while filled < frames:
        try:
            block = buffer.queue[0]
        except IndexError:
            break
        take = min(frames - filled, len(block))
        out[filled:filled + take] = block[:take]
        filled += take
        if take == len(block):
            buffer.get_nowait()
        else:
            buffer.queue[0] = block[take:]
    outdata[:, 0] = out


rclpy.init()
node = rclpy.create_node('bag_audio_player')
node.create_subscription(AudioChunk, '/audio/raw', on_chunk, 100)
print('Reproduciendo /audio/raw por las bocinas. Ctrl+C para salir.')
with sd.OutputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=512,
                     callback=callback):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

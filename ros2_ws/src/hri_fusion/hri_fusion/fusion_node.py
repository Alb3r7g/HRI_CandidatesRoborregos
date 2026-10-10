import textwrap
import time
from functools import partial
import unicodedata
from collections import Counter, deque, namedtuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from hri_fusion.fusion import (ANSWER, ASK, HIGH_THRESHOLD, LIVENESS_MIN, LOW_THRESHOLD,
                               MARGIN, REASK_TEXT, AccessPolicy, Utterance,
                               fused_confidence, pick_speaker)
from hri_interfaces.msg import DialogTurn, IdentifiedFaceArray, Transcript, VoiceIdentity
from hri_interfaces.srv import QueryDocument

UNKNOWN = 'desconocido'
DENIED_TEXT = 'Lo siento, no tienes acceso, así que no puedo consultar ningún documento.'
UNAVAILABLE_TEXT = 'Ahora mismo no puedo consultar los documentos. Intenta de nuevo en un momento.'
FACE_HISTORY_S = 30.0
VOICE_WAIT_S = 1.5
MIN_VOICE_S = 1.0
ORPHAN_S = 10.0
OVERLAY_FRESH_S = 0.5
SPEAKER_SHOWN_S = 3.0

Observation = namedtuple('Observation', 't track name score mouth box')


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def ascii_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


class FusionNode(Node):

    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter('high_threshold', HIGH_THRESHOLD)
        self.declare_parameter('low_threshold', LOW_THRESHOLD)
        self.declare_parameter('margin', MARGIN)
        self.declare_parameter('liveness_min', LIVENESS_MIN)
        self.declare_parameter('window_pad_s', 0.3)
        self.declare_parameter('face_absent_s', 2.0)
        self.declare_parameter('revoke_after_s', 1.0)
        self.declare_parameter('reask_window_s', 20.0)

        self.policy = AccessPolicy(self.param('high_threshold'), self.param('low_threshold'),
                                   self.param('margin'))
        self.faces = deque()
        self.latest_faces = None
        self.pending = {}
        self.session = {}
        self.asked_at = {}
        self.last = {'track': None, 'identity': '', 'confidence': 0.0, 'decision': '',
                     'answer': '', 'when': 0.0}
        self.bridge = CvBridge()

        self.response_pub = self.create_publisher(String, '/hri/response', 10)
        self.overlay_pub = self.create_publisher(Image, '/fusion/overlay_image', 10)
        self.dialog_pub = self.create_publisher(DialogTurn, '/hri/dialog', 10)
        self.create_subscription(IdentifiedFaceArray, '/vision/identified_faces',
                                 self.faces_callback, 10)
        self.create_subscription(Transcript, '/audio/transcript', self.transcript_callback, 10)
        self.create_subscription(VoiceIdentity, '/audio/voice_identity', self.voice_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.rag = self.create_client(QueryDocument, '/rag/query')
        self.create_timer(0.1, self.tick)

        self.get_logger().info(
            f'Cortes: responder >= {self.policy.high:.2f}, re-preguntar >= {self.policy.low:.2f}, '
            f'margen {self.policy.margin:.2f}, vivacidad >= {self.param("liveness_min"):.3f}')

    def param(self, name):
        return self.get_parameter(name).value

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def faces_callback(self, msg):
        t = seconds(msg.header.stamp)
        if self.faces and t < self.faces[-1].t - 1.0:
            self.faces.clear()
        for f in msg.faces:
            box = (f.face.bbox_x, f.face.bbox_y, f.face.bbox_width, f.face.bbox_height)
            self.faces.append(Observation(t, f.face.track_id, f.identity_name,
                                          f.identity_confidence, f.mouth_open, box))
        while self.faces and self.faces[0].t < t - FACE_HISTORY_S:
            self.faces.popleft()
        self.latest_faces = (time.monotonic(), msg)

    def entry(self, header):
        key = (header.stamp.sec, header.stamp.nanosec)
        return self.pending.setdefault(
            key, {'transcript': None, 'voice': None, 'arrived': time.monotonic()})

    def transcript_callback(self, msg):
        self.entry(msg.header)['transcript'] = msg

    def voice_callback(self, msg):
        self.entry(msg.header)['voice'] = msg

    def tick(self):
        now = time.monotonic()
        for key, e in list(self.pending.items()):
            waited = now - e['arrived']
            if e['transcript'] is None:
                if waited > ORPHAN_S:
                    del self.pending[key]
                continue
            if (e['voice'] is not None or e['transcript'].duration < MIN_VOICE_S
                    or waited > VOICE_WAIT_S):
                del self.pending[key]
                self.process_utterance(e['transcript'], e['voice'])
        self.monitor_access()

    def speaker(self, start, end):
        pad = self.param('window_pad_s')
        by_track = {}
        for o in self.faces:
            if start - pad <= o.t <= end + pad:
                by_track.setdefault(o.track, []).append(o)
        track, activity = pick_speaker({k: [o.mouth for o in v] for k, v in by_track.items()})
        if track is None:
            return None, '', None, None
        names = [o.name for o in by_track[track] if o.name != UNKNOWN]
        if not names:
            return track, '', None, activity
        identity = Counter(names).most_common(1)[0][0]
        face = float(np.median([o.score for o in by_track[track] if o.name == identity]))
        return track, identity, face, activity

    def process_utterance(self, transcript, voice_msg):
        start = seconds(transcript.header.stamp)
        track, identity, face, activity = self.speaker(start, start + transcript.duration)
        voice = None
        if identity and voice_msg is not None and voice_msg.identity_name == identity:
            voice = float(voice_msg.similarity)
        utterance = Utterance(identity, face, voice, activity, self.param('liveness_min'))
        confidence = utterance.confidence if identity else 0.0
        self.expire_asked()
        if identity in self.policy.granted and self.session.get(identity, {}).get('track') != track:
            self.revoke(identity, f'ahora habla otro track ({track})')
        was_granted = identity in self.policy.granted
        was_asked = identity in self.policy.asked
        decision = self.policy.decide(identity, confidence, utterance.live)
        if decision == ASK and identity:
            self.asked_at[identity] = time.monotonic()

        self.get_logger().info(
            f'"{transcript.text}" -> track {track} {identity or "-"}  cara {face or 0:.2f}  '
            f'voz {voice or 0:.2f}  boca {activity or 0:.3f} '
            f'{"vivo" if utterance.live else "QUIETO"}  C {confidence:.2f} -> {decision}')
        self.last.update(track=track, identity=identity, confidence=confidence,
                         decision=decision, when=time.monotonic())

        turn = DialogTurn()
        turn.header = transcript.header
        turn.question = transcript.text
        turn.identity = identity
        turn.face = float(face or 0.0)
        turn.voice = float(voice or 0.0)
        turn.mouth = float(activity or 0.0)
        turn.live = utterance.live
        turn.confidence = float(confidence)
        turn.decision = decision
        turn.reason = self.reason(decision, identity, utterance.live, was_granted, was_asked)

        if decision == ANSWER:
            self.session[identity] = {'track': track, 'voice': voice, 'mouth': activity,
                                      'confidence': confidence, 'below_since': None}
            self.query(identity, confidence, transcript.text, turn)
        elif decision == ASK:
            self.say(REASK_TEXT, turn)
        else:
            self.session.pop(identity, None)
            self.say(DENIED_TEXT, turn)

    def reason(self, decision, identity, live, was_granted, was_asked):
        high, low = self.policy.high, self.policy.low
        if not identity:
            return 'persona no reconocida'
        if not live:
            if decision == ASK:
                return 'no detecto movimiento de boca: ¿foto o audio grabado?'
            return 'boca quieta por segunda vez'
        if decision == ANSWER:
            if was_granted:
                return f'acceso vigente (≥ {high - self.policy.margin:.2f})'
            return f'≥ {high:.2f}'
        if decision == ASK:
            return f'{low:.2f} ≤ C < {high:.2f}'
        if was_asked:
            return 'segunda vez en la franja de re-pregunta'
        return f'C < {low:.2f}'

    def expire_asked(self):
        for identity, when in list(self.asked_at.items()):
            if time.monotonic() - when > self.param('reask_window_s'):
                self.policy.asked.discard(identity)
                del self.asked_at[identity]

    def query(self, identity, confidence, question, turn):
        if not self.rag.service_is_ready():
            self.get_logger().error('/rag/query no está disponible.')
            self.say(UNAVAILABLE_TEXT, turn)
            return
        request = QueryDocument.Request()
        request.person = identity
        request.identity_confidence = float(confidence)
        request.question = question
        self.rag.call_async(request).add_done_callback(
            partial(self.on_answer, turn, time.monotonic()))

    def on_answer(self, turn, started, future):
        turn.answer_seconds = float(time.monotonic() - started)
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f'Falló la consulta RAG: {error}')
            self.say(UNAVAILABLE_TEXT, turn)
            return
        turn.quotes = len(response.quotes)
        self.say(response.answer, turn)

    def say(self, text, turn=None):
        self.last['answer'] = text
        self.response_pub.publish(String(data=text))
        self.get_logger().info(f'Robot: {text}')
        if turn is not None:
            turn.answer = text
            self.dialog_pub.publish(turn)

    def monitor_access(self):
        now = self.now()
        floor = self.policy.high - self.policy.margin
        for identity in list(self.policy.granted):
            state = self.session.get(identity)
            if state is None:
                self.revoke(identity, 'sin sesión')
                continue
            recent = [o for o in self.faces
                      if o.track == state['track'] and o.name == identity
                      and o.t >= now - self.param('face_absent_s')]
            if not recent:
                self.revoke(identity, f'su cara no aparece hace más de {self.param("face_absent_s"):.0f} s')
                continue
            window = [o.score for o in recent if o.t >= now - 1.0] or [o.score for o in recent]
            state['confidence'] = fused_confidence(float(np.median(window)), state['voice'],
                                                   state['mouth'])
            if state['confidence'] >= floor:
                state['below_since'] = None
            elif state['below_since'] is None:
                state['below_since'] = now
            elif now - state['below_since'] > self.param('revoke_after_s'):
                self.revoke(identity, f'su confianza bajó a {state["confidence"]:.2f} (< {floor:.2f})')

    def revoke(self, identity, reason):
        self.policy.revoke(identity)
        self.session.pop(identity, None)
        self.get_logger().warn(f'Acceso revocado a {identity}: {reason}')
        turn = DialogTurn()
        turn.header.stamp = self.get_clock().now().to_msg()
        turn.identity = identity
        turn.decision = 'revocar'
        turn.reason = reason
        self.dialog_pub.publish(turn)

    def image_callback(self, msg):
        if self.overlay_pub.get_subscription_count() == 0:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().error(f'No se pudo convertir la imagen: {error}')
            return
        self.draw(frame)
        overlay = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        overlay.header = msg.header
        self.overlay_pub.publish(overlay)

    def draw(self, frame):
        now = time.monotonic()
        speaking = (self.last['track']
                    if now - self.last['when'] < SPEAKER_SHOWN_S else None)
        if self.latest_faces and now - self.latest_faces[0] < OVERLAY_FRESH_S:
            for f in self.latest_faces[1].faces:
                x, y = int(f.face.bbox_x), int(f.face.bbox_y)
                w, h = int(f.face.bbox_width), int(f.face.bbox_height)
                if f.identity_name in self.policy.granted:
                    color = (0, 200, 0)
                elif f.identity_name != UNKNOWN:
                    color = (0, 200, 255)
                else:
                    color = (0, 0, 230)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f'{f.identity_name} cara {f.identity_confidence:.2f}'
                if f.face.track_id == speaking:
                    label += '  HABLA'
                    cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), (255, 255, 255), 1)
                cv2.putText(frame, ascii_text(label), (x, max(y - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        granted = ', '.join(f'{n} C={self.session.get(n, {}).get("confidence", 0):.2f}'
                            for n in sorted(self.policy.granted)) or 'nadie'
        lines = [
            f'Acceso: {granted}   (responde >= {self.policy.high:.2f}, '
            f'repregunta >= {self.policy.low:.2f})',
            f'Ultima: {self.last["identity"] or "?"}  C={self.last["confidence"]:.2f}  '
            f'-> {self.last["decision"] or "-"}',
        ] + [f'{"Robot: " if i == 0 else "       "}{chunk}'
             for i, chunk in enumerate(textwrap.wrap(self.last['answer'], 66)[:3])]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22 * len(lines) + 8), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, ascii_text(line), (8, 20 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
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

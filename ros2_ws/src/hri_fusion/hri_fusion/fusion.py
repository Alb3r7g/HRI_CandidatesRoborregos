from dataclasses import dataclass, field

import numpy as np

ANCHORS = {
    'face': (0.20, 0.55),
    'voice': (0.35, 0.60),
    'mouth': (0.006, 0.020),
}
WEIGHTS = {'face': 0.5, 'voice': 0.3, 'mouth': 0.2}
LIVENESS_MIN = 0.015
HIGH_THRESHOLD = 0.90
LOW_THRESHOLD = 0.65
MARGIN = 0.05

ANSWER = 'responder'
ASK = 'preguntar'
DENY = 'negar'
REASK_TEXT = 'Por seguridad. Mira a la cámara y repite tu pregunta.'


def normalize(value, signal):
    low, high = ANCHORS[signal]
    if value is None:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def fused_confidence(face, voice, mouth):
    parts = {'face': face, 'voice': voice, 'mouth': mouth}
    return sum(WEIGHTS[s] * normalize(v, s) for s, v in parts.items())


def mouth_activity(openings):
    return float(np.std(openings)) if len(openings) >= 3 else 0.0


def pick_speaker(openings_by_track):
    activity = {t: mouth_activity(v) for t, v in openings_by_track.items()}
    if not activity:
        return None, 0.0
    track = max(activity, key=activity.get)
    return track, activity[track]


@dataclass
class Utterance:
    identity: str
    face: float
    voice: float
    mouth: float
    liveness_min: float = LIVENESS_MIN

    @property
    def live(self):
        return self.mouth is not None and self.mouth >= self.liveness_min

    @property
    def confidence(self):
        return fused_confidence(self.face, self.voice, self.mouth)

    @property
    def score(self):
        return self.confidence if self.live else 0.0


@dataclass
class AccessPolicy:
    high: float
    low: float
    margin: float = MARGIN
    granted: set = field(default_factory=set)
    asked: set = field(default_factory=set)

    def decide(self, identity, confidence, live=True):
        if not identity:
            return DENY
        if not live:
            if identity in self.asked:
                self.asked.discard(identity)
                self.revoke(identity)
                return DENY
            self.asked.add(identity)
            return ASK
        if identity in self.granted and confidence >= self.high - self.margin:
            self.asked.discard(identity)
            return ANSWER
        if confidence >= self.high:
            self.granted.add(identity)
            self.asked.discard(identity)
            return ANSWER
        if confidence >= self.low and identity not in self.asked:
            self.asked.add(identity)
            return ASK
        self.asked.discard(identity)
        self.revoke(identity)
        return DENY

    def revoke(self, identity):
        self.granted.discard(identity)

import textwrap

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hri_fusion.fusion import ANSWER, ASK
from hri_interfaces.msg import DialogTurn

WIDTH = 62
RULE = '─' * (WIDTH + 12)
INDENT = ' ' * 12
DECISIONS = {ANSWER: 'RESPONDER', ASK: 'RE-PREGUNTAR'}


def wrapped(prefix, text):
    lines = textwrap.wrap(text, WIDTH) or ['']
    return [prefix + lines[0]] + [INDENT + line for line in lines[1:]]


class ConversationView(Node):

    def __init__(self):
        super().__init__('conversation_view')
        self.create_subscription(DialogTurn, '/hri/dialog', self.show, 10)
        print(RULE)
        print(' Conversación HRI: esperando a que alguien hable...', flush=True)

    def show(self, turn):
        if turn.decision == 'revocar':
            print(RULE)
            print(f' ACCESO REVOCADO a {turn.identity}: {turn.reason}', flush=True)
            return

        voice = f'{turn.voice:.2f}' if turn.voice > 0 else '—'
        mouth = 'boca viva' if turn.live else 'boca QUIETA'
        who = turn.identity or '?'
        lines = [RULE]
        lines += wrapped(' Tú:        ', turn.question)
        lines.append(f' Quién:     {who:<8} cara {turn.face:.2f} · voz {voice} · {mouth}'
                     f'      C = {turn.confidence:.2f}')
        lines.append(f' Decisión:  {DECISIONS.get(turn.decision, "NEGAR")}  ({turn.reason})')
        lines += wrapped(' Robot:     ', turn.answer)
        if turn.answer_seconds > 0:
            detail = f'{turn.answer_seconds:.1f} s · {turn.quotes} cita(s)'
            lines.append(INDENT + detail.rjust(WIDTH))
        print('\n'.join(lines), flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = ConversationView()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(RULE, flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

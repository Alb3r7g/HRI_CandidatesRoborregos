import json
import os
import re
import time
import unicodedata
import urllib.request
from difflib import SequenceMatcher

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
LLM_MODEL = 'qwen2.5:7b'
CONTEXT_TOKENS = 4096
KEEP_ALIVE = '30m'
MIN_SHARED_RUN = 3
MIN_QUOTE_COVERAGE = 0.8
REQUEST_TIMEOUT_S = 120

REFUSAL_TEXT = 'No tengo esa informacion en el documento de {person}.'

SYSTEM_PROMPT = """Eres el asistente de un robot de servicio. Respondes preguntas sobre {person} \
usando UNICAMENTE los fragmentos de su documento privado que aparecen abajo.

Reglas:
1. Solo puedes usar informacion escrita literalmente en los fragmentos. No uses conocimiento \
general ni supongas nada.
2. Que un fragmento hable del mismo tema NO significa que contenga el dato. Si el dato exacto \
que se pide no esta escrito, la pregunta no tiene respuesta.
3. Si la pregunta es sobre otra persona distinta de {person}, solo puedes responder con lo que \
el documento de {person} diga de ella; si no lo dice, no tiene respuesta.
4. Devuelve JSON con dos campos:
   - "citas": lista con las frases exactas que contienen el dato, copiadas letra por letra \
de los fragmentos, sin resumir, sin parafrasear y sin cambiar palabras. Si la respuesta usa \
datos de varios lugares, una cita por cada uno. Si el dato no esta, lista vacia.
   - "respuesta": respuesta breve en espanol, en una o dos oraciones, redactada con \
naturalidad a partir de las citas. Si "citas" esta vacia, escribe exactamente: \
"{refusal}"

Fragmentos del documento de {person}:
{context}"""

ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {
        'citas': {'type': 'array', 'items': {'type': 'string'}},
        'respuesta': {'type': 'string'},
    },
    'required': ['citas', 'respuesta'],
}


class LlmError(RuntimeError):
    pass


def build_context(fragments):
    return '\n\n'.join(f'[{i + 1}] {f["text"]}' for i, f in enumerate(fragments))


NON_WORD = re.compile(r'[^\w]+')


def normalize(text):
    decomposed = unicodedata.normalize('NFKD', text.casefold())
    bare = ''.join(c for c in decomposed if not unicodedata.combining(c))
    return NON_WORD.sub(' ', bare).strip()


REPEATED_RUN = re.compile(r'\b((?:\w+ ){1,8}?\w+)(?: \1\b)+')


def collapse_repeats(text):
    return REPEATED_RUN.sub(r'\1', text)


def coverage(quote_words, source_words):
    min_run = min(MIN_SHARED_RUN, len(quote_words))
    matcher = SequenceMatcher(None, quote_words, source_words, autojunk=False)
    shared = sum(b.size for b in matcher.get_matching_blocks() if b.size >= min_run)
    return shared / len(quote_words)


def quote_is_grounded(quote, sources):
    words = collapse_repeats(normalize(quote)).split()
    if not words:
        return False
    return any(coverage(words, source) >= MIN_QUOTE_COVERAGE for source in sources)


def quotes_are_grounded(quotes, fragments):
    if not quotes:
        return False
    sources = [normalize(f['text']).split() for f in fragments]
    return all(quote_is_grounded(q, sources) for q in quotes)


def _post(path, payload, url):
    request = urllib.request.Request(
        f'{url}{path}',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except OSError as error:
        raise LlmError(f'Ollama no responde en {url}: {error}') from error


def answer(person, question, fragments, model=LLM_MODEL, url=OLLAMA_URL):
    refusal = REFUSAL_TEXT.format(person=person)
    system = SYSTEM_PROMPT.format(
        person=person, refusal=refusal, context=build_context(fragments))
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': question},
        ],
        'format': ANSWER_SCHEMA,
        'stream': False,
        'keep_alive': KEEP_ALIVE,
        'options': {'temperature': 0, 'seed': 0, 'num_ctx': CONTEXT_TOKENS},
    }
    started = time.monotonic()
    reply = _post('/api/chat', payload, url)
    elapsed = time.monotonic() - started
    try:
        parsed = json.loads(reply['message']['content'])
        quotes, text = parsed['citas'], parsed['respuesta']
    except (KeyError, ValueError) as error:
        raise LlmError(f'Respuesta no valida del modelo: {reply}') from error
    grounded = quotes_are_grounded(quotes, fragments)
    refused = normalize(text).startswith(normalize(refusal))
    return {
        'answer': text,
        'quotes': quotes,
        'grounded': grounded,
        'refused': refused,
        'answered': grounded and not refused,
        'prompt_tokens': reply.get('prompt_eval_count', 0),
        'seconds': elapsed,
    }

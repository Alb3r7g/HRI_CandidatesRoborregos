#!/usr/bin/env python3
"""Mide las respuestas del LLM sobre el set de evaluacion: aciertos en las contestables y rechazos en las trampas."""
import argparse
import json
import pathlib
import statistics
import sys

from hri_rag.chunking import collapse
from hri_rag.llm import LLM_MODEL, OLLAMA_URL, LlmError, answer
from hri_rag.store import CHROMA_PATH, DEFAULT_TOP_K, DocumentStore

EVAL_DIR = '/data/eval'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('person')
    parser.add_argument('--eval-dir', default=EVAL_DIR)
    parser.add_argument('--chroma', default=CHROMA_PATH)
    parser.add_argument('--model', default=LLM_MODEL)
    parser.add_argument('--url', default=OLLAMA_URL)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def contains(text, needle):
    return collapse(needle).casefold() in collapse(text).casefold()


def ask(store, args, question):
    fragments = store.query(args.person, question, top_k=args.top_k)
    return answer(args.person, question, fragments, model=args.model, url=args.url)


def report(item, result, verdict, detail=''):
    if result['refused']:
        state = 'se nego'
    elif result['grounded']:
        state = 'citas verificadas'
    else:
        state = 'citas NO verificadas'
    print(f'  {item["id"]:<4} {verdict}  {result["seconds"]:5.1f} s  {state:<21}{detail}')
    print(f'         {result["answer"]}')
    if not result['grounded'] and not result['refused']:
        for quote in result['quotes']:
            print(f'         cita rechazada: {quote}')


def main():
    args = parse_args()
    path = pathlib.Path(args.eval_dir) / f'{args.person}.json'
    if not path.is_file():
        print(f'No existe {path}', file=sys.stderr)
        return 1
    with open(path, encoding='utf-8') as handle:
        spec = json.load(handle)

    store = DocumentStore(path=args.chroma)
    if args.person not in store.people():
        print(f'No hay coleccion "{args.person}". Corre ingest_documents.py primero.',
              file=sys.stderr)
        return 1

    print(f'Persona: {args.person}   modelo: {args.model}   top-k: {args.top_k}\n')
    latencies = []
    try:
        answer(args.person, 'Hola', [], model=args.model, url=args.url)

        print('  Contestables (deben responder con el dato):')
        answered = 0
        for item in spec['answerable']:
            result = ask(store, args, item['question'])
            latencies.append(result['seconds'])
            missing = [n for n in item['needles'] if not contains(result['answer'], n)]
            ok = result['answered'] and not missing
            answered += ok
            detail = f'   falta: {", ".join(missing)}' if missing else ''
            report(item, result, 'OK   ' if ok else 'FALLA', detail)

        print('\n  Trampas (deben rechazarse):')
        refused = 0
        for item in spec['unanswerable']:
            result = ask(store, args, item['question'])
            latencies.append(result['seconds'])
            ok = not result['answered']
            refused += ok
            report(item, result, 'OK   ' if ok else 'FALLA', f'   [{item["tipo"]}]')
    except LlmError as error:
        print(f'\n{error}', file=sys.stderr)
        return 1

    print(f'\n  Contestables: {answered}/{len(spec["answerable"])}   '
          f'trampas rechazadas: {refused}/{len(spec["unanswerable"])}')
    print(f'  Latencia: mediana {statistics.median(latencies):.1f} s   '
          f'maxima {max(latencies):.1f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())

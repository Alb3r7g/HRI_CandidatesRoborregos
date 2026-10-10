#!/usr/bin/env python3
"""Mide la recuperacion del indice persistido contra el set de evaluacion de una persona."""
import argparse
import json
import pathlib
import sys

from hri_rag.chunking import collapse
from hri_rag.store import CHROMA_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL, DocumentStore

EVAL_DIR = '/data/eval'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('person')
    parser.add_argument('--eval-dir', default=EVAL_DIR)
    parser.add_argument('--chroma', default=CHROMA_PATH)
    parser.add_argument('--model', default=EMBEDDING_MODEL)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main():
    args = parse_args()
    path = pathlib.Path(args.eval_dir) / f'{args.person}.json'
    if not path.is_file():
        print(f'No existe {path}', file=sys.stderr)
        return 1

    with open(path, encoding='utf-8') as handle:
        spec = json.load(handle)

    store = DocumentStore(path=args.chroma, model_name=args.model)
    if args.person not in store.people():
        print(f'No hay coleccion "{args.person}". Corre ingest_documents.py primero.',
              file=sys.stderr)
        return 1

    print(f'Persona: {args.person}   modelo: {args.model}   top-k: {args.top_k}\n')

    hits = 0
    positive_scores = []
    for item in spec['answerable']:
        results = store.query(args.person, item['question'], top_k=args.top_k)
        context = collapse(' '.join(r['text'] for r in results))
        missing = [n for n in item['needles'] if collapse(n) not in context]
        top = results[0]['similarity']
        positive_scores.append(top)
        hits += not missing
        mark = 'OK   ' if not missing else 'FALLA'
        extra = '' if not missing else f'   falta: {", ".join(missing)}'
        print(f'  {item["id"]:<4} {mark}  sim {top:.3f}   {results[0]["heading"]}{extra}')

    print(f'\n  Aciertos: {hits}/{len(spec["answerable"])}\n')

    print('  Sin respuesta en el documento:')
    negative_scores = []
    for item in spec['unanswerable']:
        results = store.query(args.person, item['question'], top_k=args.top_k)
        top = results[0]['similarity']
        negative_scores.append(top)
        print(f'  {item["id"]:<4}        sim {top:.3f}   {results[0]["heading"]}'
              f'   [{item["tipo"]}]')

    worst = min(positive_scores)
    best = max(negative_scores)
    print(f'\n  Peor contestable {worst:.3f}   mejor sin respuesta {best:.3f}   '
          f'margen {worst - best:+.3f}')
    if worst <= best:
        print('  Los rangos se solapan: el rechazo no puede venir de un umbral de similitud.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

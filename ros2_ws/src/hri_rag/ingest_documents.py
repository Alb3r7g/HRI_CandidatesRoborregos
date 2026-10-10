#!/usr/bin/env python3
"""Indexa los documentos de /data/documents en ChromaDB, una coleccion por persona."""
import argparse
import pathlib
import sys

from hri_rag.chunking import chunk_file
from hri_rag.store import CHROMA_PATH, EMBEDDING_MODEL, DocumentStore

DOCUMENTS_DIR = '/data/documents'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', default=DOCUMENTS_DIR)
    parser.add_argument('--chroma', default=CHROMA_PATH)
    parser.add_argument('--model', default=EMBEDDING_MODEL)
    parser.add_argument('--person', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    directory = pathlib.Path(args.documents)
    if not directory.is_dir():
        print(f'No existe el directorio {directory}', file=sys.stderr)
        return 1

    paths = sorted(directory.glob('*.md'))
    if args.person:
        paths = [p for p in paths if p.stem == args.person]
    if not paths:
        print(f'No hay documentos que indexar en {directory}', file=sys.stderr)
        return 1

    store = DocumentStore(path=args.chroma, model_name=args.model)
    print(f'Modelo: {args.model}')
    print(f'Indice: {args.chroma}\n')

    for path in paths:
        chunks = chunk_file(path)
        count = store.replace(path.stem, chunks, str(path))
        longest = max(len(c['text']) for c in chunks)
        print(f'{path.stem:<16} {count:>3} fragmentos   '
              f'mas largo {longest:>4} car.   {path.name}')

    print(f'\nColecciones: {", ".join(store.people())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Comparacion de modelos de embeddings sobre el corpus real del sistema.

Trocea un documento, lo indexa en ChromaDB con cada modelo candidato y mide cuantas
preguntas del set de evaluacion recuperan el fragmento que contiene la respuesta.
Mide tambien el margen de similitud entre preguntas contestables y preguntas sin
respuesta en el documento, que es el dato que determina si conviene un umbral de
similitud en el servicio de RAG.

Uso (dentro del contenedor):
    python3 /ros2_ws/scripts/compare_embeddings.py
    python3 /ros2_ws/scripts/compare_embeddings.py --top-k 5
"""
import argparse
import re
import time

import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENT = '/data/documents/Albert.md'

# multilingual-e5 fue entrenado con prefijos que distinguen consulta de pasaje y su
# calidad cae varios puntos sin ellos; bge-m3 no los usa. Se declaran por modelo para
# que la comparacion mida el modelo y no el formato de entrada.
MODELS = [
    {'name': 'BAAI/bge-m3', 'query_prefix': '', 'passage_prefix': ''},
    {'name': 'intfloat/multilingual-e5-large', 'query_prefix': 'query: ',
     'passage_prefix': 'passage: '},
    {'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
     'query_prefix': '', 'passage_prefix': ''},
]

# Cada pregunta se acompaña de las cadenas que deben aparecer en el contexto
# recuperado. Evita etiquetar fragmentos a mano: si el texto recuperado contiene la
# cifra o el nombre que responde, la recuperacion sirvio.
QUESTIONS = [
    ('A1', '¿En qué carrera y campus estudia actualmente, y cuántos semestres lleva cursados?',
     ['Robótica y Sistemas Digitales', 'cinco semestres']),
    ('A2', '¿Quiénes son sus roomies en Monterrey?',
     ['Alejandro del Ángel', 'Sebastián Alemán']),
    ('A3', '¿Qué tiempo registró en su primera competencia de la temporada, en enero de 2024, y por qué fue un tiempo lento?',
     ['12.30', 'fiebre']),
    ('A4', '¿Qué tiempo logró en la competencia de la UNAM y qué consiguió con ese resultado?',
     ['11.50', 'CONADEIP']),
    ('A5', '¿Quiénes eran sus entrenadores de atletismo y cuál era su horario de entrenamiento?',
     ['Chava', '6:00 a 8:30']),
    ('A6', '¿Qué tiempo registró en la carrera individual dentro de CONADEIP y qué puesto obtuvo?',
     ['11.92', 'puesto 11']),
    ('A7', '¿Cuál fue el orden del relevo 4x100 en CONADEIP y qué resultado obtuvieron?',
     ['Diego Silva', 'segundo lugar']),
    ('A8', '¿Cuál fue su mejor marca personal en 100 m planos y dónde la logró?',
     ['11.20', 'Campus Ciudad de México']),
    ('A9', '¿Cuánto mejoró su tiempo entre su primera competencia de la temporada y su mejor marca personal?',
     ['12.30', '11.20']),
    ('A10', '¿Alguna vez se lesionó mientras entrenaba?',
     ['contracturas']),
]

# Sin respuesta en el documento. No se puntuan por recuperacion: siempre va a volver
# algo. Se usan para medir con que similitud vuelve, comparada con las contestables.
NEGATIVES = [
    ('B1', '¿Quiénes fueron sus compañeros de equipo en LARC 2025?'),
    ('B2', '¿En qué hotel se hospedó el equipo durante su estancia en Monterrey?'),
    ('B3', '¿Qué carrera universitaria estudia Sebastián Alemán?'),
]


def collapse(text):
    """Colapsa cualquier secuencia de espacios en blanco a un solo espacio."""
    return re.sub(r'\s+', ' ', text)


def chunk_document(path):
    """Trocea por encabezado y, dentro de una seccion, por bloque en negritas.

    Las cinco marcas de tiempo del documento son casi identicas como texto; lo unico
    que las distingue es el mes y el lugar que las acompañan. Cada fragmento arrastra
    su encabezado para que no quede una cifra suelta imposible de desambiguar.
    """
    text = open(path, encoding='utf-8').read()
    chunks = []
    for section in re.split(r'\n## ', text):
        section = section.strip()
        if not section or section.startswith('# '):
            continue
        heading, _, body = section.partition('\n')
        heading = heading.strip()

        blocks = re.split(r'\n\n(?=\*\*)', body.strip())
        for block in blocks:
            block = block.strip()
            if block:
                chunks.append(f'{heading}\n\n{block}')
    return chunks


def evaluate(model_spec, chunks, top_k):
    model = SentenceTransformer(model_spec['name'], device='cpu')

    started = time.perf_counter()
    passages = [model_spec['passage_prefix'] + c for c in chunks]
    embeddings = model.encode(passages, normalize_embeddings=True).tolist()
    index_seconds = time.perf_counter() - started

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name='comparacion', metadata={'hnsw:space': 'cosine'})
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings)

    def retrieve(question):
        vector = model.encode(
            [model_spec['query_prefix'] + question],
            normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=vector, n_results=top_k)
        # Chroma devuelve distancia coseno; la similitud es su complemento.
        return result['documents'][0], [1 - d for d in result['distances'][0]]

    hits, rows = 0, []
    started = time.perf_counter()
    for qid, question, needles in QUESTIONS:
        documents, similarities = retrieve(question)
        # El documento viene con saltos de linea, asi que un nombre puede quedar
        # partido ("Diego\nSilva") y no coincidir como subcadena. Se comparan los dos
        # lados con los espacios colapsados.
        context = collapse(' '.join(documents))
        found = all(collapse(needle) in context for needle in needles)
        hits += found
        missing = [n for n in needles if collapse(n) not in context]
        rows.append((qid, found, similarities[0], missing))
    query_seconds = (time.perf_counter() - started) / len(QUESTIONS)

    negatives = []
    for qid, question in NEGATIVES:
        _, similarities = retrieve(question)
        negatives.append((qid, similarities[0]))

    client.delete_collection('comparacion')
    return {
        'hits': hits, 'rows': rows, 'negatives': negatives,
        'index_seconds': index_seconds, 'query_seconds': query_seconds,
        'dimensions': len(embeddings[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=3,
                        help='Fragmentos recuperados por consulta (default: 3)')
    args = parser.parse_args()

    chunks = chunk_document(DOCUMENT)
    print(f'Documento: {DOCUMENT}')
    print(f'Fragmentos: {len(chunks)}')
    print(f'Longitud media: {sum(len(c) for c in chunks) // len(chunks)} caracteres')
    print(f'Fragmento mas largo: {max(len(c) for c in chunks)} caracteres')
    print(f'\nPreguntas contestables: {len(QUESTIONS)}   sin respuesta: {len(NEGATIVES)}')
    print(f'Recuperando {args.top_k} fragmentos por consulta\n')

    summary = []
    for spec in MODELS:
        print('=' * 78)
        print(spec['name'])
        print('=' * 78)
        result = evaluate(spec, chunks, args.top_k)

        for qid, found, similarity, missing in result['rows']:
            mark = 'OK  ' if found else 'FALLA'
            detail = '' if found else f'   no recuperado: {", ".join(missing)}'
            print(f'  {qid:<4} {mark}  sim. maxima {similarity:.3f}{detail}')

        print(f'\n  Aciertos: {result["hits"]}/{len(QUESTIONS)}')
        print(f'  Dimensiones: {result["dimensions"]}')
        print(f'  Indexado: {result["index_seconds"]:.1f} s   '
              f'Consulta: {result["query_seconds"] * 1000:.0f} ms')

        print('\n  Preguntas SIN respuesta (similitud del mejor fragmento):')
        for qid, similarity in result['negatives']:
            print(f'    {qid}  {similarity:.3f}')

        positives = [s for _, _, s, _ in result['rows']]
        worst_positive = min(positives)
        best_negative = max(s for _, s in result['negatives'])
        margin = worst_positive - best_negative
        print(f'\n  Peor contestable: {worst_positive:.3f}   '
              f'Mejor sin respuesta: {best_negative:.3f}   Margen: {margin:+.3f}')
        if margin <= 0:
            print('  Los rangos se solapan: un umbral de similitud no separa los dos casos.')
        print()

        summary.append((spec['name'], result['hits'], result['dimensions'],
                        result['query_seconds'] * 1000, margin))

    print('=' * 78)
    print('RESUMEN')
    print('=' * 78)
    print(f'{"modelo":<58} {"acierto":>8} {"dim":>5} {"ms":>6} {"margen":>8}')
    for name, hits, dims, ms, margin in summary:
        print(f'{name:<58} {hits:>4}/{len(QUESTIONS)} {dims:>5} {ms:>6.0f} {margin:>+8.3f}')


if __name__ == '__main__':
    main()

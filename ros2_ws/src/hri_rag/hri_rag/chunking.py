import re

SECTION_SEPARATOR = re.compile(r'\n## ')
BLOCK_SEPARATOR = re.compile(r'\n\n(?=\*\*)')
WHITESPACE = re.compile(r'\s+')


def collapse(text):
    return WHITESPACE.sub(' ', text).strip()


def chunk_markdown(text):
    chunks = []
    for section in SECTION_SEPARATOR.split(text):
        section = section.strip()
        if not section or section.startswith('# '):
            continue
        heading, _, body = section.partition('\n')
        heading = heading.strip()
        for block in BLOCK_SEPARATOR.split(body.strip()):
            block = block.strip()
            if block:
                chunks.append({'heading': heading, 'text': f'{heading}\n\n{block}'})
    return chunks


def chunk_file(path):
    with open(path, encoding='utf-8') as handle:
        return chunk_markdown(handle.read())

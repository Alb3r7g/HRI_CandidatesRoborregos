import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = 'BAAI/bge-m3'
CHROMA_PATH = '/data/chroma'
DEFAULT_TOP_K = 3


class UnknownPerson(KeyError):
    pass


class DocumentStore:

    def __init__(self, path=CHROMA_PATH, model_name=EMBEDDING_MODEL, device='cpu'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.client = chromadb.PersistentClient(path=path)

    def people(self):
        return sorted(c.name for c in self.client.list_collections())

    def _collection(self, person):
        return self.client.get_or_create_collection(
            name=person, metadata={'hnsw:space': 'cosine'})

    def _encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def replace(self, person, chunks, source):
        if person in self.people():
            self.client.delete_collection(person)
        collection = self._collection(person)
        texts = [c['text'] for c in chunks]
        collection.add(
            ids=[f'{person}-{i:03d}' for i in range(len(chunks))],
            documents=texts,
            embeddings=self._encode(texts),
            metadatas=[
                {'person': person, 'heading': c['heading'], 'source': source}
                for c in chunks
            ])
        return collection.count()

    def query(self, person, question, top_k=DEFAULT_TOP_K):
        if person not in self.people():
            raise UnknownPerson(person)
        result = self._collection(person).query(
            query_embeddings=self._encode([question]), n_results=top_k)
        return [
            {'text': text, 'similarity': 1.0 - distance, 'heading': meta['heading']}
            for text, distance, meta in zip(result['documents'][0],
                                            result['distances'][0],
                                            result['metadatas'][0])
        ]

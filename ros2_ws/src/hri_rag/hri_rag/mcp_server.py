from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from hri_rag.store import DEFAULT_TOP_K, DocumentStore, UnknownPerson

server = MCPServer(
    'hri-documentos',
    instructions='Busqueda semantica en el documento privado de una persona ya identificada. '
                 'Cada consulta se limita a la coleccion de esa persona.',
    log_level='WARNING')
store = DocumentStore()


@server.tool(description='Devuelve los fragmentos del documento privado de `person` mas '
                         'parecidos a `question`. Solo busca en la coleccion de esa persona.')
def search_document(person: str, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    person = person.strip()
    if not person:
        raise ToolError('Consulta rechazada: falta la identidad de la persona.')
    try:
        return store.query(person, question, top_k=top_k)
    except UnknownPerson:
        raise ToolError(f'No existe documento indexado para "{person}".') from None


def main():
    server.run('stdio')


if __name__ == '__main__':
    main()

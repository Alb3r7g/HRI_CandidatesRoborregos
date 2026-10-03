import rclpy
from rclpy.node import Node

from hri_interfaces.srv import QueryDocument
from hri_rag.document_search import DocumentSearchClient, SearchRejected
from hri_rag.llm import LLM_MODEL, OLLAMA_URL, REFUSAL_TEXT, LlmError, answer
from hri_rag.store import DEFAULT_TOP_K

DENIED_TEXT = 'No tengo permiso para consultar ese documento.'
UNAVAILABLE_TEXT = 'El modelo de lenguaje no esta disponible en este momento.'


class RagServiceNode(Node):

    def __init__(self):
        super().__init__('rag_service_node')

        self.declare_parameter('access_threshold', 0.5)
        self.declare_parameter('top_k', DEFAULT_TOP_K)
        self.declare_parameter('llm_model', LLM_MODEL)
        self.declare_parameter('ollama_url', OLLAMA_URL)

        self.top_k = self.get_parameter('top_k').value
        self.llm_model = self.get_parameter('llm_model').value
        self.ollama_url = self.get_parameter('ollama_url').value

        self.get_logger().info('Arrancando el servidor MCP de documentos...')
        self.search = DocumentSearchClient()
        self.get_logger().info(
            f'MCP listo, herramientas: {", ".join(self.search.tools())}   '
            f'umbral de acceso PROVISIONAL: {self.access_threshold():.2f}')

        self.create_service(QueryDocument, '/rag/query', self.handle_query)

    def access_threshold(self):
        return self.get_parameter('access_threshold').value

    def denial_reason(self, request):
        person = request.person.strip()
        if not person:
            return 'peticion sin identidad'
        if request.identity_confidence < self.access_threshold():
            return (f'confianza {request.identity_confidence:.2f} por debajo del '
                    f'umbral {self.access_threshold():.2f}')
        return None

    def handle_query(self, request, response):
        reason = self.denial_reason(request)
        person = request.person.strip()
        if not reason:
            try:
                fragments = self.search.search(person, request.question, self.top_k)
            except SearchRejected as error:
                reason = f'MCP: {error}'
        if reason:
            self.get_logger().warn(f'DENEGADO: {reason}')
            response.status = QueryDocument.Response.DENIED
            response.answer = DENIED_TEXT
            return response

        try:
            result = answer(person, request.question, fragments,
                            model=self.llm_model, url=self.ollama_url)
        except LlmError as error:
            self.get_logger().error(str(error))
            response.status = QueryDocument.Response.LLM_UNAVAILABLE
            response.answer = UNAVAILABLE_TEXT
            return response

        if result['answered']:
            response.status = QueryDocument.Response.ANSWERED
            response.answer = result['answer']
            response.quotes = result['quotes']
        else:
            response.status = QueryDocument.Response.NOT_IN_DOCUMENT
            response.answer = REFUSAL_TEXT.format(person=person)
        self.get_logger().info(
            f'{person} ({request.identity_confidence:.2f}): "{request.question}" -> '
            f'{"respondida" if result["answered"] else "sin dato"} en {result["seconds"]:.1f} s')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RagServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.search.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

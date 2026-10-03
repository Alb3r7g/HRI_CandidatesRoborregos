import asyncio
import concurrent.futures
import os
import sys
import threading

from mcp import Client, StdioServerParameters

TOOL_NAME = 'search_document'
CONNECT_TIMEOUT_S = 120
CALL_TIMEOUT_S = 30


class SearchRejected(RuntimeError):
    pass


class DocumentSearchClient:

    def __init__(self):
        env = dict(os.environ)
        env.setdefault('HF_HUB_OFFLINE', '1')
        self.params = StdioServerParameters(
            command=sys.executable, args=['-m', 'hri_rag.mcp_server'], env=env)
        self.loop = asyncio.new_event_loop()
        self.ready = concurrent.futures.Future()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self.loop)
        self.ready.result(timeout=CONNECT_TIMEOUT_S)

    async def _serve(self):
        self.requests = asyncio.Queue()
        try:
            async with Client(self.params) as client:
                tools = await client.list_tools()
                self.ready.set_result([t.name for t in tools.tools])
                while (request := await self.requests.get()) is not None:
                    arguments, reply = request
                    try:
                        reply.set_result(await client.call_tool(TOOL_NAME, arguments))
                    except Exception as error:
                        reply.set_exception(error)
        except Exception as error:
            if not self.ready.done():
                self.ready.set_exception(error)

    def tools(self):
        return self.ready.result()

    def search(self, person, question, top_k):
        reply = concurrent.futures.Future()
        arguments = {'person': person, 'question': question, 'top_k': top_k}
        self.loop.call_soon_threadsafe(self.requests.put_nowait, (arguments, reply))
        result = reply.result(timeout=CALL_TIMEOUT_S)
        if result.is_error:
            raise SearchRejected(' '.join(c.text for c in result.content if hasattr(c, 'text')))
        return result.structured_content['result']

    def close(self):
        self.loop.call_soon_threadsafe(self.requests.put_nowait, None)

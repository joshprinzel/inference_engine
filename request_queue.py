from queue import Empty, Queue

from request_state import RequestState


class RequestQueue:
    def __init__(self) -> None:
        self._queue: Queue[RequestState] = Queue()

    def put(self, request_state: RequestState) -> None:
        self._queue.put(request_state)

    def get(self) -> RequestState:
        return self._queue.get()

    def get_nowait(self) -> RequestState:
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()
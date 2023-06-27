from __future__ import annotations

import atexit
import logging
from inspect import isclass
from threading import Condition, Thread

import tornado
import tornado.httpserver
import tornado.ioloop
import tornado.web
from rich.logging import RichHandler


def configure_logging() -> None:
    rich_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_suppress=[tornado],
    )
    FORMAT = "[%(threadName)-10s] %(name)s: %(message)s"
    # FORMAT = "[%(threadName)-10s] %(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[rich_handler]
    )
    # Reduce noise
    logging.getLogger('asyncio').setLevel(logging.INFO)
    logging.getLogger('parso').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.INFO)
    logging.getLogger('hpack').setLevel(logging.INFO)
    logging.getLogger('botocore').setLevel(logging.INFO)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)


# FIXME: probably not the best place for this - move to a separate module?
configure_logging()

log = logging.getLogger(__name__)

_servers: list[HttpServer] = []


def get_servers() -> list[HttpServer]:
    return _servers


def start_servers(restart: bool = False) -> None:
    if restart is False:
        servers = [s for s in _servers if not s.is_running]
    else:
        servers = _servers
    if len(servers) == 0:
        log.info('No servers to start')
        return
    log.info('Starting %s servers', len(servers))
    for server in servers:
        server.start(restart=True)


def stop_servers(join: bool = True) -> None:
    log.info('Stopping %s servers', len(_servers))
    for server in _servers:
        server.stop()
    if join:
        for server in _servers:
            server.join()
    log.debug('All servers stopped')


@atexit.register
def _at_exit() -> None:
    stop_servers()
    logging.getLogger().handlers.clear()


class HttpServer:
    _instance: HttpServer | None = None
    port: int | None = None
    address: str | None = None
    _name: str | None = None
    handlers: list[tuple[str, type['HttpRequestHandler']]] = []
    _thread: Thread | None = None
    _lock: Condition
    _server: tornado.httpserver.HTTPServer | None = None
    _app: tornado.web.Application | None = None
    _io_loop: tornado.ioloop.IOLoop | None = None

    autostart: bool = True

    def __init_subclass__(cls) -> None:
        if cls.port is not None:
            cls._instance = cls()

    @classmethod
    def get_instance(cls) -> HttpServer | None:
        return cls._instance

    def _same_server(self, other: HttpServer) -> bool:
        # TODO: how to support multiple ports and addresses?
        if self.port != other.port:
            return False
        if self.address != other.address:
            # FIXME: if address is None, it could collide with 0.0.0.0 or ::
            return False
        return True

    def __init__(
        self,
        port: int | None = None,
        address: str | None = None,
        name: str | None = None,
        autostart: bool | None = None,
        handlers: list[tuple[str, type['HttpRequestHandler']]] | None = None,
    ):
        self.address = address
        self._name = name
        if port is not None:
            self.port = port
        if self.port is None:
            raise ValueError('Port must be specified')
        self.handlers = handlers or []
        self._thread = None
        self._lock = Condition()
        self._server = None
        self._app = None
        self._io_loop = None
        # TODO: this should probably go in start() (and removed in stop())
        # TODO: so that _servers is only populated with running servers?
        existing_servers = [s for s in _servers if self._same_server(s)]
        if existing_servers:
            # We have an existing server with the same port and address
            # this is typically a reload of the module. We need to stop
            # the existing server.
            for server in existing_servers:
                log.info('Removing existing server: %s', server)
                if server.is_running:
                    server.stop()
                    server.join()
                _servers.remove(server)
        _servers.append(self)
        if autostart is not None:
            self.autostart = autostart
        if self.autostart is True:
            self.start()

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __repr__(self) -> str:
        if self.is_running:
            if self.address is None:
                addr = '0.0.0.0'
            else:
                addr = self.address
            return f'<{self.name} listening at {addr}:{self.port}>'
        return f'<{self.name} stopped>'

    def add_handler(self, handler: type['HttpRequestHandler']) -> None:
        if hasattr(handler, 'path') and handler.path is not None:
            handler.server = self
            self.handlers.append((handler.path, handler))
            if self.is_running and self._io_loop is not None:

                def _add_handler() -> None:
                    log.debug('Adding handler: %s', (handler.path, handler))
                    with self._lock:
                        if self._app is not None:
                            self._app.add_handlers(
                                '.*',
                                [(handler.path, handler)],  # type: ignore
                            )

                self._io_loop.add_callback(_add_handler)

    def _run(self) -> None:
        with self._lock:
            if self.port is None:
                raise ValueError('Port must be specified')
            self._io_loop = tornado.ioloop.IOLoop()
            self._io_loop.make_current()
            self._app = tornado.web.Application(
                self.handlers,  # type: ignore
                debug=True,
                autoreload=False,
            )
            self._server = tornado.httpserver.HTTPServer(self._app)
            # self._server.listen(port=self.port, address=self.address, reuse_port=True)
            self._server.listen(port=self.port, address=self.address)
            log.info('Listening at %s:%s', self.address or '0.0.0.0', self.port)
            log.debug('Handlers: %s', self.handlers)
        self._io_loop.start()
        self._io_loop.close()
        self._io_loop = None
        # log.debug('Server stopped')

    def start(self, restart: bool = True) -> None:
        if self.is_running:
            if restart is False:
                return
            # log.info('Restarting server')
            self.stop()
            self.join()
        self._thread = Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()
        log.debug('Started %s', self._thread)

    def restart(self) -> None:
        self.start()

    def stop(self) -> None:
        with self._lock:
            if not self._server:
                log.debug('Server %s not running. Nothing to stop', id(self))
                return
            # No more requests are allowed after stopping
            self._server.stop()

        if self._io_loop is not None:

            async def stopper() -> None:
                if self._server is not None:
                    # Wait for all connections to close cleanly
                    await self._server.close_all_connections()
                if self._io_loop is not None:
                    self._io_loop.stop()

            self._io_loop.add_callback(stopper)

    def join(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
            self._thread = None


class HttpRequestHandler(tornado.web.RequestHandler):
    server: HttpServer | type[HttpServer] | None = None
    path: str | None = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.server is not None and cls.path is not None:
            if isinstance(cls.server, HttpServer):
                cls.server.add_handler(cls)
            if isclass(cls.server) and issubclass(cls.server, HttpServer):
                if cls.server._instance is not None:
                    cls.server._instance.add_handler(cls)

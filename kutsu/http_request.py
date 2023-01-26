"""HTTP Requests"""
from __future__ import annotations

import base64
import json as jsonlib
import logging
import time
from typing import IO, Any, Generator

import rich.console
import rich.syntax
from httpx import AsyncByteStream, AsyncClient, Cookies, Request, Response

from .expressions import Node, evaluate
from .state import AsyncAction, State, StateArg
from .util import (
    bytes_to_readable,
    get_console,
    get_lexer_for_content_type,
    make_object_id,
)

Json = dict[str, 'Json'] | list['Json'] | str | int | float | bool | None

log = logging.getLogger(__name__)


class HttpRequest(AsyncAction):
    """Base Class for HTTP Requests

    Not very usable on its own. Subclass SyncHttpRequest or AsyncHttpRequest instead.

    """

    # TODO: we need good argument descriptions
    method: Node | str | None = 'GET'
    url: Node | str | None = ''
    params: Node | dict[str, str] | None = None
    headers: Node | dict[str, str] | dict[str, list[str]] | None = None
    authorization: Node | str | None = None
    auth_scheme: Node | str | None = None
    auth_token: Node | str | None = None
    auth_username: Node | str | None = None
    auth_password: Node | str | None = None
    content_type: Node | str | None = None
    accept: Node | str | list[str] | None = None
    content: Node | str | bytes | None = None
    data: Node | dict[str, str] | None = None
    files: Node | dict[str, bytes | IO[bytes]] | None = None
    json: Node | Json | None = None
    stream: Node | AsyncByteStream | None = None
    cookies: Node | Cookies | dict[str, str] | None = None

    # Raise error on non-ok response status code
    raise_error: bool = True

    # Automatically follow redirects
    follow_redirects: bool = True

    # Read cookies from this state variable and add them to the request
    read_cookies_from: str | None = 'cookies'

    # Save cookies from response to this state variable
    save_cookies_to: str | None = 'cookies'

    # Use this as default state. Incoming state overrides these values.
    defaults: State | dict[str, Any] | None = None
    input_state: State | None = None
    output_state: State | None = None

    # Name of this request, used for printing
    name: str | None = None

    # Whether to print various things
    show_input_state: bool = False
    show_output_state: bool = False
    show_request: bool = True
    show_request_headers: bool = False
    show_response: bool = True
    show_response_headers: bool = False

    show_response_body: bool | None = None
    show_headers: bool | None = None
    verbose: bool = False
    quiet: bool = False

    # Suppress text response bodies larger than this from printing
    show_max_body: int | None = 50 * 1024

    request: Request | None = None
    response: Response | None = None

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield 'method', self.method
        yield 'url', self.url

        if self.params is not None:
            yield 'params', self.params
        if self.headers is not None:
            yield 'headers', self.headers
        if self.authorization is not None:
            yield 'authorization', self.authorization
        if self.auth_scheme is not None:
            yield 'auth_scheme', self.auth_scheme
        if self.auth_token is not None:
            yield 'auth_token', self.auth_token
        if self.auth_username is not None:
            yield 'auth_username', self.auth_username
        if self.auth_username is not None:
            yield 'auth_password', self.auth_password
        if self.content_type is not None:
            yield 'content_type', self.content_type
        if self.accept is not None:
            yield 'accept', self.accept
        if self.content is not None:
            yield 'content', self.content
        if self.data is not None:
            yield 'data', self.data
        if self.files is not None:
            yield 'files', self.files
        if self.json is not None:
            yield 'json', self.json
        if self.stream is not None:
            yield 'stream', self.stream
        if self.cookies is not None:
            yield 'cookies', self.cookies

        yield 'defaults', self.defaults, None
        yield 'raise_error', self.raise_error, True
        yield 'follow_redirects', self.follow_redirects, True
        yield 'read_cookies_from', self.read_cookies_from, 'cookies'
        yield 'save_cookies_to', self.save_cookies_to, 'cookies'
        yield 'show_input_state', self.show_input_state, False
        yield 'show_output_state', self.show_output_state, False
        yield 'show_request', self.show_request, True
        yield 'show_request_headers', self.show_request_headers, False
        yield 'show_response', self.show_response, True
        yield 'show_response_headers', self.show_response_headers, False
        yield 'show_response_body', self.show_response_body, None
        yield 'show_headers', self.show_headers, None
        yield 'show_max_body', self.show_max_body, 50 * 1024
        yield 'verbose', self.verbose, False
        yield 'quiet', self.quiet, False
        yield 'request_prepared', self.request_prepared, False
        yield 'input_state', self.input_state, None
        yield 'request', self.request, None
        yield 'response_received', self.response_received, False
        yield 'response', self.response, None
        yield 'output_state', self.output_state, None

    def _update_parameters(
        self,
        url: Node | str | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        authorization: Node | str | None = None,
        auth_scheme: Node | str | None = None,
        auth_token: Node | str | None = None,
        auth_username: Node | str | None = None,
        auth_password: Node | str | None = None,
        content_type: Node | str | None = None,
        accept: Node | str | list[str] | None = None,
        content: Node | str | bytes | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Cookies | dict[str, str] | None = None,
        defaults: State | dict[str, Any] | None = None,
        raise_error: bool | None = None,
        follow_redirects: bool | None = None,
        name: str | None = None,
        read_cookies_from: str | None = None,
        save_cookies_to: str | None = None,
        show_input_state: bool | None = None,
        show_output_state: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_body: bool | None = None,
        show_response_headers: bool | None = None,
        show_max_body: int | None = None,
        show_headers: bool | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
    ) -> None:
        if url is not None:
            self.url = url
        if params is not None:
            self.params = params
        if method is not None:
            self.method = method
        if headers is not None:
            self.headers = headers
        if authorization is not None:
            self.authorization = authorization
        if auth_scheme is not None:
            self.auth_scheme = auth_scheme
        if auth_token is not None:
            self.auth_token = auth_token
        if auth_username is not None:
            self.auth_username = auth_username
        if auth_password is not None:
            self.auth_password = auth_password
        if content_type is not None:
            self.content_type = content_type
        if accept is not None:
            self.accept = accept
        if content is not None:
            self.content = content
        if data is not None:
            self.data = data
        if files is not None:
            self.files = files
        if json is not None:
            self.json = json
        if stream is not None:
            self.stream = stream
        if cookies is not None:
            self.cookies = cookies
        if defaults is not None:
            self.defaults = defaults
        if raise_error is not None:
            self.raise_error = raise_error
        if follow_redirects is not None:
            self.follow_redirects = follow_redirects
        if name is not None:
            self.name = name
        if read_cookies_from is not None:
            self.read_cookies_from = read_cookies_from
        if save_cookies_to is not None:
            self.save_cookies_to = save_cookies_to
        if verbose is True:
            self.verbose = True
        if self.verbose is True:
            # TODO: also print settings
            self.show_input_state = True
            self.show_output_state = True
            self.show_request = True
            self.show_request_headers = True
            self.show_response = True
            self.show_response_headers = True
        if show_input_state is not None:
            self.show_input_state = show_input_state
        if show_output_state is not None:
            self.show_output_state = show_output_state
        if show_headers is not None:
            self.show_headers = show_headers
        if self.show_headers is not None:
            self.show_request_headers = self.show_headers
            self.show_response_headers = self.show_headers
        if show_request is not None:
            self.show_request = show_request
        if show_request_headers is not None:
            self.show_request_headers = show_request_headers
        if show_response is not None:
            self.show_response = show_response
        if show_response_body is not None:
            self.show_response_body = show_response_body
        if self.show_response_body is not None:
            if self.show_response_body is True:
                self.show_max_body = None
        if show_response_headers is not None:
            self.show_response_headers = show_response_headers
        if show_max_body is not None:
            self.show_max_body = show_max_body
        if quiet is True:
            self.quiet = True
        if self.quiet is True:
            self.show_input_state = False
            self.show_output_state = False
            self.show_request = False
            self.show_request_headers = False
            self.show_response = False
            self.show_response_headers = False

    def __init__(
        self,
        url: Node | str | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        authorization: Node | str | None = None,
        auth_scheme: Node | str | None = None,
        auth_token: Node | str | None = None,
        auth_username: Node | str | None = None,
        auth_password: Node | str | None = None,
        content_type: Node | str | None = None,
        accept: Node | str | list[str] | None = None,
        content: Node | str | bytes | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Cookies | dict[str, str] | None = None,
        defaults: State | dict[str, Any] | None = None,
        raise_error: bool | None = None,
        follow_redirects: bool | None = None,
        name: str | None = None,
        read_cookies_from: str | None = None,
        save_cookies_to: str | None = None,
        show_input_state: bool | None = None,
        show_output_state: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
    ) -> None:
        # TODO: docstring with argument descriptions
        super().__init__()
        self._update_parameters(
            url=url,
            method=method,
            params=params,
            headers=headers,
            authorization=authorization,
            auth_scheme=auth_scheme,
            auth_token=auth_token,
            auth_username=auth_username,
            auth_password=auth_password,
            content_type=content_type,
            accept=accept,
            content=content,
            data=data,
            files=files,
            json=json,
            stream=stream,
            cookies=cookies,
            defaults=defaults,
            raise_error=raise_error,
            follow_redirects=follow_redirects,
            name=name,
            read_cookies_from=read_cookies_from,
            save_cookies_to=save_cookies_to,
            show_input_state=show_input_state,
            show_output_state=show_output_state,
            show_request=show_request,
            show_request_headers=show_request_headers,
            show_response=show_response,
            show_response_headers=show_response_headers,
            show_response_body=show_response_body,
            show_headers=show_headers,
            show_max_body=show_max_body,
            verbose=verbose,
            quiet=quiet,
        )

    @property
    def request_prepared(self) -> bool:
        return self.request is not None

    @property
    def response_received(self) -> bool:
        return self.response is not None

    def reset(self) -> None:
        self.request = None
        self.response = None

    def prepare(self, state: State | dict[str, Any] | None = None) -> Request:
        """Prepare the request.

        This method is called by the HTTP client before sending the request.
        You may also call it manually to prepare the request before sending it,
        and possibly mutate self.request before sending.
        """
        self.input_state = State(state)
        s = self.input_state(State(self.defaults))
        self.request = Request(
            method=self._prepare_method(s),
            url=self._prepare_url(s),
            params=self._prepare_params(s),
            headers=self._prepare_headers(s),
            content=self._prepare_content(s),
            data=self._prepare_data(s),
            files=self._prepare_files(s),
            json=self._prepare_json(s),
            stream=self._prepare_stream(s),
            cookies=self._prepare_cookies(s),
        )
        # FIXME: mask_secrets
        # log.info(self.request)
        return self.request

    def _prepare_method(self, state: State) -> str:
        return str(evaluate(self.method, state))

    def _prepare_url(self, state: State) -> str:
        return str(evaluate(self.url, state))

    def _prepare_params(self, state: State) -> dict[str, str] | None:
        params = evaluate(self.params, state)
        if params is None:
            return None
        if not isinstance(params, dict):
            raise TypeError(f'params must be a dict, not {type(params)}')
        return params

    def _prepare_authorization_header(self, state: State) -> str | None:
        authorization = evaluate(self.authorization, state)
        if authorization is not None:
            if not isinstance(authorization, str):
                raise TypeError(f'authorization must be a str, not {type(authorization)}')
            return authorization

        scheme = evaluate(self.auth_scheme, state)
        token = evaluate(self.auth_token, state)
        username = evaluate(self.auth_username, state)
        password = evaluate(self.auth_password, state)

        if token is None and None not in (username, password):
            if scheme is None:
                scheme = 'Basic'
            if scheme == 'Basic':
                token = base64.b64encode('f{username}:{password}'.encode('utf-8'))
            else:
                raise ValueError(f'Unsupported authorization scheme: {scheme}')

        if scheme is not None and token is not None:
            return f'{scheme} {token}'

        return None

    def _prepare_headers(self, state: State) -> dict[str, str] | None:
        headers = evaluate(self.headers or {}, state)
        if not isinstance(headers, dict):
            raise TypeError(f'headers must be a dict, not {type(headers)}')
        if self.content_type is not None:
            headers['Content-Type'] = self.content_type
        authorization = self._prepare_authorization_header(state)
        if authorization is not None:
            headers['Authorization'] = authorization
        if self.accept is not None:
            headers['Accept'] = self.accept
        for k, v in headers.items():
            if isinstance(v, list):
                headers[k] = ', '.join(v)
        return headers

    def _prepare_content(self, state: State) -> str | bytes | None:
        content = evaluate(self.content, state)
        if content is None:
            return None
        if isinstance(content, (str, bytes)):
            return content
        raise TypeError(f'content must be a str or bytes, not {type(content)}')

    def _prepare_data(self, state: State) -> dict[str, str] | None:
        data = evaluate(self.data, state)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise TypeError(f'data must be a dict, not {type(data)}')
        return data

    def _prepare_files(self, state: State) -> dict[str, bytes | IO[bytes]] | None:
        files = evaluate(self.files, state)
        if files is None:
            return None
        if not isinstance(files, dict):
            raise TypeError(f'files must be a dict, not {type(files)}')
        return files

    def _prepare_json(self, state: State) -> Json | None:
        json = evaluate(self.json, state)
        if json is None:
            return None
        if isinstance(json, (dict, list, str, int, float, bool, type(None))):
            return json
        raise TypeError(
            f'json must be a dict, list, str, int, float, bool, or None, not {type(json)}'
        )

    def _prepare_stream(self, state: State) -> AsyncByteStream | None:
        stream = evaluate(self.stream, state)
        if stream is None:
            return None
        if isinstance(stream, AsyncByteStream):
            return stream
        raise TypeError(f'stream must be an AsyncByteStream, not {type(stream)}')

    def _prepare_cookies(self, state: State) -> Cookies | None:
        cookies = Cookies()

        # Read cookies from state if present
        if self.read_cookies_from and self.read_cookies_from in state:
            state_cookies = state[self.read_cookies_from]
            if state_cookies is not None:
                cookies.update(state_cookies)

        # Add cookies from self.cookies
        self_cookies = evaluate(self.cookies, state)
        if self_cookies is not None:
            if not isinstance(cookies, (dict, Cookies)):
                raise TypeError(f'cookies must be a dict or Cookies, not {type(cookies)}')
            cookies.update(self_cookies)

        return cookies or None

    def _prepare_call(self, state: State, /, **kwargs: Any) -> Request:
        for k, v in kwargs.items():
            state[k] = v
        if not self.request:
            # Request was not prepared yet
            request = self.prepare(state)
        elif self.input_state != state:
            # State has changed since request was prepared
            request = self.prepare(state)
        else:
            # Use the prepared request
            request = self.request
        return request

    async def call_async(
        self,
        state: StateArg | None = None,
        /,
        url: Node | str | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        authorization: Node | str | None = None,
        auth_scheme: Node | str | None = None,
        auth_token: Node | str | None = None,
        auth_username: Node | str | None = None,
        auth_password: Node | str | None = None,
        content_type: Node | str | None = None,
        accept: Node | str | list[str] | None = None,
        content: Node | str | bytes | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Cookies | dict[str, str] | None = None,
        defaults: State | dict[str, Any] | None = None,
        raise_error: bool | None = None,
        follow_redirects: bool | None = None,
        name: str | None = None,
        read_cookies_from: str | None = None,
        save_cookies_to: str | None = None,
        show_input_state: bool | None = None,
        show_output_state: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
    ) -> State:
        state = await super().call_async(State(state))
        # TODO: we should probably not update parameters, but instead use overridden values
        # TODO: for just this call - so: take instance values, make those into a runtime
        # TODO: inputs object and update that object with there values, then pass the
        # TODO: object to the request call for actual use
        self._update_parameters(
            url=url,
            method=method,
            params=params,
            headers=headers,
            authorization=authorization,
            auth_scheme=auth_scheme,
            auth_token=auth_token,
            auth_username=auth_username,
            auth_password=auth_password,
            content_type=content_type,
            accept=accept,
            content=content,
            data=data,
            files=files,
            json=json,
            stream=stream,
            cookies=cookies,
            defaults=defaults,
            raise_error=raise_error,
            follow_redirects=follow_redirects,
            name=name,
            read_cookies_from=read_cookies_from,
            save_cookies_to=save_cookies_to,
            show_input_state=show_input_state,
            show_output_state=show_output_state,
            show_request=show_request,
            show_request_headers=show_request_headers,
            show_response=show_response,
            show_response_headers=show_response_headers,
            show_response_body=show_response_body,
            show_headers=show_headers,
            show_max_body=show_max_body,
            verbose=verbose,
            quiet=quiet,
        )
        request = self._prepare_call(state)
        self._print_input_state()
        self._print_request()
        async with AsyncClient() as client:
            response = await client.send(request, follow_redirects=self.follow_redirects)

        return self._process_response(state, response)

    def _process_response(self, state: State, res: Response) -> State:
        self.response = res
        # log.info(self.response)
        self._print_response()
        if self.raise_error:
            res.raise_for_status()
        if self.save_cookies_to:
            state[self.save_cookies_to] = res.cookies
        self.output_state = State(state)
        self._print_output_state()
        self._print_end_request_processing()
        # TODO: on_response hook
        # TODO: json hook
        return state

    def _make_request_name(self) -> str:
        if self.name:
            return self.name
        # TODO: object id should change from call to call
        return f'{self.__class__.__name__}<{make_object_id(self)}>'

    def _print_request(self) -> None:
        # TODO: mask secrets
        if not self.show_request and not self.show_request_headers:
            return
        name = self._make_request_name()
        console = get_console(no_color=True)
        syntax_console = get_console()
        if self.request is None:
            console.print(f'[italic]{name} request not prepared[/italic]')
            return
        req = self.request
        now = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
        # console.rule(f'{name} Fetch {now}')
        console.print(f'[bold]*** Fetch {now} [{name}][/bold]')
        status = f'{req.method} {req.url} HTTP/1.1'
        headers = [
            f'{name.decode("ascii")}: {value.decode("ascii")}'
            for name, value in req.headers.raw
        ]
        hdrs = '\n'.join(headers)
        if self.show_request_headers:
            s = f'{status}\n{hdrs}\n'
        else:
            s = status
        syntax_console.print(
            rich.syntax.Syntax(
                f'{s}',
                'http',
                theme='ansi_dark',
                word_wrap=True,
            )
        )
        content = req.content.decode('utf-8')
        if content:
            # console.rule(characters='–')
            syntax_console.print(content)

    def _print_response(self) -> None:
        if not self.show_response and not self.show_response_headers:
            return
        name = self._make_request_name()
        console = get_console(no_color=True)
        syntax_console = get_console()
        if self.response is None:
            console.print(f'[italic]{name} no response received[/italic]')
            return
        res = self.response

        if self.show_request_headers:
            console.print(f'[bold]*** Response [{name}][/bold]')
        downloaded = bytes_to_readable(res.num_bytes_downloaded)
        elapsed = f'{int(res.elapsed.total_seconds()*1000)} ms'
        # console.rule(f'{name} Response {downloaded} in {elapsed}')
        status = f'{res.http_version} {res.status_code} {res.reason_phrase}'
        headers = [
            f'{name.decode("ascii")}: {value.decode("ascii")}'
            for name, value in res.headers.raw
        ]
        hdrs = '\n'.join(headers)
        if self.show_response_headers:
            s = f'{status}\n{hdrs}\n'
        else:
            s = status
        syntax_console.print(
            rich.syntax.Syntax(f'{s}', 'http', theme='ansi_dark', word_wrap=True)
        )

        # console.rule(characters='–')
        lexer_name = get_lexer_for_content_type(res.headers.get('Content-Type'))
        if lexer_name:
            if (self.show_max_body is not None and len(res.text) > self.show_max_body):
                max_ = bytes_to_readable(float(self.show_max_body))
                console.print(
                    f'[italic]Response body over {max_} suppressed by default[/italic]'
                )
            else:
                if lexer_name.lower() == 'json':
                    try:
                        data = res.json()
                        text = jsonlib.dumps(data, indent=4)
                    except ValueError:
                        text = res.text
                else:
                    text = res.text

                syntax = rich.syntax.Syntax(
                    text, lexer_name, theme='ansi_dark', word_wrap=True
                )
                syntax_console.print(syntax)
        else:
            console.print(f'<{len(res.content)} bytes of binary data>')
        console.print(
            f'[bold]*** Response {downloaded} received in {elapsed} [{name}][/bold]'
        )

    def _print_input_state(self) -> None:
        if not self.show_input_state:
            return
        name = self._make_request_name()
        console = get_console(no_color=True)
        # console.rule(f'{name} Input State')
        console.print(f'[bold]*** Input State [{name}][/bold]')
        if self.input_state is None:
            console.print('[italic]No input state[/italic]')
            return
        self._print_state(self.input_state)

    def _print_output_state(self) -> None:
        if not self.show_output_state:
            return
        name = self._make_request_name()
        console = get_console(no_color=True)
        # console.rule(f'{name} Output State')
        console.print(f'[bold]*** Output State [{name}][/bold]')
        if self.output_state is None:
            console.print('[italic]No output state[/italic]')
            return
        self._print_state(self.output_state)

    def _print_state(self, state: State) -> None:
        console = get_console(soft_wrap=True)
        if len(state) == 0:
            console.print('[italic]Empty state[/italic]')
        else:
            for k in state:
                v = state[k]
                if k in {self.read_cookies_from, self.save_cookies_to}:
                    continue
                console.print(f'{k}={v}')
            if (
                self.read_cookies_from and self.read_cookies_from in state
                or self.save_cookies_to and self.save_cookies_to in state
            ):
                if self.read_cookies_from and self.read_cookies_from in state:
                    cookies = state[self.read_cookies_from]
                else:
                    assert self.save_cookies_to is not None
                    cookies = state[self.save_cookies_to]
                if cookies:
                    print('cookies=')
                    for cookie in cookies.jar:
                        console.print(
                            # f'- {repr(cookie)}'
                            f'<Cookie {cookie.name}="{cookie.value}" for {cookie.domain} {cookie.path}>',
                            # no_wrap=True,
                        )
                else:
                    console.print('[italic]Empty cookie Jar[/italic]')

    def _print_end_request_processing(self) -> None:
        if self.show_output_state:
            console = get_console(no_color=True)
            # console.rule()
            name = self._make_request_name()
            console.print(f'[bold]*** End Request Processing {name}[/bold]')

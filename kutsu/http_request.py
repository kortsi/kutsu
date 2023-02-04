"""HTTP Requests"""
from __future__ import annotations

import base64
import json as jsonlib
import logging
import time
from typing import IO, Any, Generator, NamedTuple, TypeVar

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

T = TypeVar('T')
U = TypeVar('U')

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

    _prepared_config: RequestConfig | None = None
    _prepared_data: RequestData | None = None
    _prepared_data_masked: RequestData | None = None

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
        if verbose is not None:
            self.verbose = verbose
        if show_input_state is not None:
            self.show_input_state = show_input_state
        if show_output_state is not None:
            self.show_output_state = show_output_state
        if show_headers is not None:
            self.show_headers = show_headers
        if show_request is not None:
            self.show_request = show_request
        if show_request_headers is not None:
            self.show_request_headers = show_request_headers
        if show_response is not None:
            self.show_response = show_response
        if show_response_body is not None:
            self.show_response_body = show_response_body
        if show_response_headers is not None:
            self.show_response_headers = show_response_headers
        if show_max_body is not None:
            self.show_max_body = show_max_body
        if quiet is not None:
            self.quiet = quiet

    @property
    def request_prepared(self) -> bool:
        return self.request is not None

    @property
    def response_received(self) -> bool:
        return self.response is not None

    def reset(self) -> None:
        self.request = None
        self.response = None
        self.input_state = None
        self.output_state = None
        self._prepared_config = None
        self._prepared_data = None

    def _input_state_changed(self, state: State) -> bool:
        if self.input_state is None:
            return True
        if set(state) != set(self.input_state):
            return True
        for k in state:
            if state[k] is not self.input_state[k]:
                return True
        return False

    def _config_changed(self, config: RequestConfig) -> bool:
        if self._prepared_config is None:
            return True
        for i, v in enumerate(config):
            if v is not self._prepared_config[i]:
                return True
        return False

    def _prepare_call(
        self,
        state: State,
        config: RequestConfig | None = None,
        /,
        **kwargs: Any
    ) -> Request:
        if config is None:
            config = RequestConfig()
        for k, v in kwargs.items():
            state[k] = v
        if not self.request:
            request = self.prepare(state, config=config)
        elif self._input_state_changed(state):
            request = self.prepare(state, config=config)
        elif self._config_changed(config):
            request = self.prepare(state, config=config)
        else:
            # Use the prepared request
            request = self.request
        return request

    def prepare(
        self,
        state: State | dict[str, Any] | None = None,
        config: RequestConfig | None = None,
    ) -> Request:
        """Prepare the request.

        This method is called before sending the request.
        You may also call it manually to prepare the request before sending it,
        and possibly mutate self.request before sending.
        """
        if config is None:
            config = self._make_request_config()
        self._prepared_config = config
        self.input_state = State(state)
        s = self.input_state(State(config.defaults))
        data = RequestData(
            method=self._prepare_method(s, config),
            url=self._prepare_url(s, config),
            params=self._prepare_params(s, config),
            headers=self._prepare_headers(s, config),
            content=self._prepare_content(s, config),
            data=self._prepare_data(s, config),
            files=self._prepare_files(s, config),
            json=self._prepare_json(s, config),
            stream=self._prepare_stream(s, config),
            cookies=self._prepare_cookies(s, config),
        )
        # TODO: use masked data for printing
        self._prepared_data_masked = RequestData(
            method=self._prepare_method(s, config, mask_secrets=True),
            url=self._prepare_url(s, config, mask_secrets=True),
            params=self._prepare_params(s, config, mask_secrets=True),
            headers=self._prepare_headers(s, config, mask_secrets=True),
            content=self._prepare_content(s, config, mask_secrets=True),
            data=self._prepare_data(s, config, mask_secrets=True),
            files=self._prepare_files(s, config, mask_secrets=True),
            json=self._prepare_json(s, config, mask_secrets=True),
            stream=self._prepare_stream(s, config, mask_secrets=True),
            cookies=self._prepare_cookies(s, config, mask_secrets=True),
        )
        self._prepared_data = data
        self.request = Request(**data._asdict())
        return self.request

    def _prepare_method(
        self, state: State, config: RequestConfig, mask_secrets: bool = False
    ) -> str:
        return str(evaluate(config.method, state, mask_secrets=mask_secrets))

    def _prepare_url(
        self, state: State, config: RequestConfig, mask_secrets: bool = False
    ) -> str:
        return str(evaluate(config.url, state, mask_secrets=mask_secrets))

    def _prepare_params(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> dict[str, str] | None:
        params = evaluate(config.params, state, mask_secrets=mask_secrets)
        if params is None:
            return None
        if not isinstance(params, dict):
            raise TypeError(f'params must be a dict, not {type(params)}')
        return params

    def _prepare_authorization_header(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> str | None:
        authorization = evaluate(
            config.authorization, state, mask_secrets=mask_secrets, as_str=True
        )
        if authorization is not None:
            if not isinstance(authorization, str):
                raise TypeError(f'authorization must be a str, not {type(authorization)}')
            return authorization

        scheme = evaluate(
            config.auth_scheme, state, mask_secrets=mask_secrets, as_str=True
        )
        token = evaluate(config.auth_token, state, mask_secrets=mask_secrets, as_str=True)
        username = evaluate(
            config.auth_username, state, mask_secrets=mask_secrets, as_str=True
        )
        password = evaluate(
            config.auth_password, state, mask_secrets=mask_secrets, as_str=True
        )

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

    def _prepare_headers(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> dict[str, str] | None:
        headers = evaluate(config.headers or {}, state, mask_secrets=mask_secrets)
        if not isinstance(headers, dict):
            raise TypeError(f'headers must be a dict, not {type(headers)}')
        content_type = evaluate(
            config.content_type, state, mask_secrets=mask_secrets, as_str=True
        )
        if content_type is not None:
            headers['Content-Type'] = content_type
        authorization = self._prepare_authorization_header(
            state, config, mask_secrets=mask_secrets
        )
        if authorization is not None:
            headers['Authorization'] = authorization
        accept = evaluate(config.accept, state, mask_secrets=mask_secrets, as_str=True)
        if accept is not None:
            headers['Accept'] = accept
        for k, v in headers.items():
            if isinstance(v, list):
                headers[k] = ', '.join(v)
        return headers

    def _prepare_content(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> str | bytes | None:
        content = evaluate(config.content, state, mask_secrets=mask_secrets, as_str=True)
        if content is None:
            return None
        if isinstance(content, (str, bytes)):
            return content
        raise TypeError(f'content must be a str or bytes, not {type(content)}')

    def _prepare_data(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> dict[str, str] | None:
        data = evaluate(config.data, state, mask_secrets=mask_secrets)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise TypeError(f'data must be a dict, not {type(data)}')
        return data

    def _prepare_files(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> dict[str, bytes | IO[bytes]] | None:
        files = evaluate(config.files, state, mask_secrets=mask_secrets)
        if files is None:
            return None
        if not isinstance(files, dict):
            raise TypeError(f'files must be a dict, not {type(files)}')
        return files

    def _prepare_json(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> Json | None:
        json = evaluate(config.json, state, mask_secrets=mask_secrets)
        if json is None:
            return None
        if isinstance(json, (dict, list, str, int, float, bool, type(None))):
            return json
        raise TypeError(
            f'json must be a dict, list, str, int, float, bool, or None, not {type(json)}'
        )

    def _prepare_stream(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> AsyncByteStream | None:
        stream = evaluate(config.stream, state, mask_secrets=mask_secrets)
        if stream is None:
            return None
        if isinstance(stream, AsyncByteStream):
            return stream
        raise TypeError(f'stream must be an AsyncByteStream, not {type(stream)}')

    def _prepare_cookies(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> Cookies | None:
        cookies = Cookies()
        read_cookies_from = evaluate(
            config.read_cookies_from, state, mask_secrets=mask_secrets, as_str=True
        )

        # Read cookies from state if present
        if read_cookies_from and read_cookies_from in state:
            state_cookies = state[read_cookies_from]
            if state_cookies is not None:
                cookies.update(state_cookies)

        # Add cookies from self.cookies
        self_cookies = evaluate(
            config.cookies, state, mask_secrets=mask_secrets, as_str=True
        )
        if self_cookies is not None:
            if not isinstance(cookies, (dict, Cookies)):
                raise TypeError(f'cookies must be a dict or Cookies, not {type(cookies)}')
            cookies.update(self_cookies)

        return cookies or None

    def _make_request_config(
        self,
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
        data: Node | dict[str, str] | None = None,
        files: Node | dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Node | Cookies | dict[str, str] | None = None,
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
    ) -> RequestConfig:

        def choose(first: T, second: U) -> T | U:
            return first if first is not None else second

        # First evaluate instance defaults
        if self.verbose is True:
            show_input_state = True
            show_output_state = True
            show_request = True
            show_request_headers = True
            show_response = True
            show_response_headers = True
        if self.show_headers is not None:
            show_request_headers = self.show_headers
            show_response_headers = self.show_headers
        if self.show_response_body is True:
            show_max_body = None
        if self.quiet is True:
            show_input_state = False
            show_output_state = False
            show_request = False
            show_request_headers = False
            show_response = False
            show_response_headers = False

        # Then evaluate function args
        if verbose is True:
            show_input_state = True
            show_output_state = True
            show_request = True
            show_request_headers = True
            show_response = True
            show_response_headers = True
        if show_headers is not None:
            show_request_headers = show_headers
            show_response_headers = show_headers
        if show_response_body is True:
            show_max_body = None
        if quiet is True:
            show_input_state = False
            show_output_state = False
            show_request = False
            show_request_headers = False
            show_response = False
            show_response_headers = False

        return RequestConfig(
            url=choose(url, self.url),
            method=choose(method, self.method),
            params=choose(params, self.params),
            headers=choose(headers, self.headers),
            authorization=choose(authorization, self.authorization),
            auth_scheme=choose(auth_scheme, self.auth_scheme),
            auth_token=choose(auth_token, self.auth_token),
            auth_username=choose(auth_username, self.auth_username),
            auth_password=choose(auth_password, self.auth_password),
            content_type=choose(content_type, self.content_type),
            accept=choose(accept, self.accept),
            content=choose(content, self.content),
            data=choose(data, self.data),
            files=choose(files, self.files),
            json=choose(json, self.json),
            stream=choose(stream, self.stream),
            cookies=choose(cookies, self.cookies),
            defaults=choose(defaults, self.defaults),
            raise_error=choose(raise_error, self.raise_error),
            follow_redirects=choose(follow_redirects, self.follow_redirects),
            name=choose(name, self.name),
            read_cookies_from=choose(read_cookies_from, self.read_cookies_from),
            save_cookies_to=choose(save_cookies_to, self.save_cookies_to),
            show_input_state=choose(show_input_state, self.show_input_state),
            show_output_state=choose(show_output_state, self.show_output_state),
            show_request=choose(show_request, self.show_request),
            show_request_headers=choose(show_request_headers, self.show_request_headers),
            show_response=choose(show_response, self.show_response),
            show_response_headers=choose(
                show_response_headers, self.show_response_headers
            ),
            show_response_body=choose(show_response_body, self.show_response_body),
            show_headers=choose(show_headers, self.show_headers),
            show_max_body=choose(show_max_body, self.show_max_body),
            verbose=choose(verbose, self.verbose),
            quiet=choose(quiet, self.quiet),
        )

    # This is here just for REPL reference and tab completion of argument names
    def __call__(
        self,
        state: StateArg | None,
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
        data: Node | dict[str, str] | None = None,
        files: Node | dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Node | Cookies | dict[str, str] | None = None,
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
        **kwargs: Any,
    ) -> State:
        return super().__call__(
            state,
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
            **kwargs,
        )

    async def call_async(
        self,
        state: StateArg | None,
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
        data: Node | dict[str, str] | None = None,
        files: Node | dict[str, bytes | IO[bytes]] | None = None,
        json: Node | Json | None = None,
        stream: Node | AsyncByteStream | None = None,
        cookies: Node | Cookies | dict[str, str] | None = None,
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
        **kwargs: Any,
    ) -> State:
        state = await super().call_async(State(state), **kwargs)
        config = self._make_request_config(
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

        request = self._prepare_call(state, config)
        self._print_input_state(config)
        # TODO: also print prepared config
        self._print_request(config)
        async with AsyncClient() as client:
            response = await client.send(
                request, follow_redirects=follow_redirects or self.follow_redirects
            )

        return self._process_response(state, response, config)

    def _process_response(
        self, state: State, res: Response, config: RequestConfig
    ) -> State:
        self.response = res
        self._print_response(config)
        if config.raise_error:
            res.raise_for_status()
        if config.save_cookies_to and res.cookies:
            if config.save_cookies_to in state:
                if not isinstance(state[config.save_cookies_to], (dict, Cookies)):
                    raise TypeError(
                        'cookies must be a dict or Cookies, '
                        f'not {type(state[config.save_cookies_to])}'
                    )
                state[config.save_cookies_to].update(res.cookies)
            else:
                state[config.save_cookies_to] = res.cookies
        self.output_state = State(state)
        self._print_output_state(config)
        self._print_end_request_processing(config)
        # TODO: on_response hook
        # TODO: json hook
        return state

    def _make_request_name(self, config: RequestConfig) -> str:
        if config.name:
            return config.name
        return f'{self.__class__.__name__}<{make_object_id(self)}>'

    def _print_request(self, config: RequestConfig) -> None:
        # TODO: mask secrets
        show_request = config.show_request
        show_request_headers = config.show_request_headers
        if not show_request and not show_request_headers:
            return
        name = self._make_request_name(config)
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
        if show_request_headers:
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

    def _print_response(self, config: RequestConfig) -> None:
        show_response = config.show_response
        show_response_headers = config.show_response_headers
        show_request_headers = config.show_request_headers
        show_max_body = config.show_max_body
        if not show_response and not show_response_headers:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True)
        syntax_console = get_console()
        if self.response is None:
            console.print(f'[italic]{name} no response received[/italic]')
            return
        res = self.response

        if show_request_headers:
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
        if show_response_headers:
            s = f'{status}\n{hdrs}\n'
        else:
            s = status
        syntax_console.print(
            rich.syntax.Syntax(f'{s}', 'http', theme='ansi_dark', word_wrap=True)
        )

        # console.rule(characters='–')
        lexer_name = get_lexer_for_content_type(res.headers.get('Content-Type'))
        if lexer_name:
            if (show_max_body is not None and len(res.text) > show_max_body):
                max_ = bytes_to_readable(float(show_max_body))
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

    def _print_input_state(self, config: RequestConfig) -> None:
        show_input_state = config.show_input_state
        if not show_input_state:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True)
        # console.rule(f'{name} Input State')
        console.print(f'[bold]*** Input State [{name}][/bold]')
        if self.input_state is None:
            console.print('[italic]No input state[/italic]')
            return
        self._print_state(self.input_state, config)

    def _print_output_state(self, config: RequestConfig) -> None:
        show_output_state = config.show_output_state
        if not show_output_state:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True)
        # console.rule(f'{name} Output State')
        console.print(f'[bold]*** Output State [{name}][/bold]')
        if self.output_state is None:
            console.print('[italic]No output state[/italic]')
            return
        self._print_state(self.output_state, config)

    def _print_state(self, state: State, config: RequestConfig) -> None:
        read_cookies_from = config.read_cookies_from
        save_cookies_to = config.save_cookies_to
        console = get_console(soft_wrap=True)
        if len(state) == 0:
            console.print('[italic]Empty state[/italic]')
        else:
            for k in state:
                v = state[k]
                if k in {read_cookies_from, save_cookies_to}:
                    continue
                console.print(f'{k}={v}')
            if (
                read_cookies_from and read_cookies_from in state
                or save_cookies_to and save_cookies_to in state
            ):
                if read_cookies_from and read_cookies_from in state:
                    cookies = state[read_cookies_from]
                else:
                    assert save_cookies_to is not None
                    cookies = state[save_cookies_to]
                if cookies:
                    print('cookies=')
                    for cookie in cookies.jar:
                        console.print(
                            # f'- {repr(cookie)}'
                            f'- {cookie.name}="{cookie.value}" for {cookie.domain} {cookie.path}',
                            # no_wrap=True,
                        )
                else:
                    console.print('[italic]Empty cookie Jar[/italic]')
        console.print('')

    def _print_end_request_processing(self, config: RequestConfig) -> None:
        show_output_state = config.show_output_state
        if show_output_state:
            console = get_console(no_color=True)
            # console.rule()
            name = self._make_request_name(config)
            console.print(f'[bold]*** End Request Processing {name}[/bold]')


class RequestConfig(NamedTuple):
    url: Node | str | None = HttpRequest.url
    method: Node | str | None = HttpRequest.method
    params: Node | dict[str, str] | None = HttpRequest.params
    headers: Node | dict[str, str] | dict[str, list[str]] | None = HttpRequest.headers
    authorization: Node | str | None = HttpRequest.authorization
    auth_scheme: Node | str | None = HttpRequest.auth_scheme
    auth_token: Node | str | None = HttpRequest.auth_token
    auth_username: Node | str | None = HttpRequest.auth_username
    auth_password: Node | str | None = HttpRequest.auth_password
    content_type: Node | str | None = HttpRequest.content_type
    accept: Node | str | list[str] | None = HttpRequest.accept
    content: Node | str | bytes | None = HttpRequest.content
    data: Node | dict[str, str] | None = HttpRequest.data
    files: Node | dict[str, bytes | IO[bytes]] | None = HttpRequest.files
    json: Node | Json | None = HttpRequest.json
    stream: Node | AsyncByteStream | None = HttpRequest.stream
    cookies: Node | Cookies | dict[str, str] | None = HttpRequest.cookies
    defaults: State | dict[str, Any] | None = HttpRequest.defaults
    raise_error: bool | None = HttpRequest.raise_error
    follow_redirects: bool | None = HttpRequest.follow_redirects
    name: str | None = HttpRequest.name
    read_cookies_from: str | None = HttpRequest.read_cookies_from
    save_cookies_to: str | None = HttpRequest.save_cookies_to
    show_input_state: bool | None = HttpRequest.show_input_state
    show_output_state: bool | None = HttpRequest.show_output_state
    show_request: bool | None = HttpRequest.show_request
    show_request_headers: bool | None = HttpRequest.show_request_headers
    show_response: bool | None = HttpRequest.show_response
    show_response_headers: bool | None = HttpRequest.show_request_headers
    show_response_body: bool | None = HttpRequest.show_response_body
    show_headers: bool | None = HttpRequest.show_headers
    show_max_body: int | None = HttpRequest.show_max_body
    verbose: bool | None = HttpRequest.verbose
    quiet: bool | None = HttpRequest.quiet


class RequestData(NamedTuple):
    method: str | None = None
    url: str | None = None
    params: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    content: str | bytes | None = None
    data: dict[str, str] | None = None
    files: dict[str, bytes | IO[bytes]] | None = None
    json: Json | None = None
    stream: AsyncByteStream | None = None
    cookies: Cookies | None = None

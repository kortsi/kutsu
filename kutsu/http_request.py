"""HTTP Requests"""
from __future__ import annotations

import asyncio
import base64
import functools
import json as jsonlib
import logging
import time
import typing
from contextlib import closing
from typing import IO, Any, Callable, Generator, NamedTuple, TypeVar

import httpcore
import httpx
import pygments
import rich.console
import rich.syntax
from httpx import AsyncByteStream, AsyncClient, Auth, Cookies, Request, Response

from .expressions import Node, Secret, _evaluate, evaluate
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
JsonTemplate = dict[
    str, 'JsonTemplate'] | list['JsonTemplate'] | str | int | float | bool | Node | None

log = logging.getLogger(__name__)


class AwsSigV4Auth(Node):
    """AWS Sigv4 Authentication (AWS4-HMAC-SHA256)

    With static access key and secret key:

        auth = AwsSigv4Auth(
            service='execute-api',
            region='us-east-1',
            access_key='AWS_ACCESS_KEY_ID',
            secret_key='AWS_SECRET_ACCESS_KEY',
        )

    With STS credentials:

        import boto3
        credentials = boto3.Session(profile_name="<profile>").get_credentials()
        auth = AwsSigv4Auth(
            service='execute-api',
            region='us-east-1',
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            token=credentials.token,
        )
    """

    def __init__(
        self,
        access_key: Node | str,
        secret_key: Node | str,
        service: Node | str,
        region: Node | str,
        token: Node | str | None = None,
    ) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token
        self.service = service
        self.region = region

    def __call__(
        self,
        state: State,
        mask_secrets: bool = False,
        context: dict[type, Any] | None = None,
    ) -> httpx.Auth:
        try:
            from httpx_auth_awssigv4 import SigV4Auth
        except ImportError as e:
            raise ImportError(
                'To use AwsSigV4Auth, you need to install kutsu[aws] or httpx-auth-aws-sigv4'
            ) from e

        mask = mask_secrets
        ctx = context
        region = _evaluate(self.region, state, mask_secrets=mask, context=ctx)
        service = _evaluate(self.service, state, mask_secrets=mask, context=ctx)
        access_key = _evaluate(self.access_key, state, mask_secrets=mask, context=ctx)
        secret_key = _evaluate(self.secret_key, state, mask_secrets=mask, context=ctx)
        token = _evaluate(self.token, state, mask_secrets=mask, context=ctx)

        # FIXME: is there a way to access FunctionAuth as non-private?
        return httpx._auth.FunctionAuth(
            SigV4Auth(
                access_key=access_key,
                secret_key=secret_key,
                token=token,
                service=service,
                region=region,
            )
        )


_boto3_credentials: dict[str, Any] = {}


class AwsBotoAuth(AsyncAction):
    """AWS Sigv4 Authentication (AWS4-HMAC-SHA256)

    With static access key and secret key:

        auth = AwsAuth(
            service='execute-api',
            region='us-east-1',
            access_key='AWS_ACCESS_KEY_ID',
            secret_key='AWS_SECRET_ACCESS_KEY',
        )

    With STS credentials:

        import boto3
        credentials = boto3.Session(profile_name="<profile>").get_credentials()
        auth = AwsAuth(
            service='execute-api',
            region='us-east-1',
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            token=credentials.token,
        )
    """

    def __init__(
        self,
        profile_name: str,
        state_access_key_var: str = 'aws_access_key',
        state_secret_key_var: str = 'aws_secret_key',
        state_token_var: str = 'aws_token',
    ) -> None:
        super().__init__()
        # FIXME: evalueate these:
        self._profile_name = profile_name
        self._state_access_key_var = state_access_key_var
        self._state_secret_key_var = state_secret_key_var
        self._state_token_var = state_token_var

    def _boto3_auth(self) -> tuple[str, str, str | None]:
        import boto3
        if self._profile_name in _boto3_credentials:
            credentials = _boto3_credentials[self._profile_name]
        else:
            session = boto3.Session(profile_name=self._profile_name)
            credentials = session.get_credentials()
            _boto3_credentials[self._profile_name] = credentials
        return credentials.access_key, credentials.secret_key, credentials.token

    async def call_async(
        self,
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        access_key, secret_key, token = await asyncio.to_thread(self._boto3_auth)
        state = State(state)
        state[self._state_access_key_var] = access_key
        state[self._state_secret_key_var] = Secret(secret_key)
        state[self._state_token_var] = Secret(token)
        return state


KUTSU_USER_AGENT = 'kutsu/0.1'


class HttpRequest(AsyncAction):
    """Base Class for HTTP Requests

    Not very usable on its own. Subclass SyncHttpRequest or AsyncHttpRequest instead.

    """

    # TODO: we need good argument descriptions
    # TODO: if some kind of body data given but no method, default to POST
    method: Node | str | None = 'GET'
    url: Node | str | None = ''
    http2: Node | bool = False
    # TODO: rename to query?
    params: Node | dict[str, str] | None = None
    headers: Node | dict[str, str] | dict[str, list[str]] | None = None
    user_agent: Node | str | None = KUTSU_USER_AGENT
    auth: Node | Auth | tuple[str, str] | None = None
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
    json: Node | JsonTemplate | None = None
    stream: Node | AsyncByteStream | None = None
    cookies: Node | Cookies | dict[str, str] | None = None

    # Raise error on non-ok response status code
    raise_error: bool = True

    # Automatically follow redirects
    follow_redirects: bool = True

    # Read cookies from this state variable and add them to the request
    read_cookies_from: str | None = '_cookies'

    # Save cookies from response to this state variable
    save_cookies_to: str | None = read_cookies_from

    # Use this as default state. Incoming state overrides these values.
    defaults: State | dict[str, Any] | None = None
    input_state: State | None = None
    output_state: State | None = None

    # Name of this request, used for printing
    name: str | None = None

    # Whether to print various things
    # TODO: optionally direct output to logger instead of printing
    # TODO: evaluate these?
    show_input_state: bool = False
    show_output_state: bool = False
    show_connection_info: bool = False
    show_ssl_info: bool = False
    show_redirects: bool = True
    show_request: bool = True
    show_request_headers: bool = False
    show_response: bool = True
    show_response_headers: bool = False
    # TODO: masked_request_headers
    # TODO: masked_response_headers
    show_response_body: bool | None = None
    show_headers: bool | None = None
    verbose: bool = False
    quiet: bool = False
    unmask: bool = False

    # Suppress text response bodies larger than this from printing
    show_max_body: int | None = 50 * 1024

    client: AsyncClient | None = None
    request: Request | None = None
    response: Response | None = None

    _prepared_config: RequestConfig | None = None
    _prepared_data: RequestData | None = None
    _prepared_data_masked: RequestData | None = None

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield 'method', self.method
        yield 'url', self.url

        if self.http2:
            yield 'http2', self.http2
        if self.params is not None:
            yield 'params', self.params
        if self.headers is not None:
            yield 'headers', self.headers
        yield 'user_agent', self.user_agent, KUTSU_USER_AGENT
        if self.auth is not None:
            yield 'auth', self.auth
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
        yield 'read_cookies_from', self.read_cookies_from, '_cookies'
        yield 'save_cookies_to', self.save_cookies_to, '_cookies'
        yield 'show_input_state', self.show_input_state, False
        yield 'show_output_state', self.show_output_state, False
        yield 'show_connection_info', self.show_connection_info, False
        yield 'show_ssl_info', self.show_ssl_info, False
        yield 'show_redirects', self.show_redirects, False
        yield 'show_request', self.show_request, True
        yield 'show_request_headers', self.show_request_headers, False
        yield 'show_response', self.show_response, True
        yield 'show_response_headers', self.show_response_headers, False
        yield 'show_response_body', self.show_response_body, None
        yield 'show_headers', self.show_headers, None
        yield 'show_max_body', self.show_max_body, 50 * 1024
        yield 'verbose', self.verbose, False
        yield 'quiet', self.quiet, False
        yield 'unmask', self.unmask, False
        yield 'request_prepared', self.request_prepared, False
        yield 'input_state', self.input_state, None
        yield 'request', self.request, None
        yield 'response_received', self.response_received, False
        yield 'response', self.response, None
        yield 'output_state', self.output_state, None

    def __init__(
        self,
        url: Node | str | None = None,
        http2: Node | bool | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        user_agent: Node | str | None = None,
        auth: Node | Auth | tuple[str, str] | None = None,
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
        show_connection_info: bool | None = None,
        show_ssl_info: bool | None = None,
        show_redirects: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
        unmask: bool | None = None,
    ) -> None:
        # TODO: docstring with argument descriptions
        super().__init__()
        if url is not None:
            self.url = url
        if http2 is not None:
            self.http2 = http2
        if params is not None:
            self.params = params
        if method is not None:
            self.method = method
        if headers is not None:
            self.headers = headers
        if user_agent is not None:
            self.user_agent = user_agent
        if auth is not None:
            self.auth = auth
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
        if show_connection_info is not None:
            self.show_connection_info = show_connection_info
        if show_ssl_info is not None:
            self.show_ssl_info = show_ssl_info
        if show_redirects is not None:
            self.show_redirects = show_redirects
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
        if unmask is not None:
            self.unmask = unmask

    @property
    def request_prepared(self) -> bool:
        return self.request is not None

    @property
    def response_received(self) -> bool:
        return self.response is not None

    def reset(self) -> None:
        self.client = None
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
            # if state[k] is not self.input_state[k]:
            if state[k] != self.input_state[k]:
                return True
        return False

    def _config_changed(self, config: RequestConfig) -> bool:
        if self._prepared_config is None:
            return True
        for i, v in enumerate(config):
            # if v is not self._prepared_config[i]:
            if v != self._prepared_config[i]:
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
            # request = self.request
            request = self.prepare(state, config=config, only_client=True)
        return request

    def prepare(
        self,
        state: State | dict[str, Any] | None = None,
        config: RequestConfig | None = None,
        only_client: bool = False,
    ) -> Request:
        """Prepare the request.

        This method is called before sending the request.
        You may also call it manually to prepare the request before sending it,
        and possibly mutate self.request before sending.
        """
        if only_client:
            # We create a new client for each request
            # Just copy the auth from the old client in this case
            assert self.client is not None
            assert self.request is not None
            self.client = httpx.AsyncClient(auth=self.client.auth, http2=config.http2)
            return self.request
        if config is None:
            config = self._make_request_config()
        self._prepared_config = config
        self.input_state = State(state)
        s = self.input_state(State(config.defaults))
        auth = self._prepare_auth(s, config)
        data = RequestData(
            method=self._prepare_method(s, config),
            url=self._prepare_url(s, config),
            params=self._prepare_params(s, config),
            headers=self._prepare_headers(s, config, with_authorization=auth is None),
            content=self._prepare_content(s, config),
            data=self._prepare_data(s, config),
            files=self._prepare_files(s, config),
            json=self._prepare_json(s, config),
            stream=self._prepare_stream(s, config),
            cookies=self._prepare_cookies(s, config),
        )
        # TODO: use masked data for printing
        # TODO: add a masked full url which contains masked query params
        masked_auth = self._prepare_auth(s, config, mask_secrets=True)
        self._prepared_data_masked = RequestData(
            method=self._prepare_method(s, config, mask_secrets=True),
            url=self._prepare_url(s, config, mask_secrets=True),
            params=self._prepare_params(s, config, mask_secrets=True),
            headers=self._prepare_headers(
                s, config, mask_secrets=True, with_authorization=masked_auth is None
            ),
            content=self._prepare_content(s, config, mask_secrets=True),
            data=self._prepare_data(s, config, mask_secrets=True),
            files=self._prepare_files(s, config, mask_secrets=True),
            json=self._prepare_json(s, config, mask_secrets=True),
            stream=self._prepare_stream(s, config, mask_secrets=True),
            cookies=self._prepare_cookies(s, config, mask_secrets=True),
        )
        self._prepared_data = data
        # self.request = Request(**data._asdict())
        self.request = Request(
            extensions={'trace': self._request_tracer(config=config)}, **data._asdict()
        )
        self.client = httpx.AsyncClient(auth=auth, http2=config.http2)
        return self.request

    # TODO: combine existing print methods with these and have flags allowing to
    # TODO: print authorization headers without masking secrets
    def _format_request_headers(
        self,
        config: RequestConfig,
        request: httpcore.Request,
        http2: bool = False,
    ) -> str:
        version = 'HTTP/2' if http2 else 'HTTP/1.1'
        headers = [
            (name.lower() if http2 else name, value) for name, value in request.headers
        ]
        method = request.method.decode('ascii')
        # TODO: mask secrets in target
        # target = request.url.target.decode('ascii')
        target = bytes(request.url).decode('ascii')
        lines = [f'{method} {target} {version}']
        if config.show_request_headers:
            lines += [
                f"{name.decode('ascii')}: {value.decode('ascii')}"
                for name, value in headers
            ]
        return '\n'.join(lines)

    def _format_response_headers(
        self,
        config: RequestConfig,
        http_version: bytes,
        status: int,
        reason_phrase: bytes | None,
        headers: list[tuple[bytes, bytes]],
    ) -> str:
        version = http_version.decode('ascii')
        reason = (
            httpx.codes.get_reason_phrase(status)
            if reason_phrase is None else reason_phrase.decode('ascii')
        )
        lines = [f'{version} {status} {reason}']
        if config.show_response_headers:
            lines += [
                f"{name.decode('ascii')}: {value.decode('ascii')}"
                for name, value in headers
            ]
        else:
            is_redirect = status in {301, 302, 303, 307, 308}
            if is_redirect:
                lines += [
                    f"{name.decode('ascii')}: {value.decode('ascii')}"
                    for name, value in headers
                    if name.decode('ascii').lower() == 'location'
                ]

        return '\n'.join(lines)

    def _show_request(
        self,
        request: httpcore.Request,
        config: RequestConfig,
        http2: bool = False,
    ) -> None:
        show_request = config.show_request
        show_request_headers = config.show_request_headers
        console = rich.console.Console(soft_wrap=True)
        if not show_request and not show_request_headers:
            return
        if getattr(self.request, '_kutsu_redirected', False):
            if not config.show_redirects:
                return
        http_text = self._format_request_headers(config, request, http2=http2)
        syntax = rich.syntax.Syntax(http_text, 'http', theme='ansi_dark', word_wrap=False)
        console.print(syntax)
        if show_request_headers:
            # syntax = rich.syntax.Syntax('', 'http', theme='ansi_dark', word_wrap=True)
            # console.print(syntax)
            # console.rule(characters='–', style='dim')
            pass
        if show_request and self.request is not None:
            if self.request.content:
                # console.print('─' * 4, style='dim')
                console.print('─' * 60, style='dim')
                # console.print('-' * 4, style='dim')
                # console.rule(characters='–', style='dim')
                # console.rule(characters='-', style='dim')
                # console.print('-' * 78, style='dim')
                # console.rule(characters='–', style='dim')
                # console.print('––––', style='dim')
                console.print(self.request.content.decode('utf-8'))
                # console.print('––––', style='dim')
                # console.rule(characters='-')
                # console.rule(characters='-', style='dim')
                # console.print('-' * 78, style='dim')
                # console.rule(characters='–', style='dim')
                # console.print('-' * 4, style='dim')
                # console.print('─' * 4, style='dim')
                console.print('─' * 60, style='dim')
            elif self.request.method in ('POST', 'PUT', 'PATCH'):
                console.print('[italic]Empty request body[/italic]')
                # console.print()

    def _show_response(
        self,
        config: RequestConfig,
        http_version: bytes,
        status: int,
        reason_phrase: typing.Optional[bytes],
        headers: typing.List[typing.Tuple[bytes, bytes]],
    ) -> None:
        show_response = config.show_response
        show_response_headers = config.show_response_headers
        show_request_headers = config.show_request_headers
        if not show_response and not show_response_headers:
            return
        console = rich.console.Console(soft_wrap=True)
        no_color_console = get_console(no_color=True, soft_wrap=True)
        is_redirect = status in {301, 302, 303, 307, 308}
        if is_redirect:
            self.request._kutsu_redirected = True
            if not config.show_redirects:
                return
        if not is_redirect:  # and show_request_headers:
            name = self._make_request_name(config)
            no_color_console.print(f'[bold]*** Response [{name}][/bold]')
        http_text = self._format_response_headers(
            config, http_version, status, reason_phrase, headers
        )
        syntax = rich.syntax.Syntax(http_text, 'http', theme='ansi_dark', word_wrap=False)
        console.print(syntax)
        # TODO: show redirect body if non-empty
        if is_redirect and config.follow_redirects:
            # console.print('↓', style='dim')
            console.print('→ redirecting:', style='dim')
            # console.print('')
        # syntax = rich.syntax.Syntax('', 'http', theme='ansi_dark', word_wrap=True)
        # console.print(syntax)

    # def _get_lexer_for_response(self, response: Response) -> str:
    #     content_type = response.headers.get('Content-Type')
    #     if content_type is not None:
    #         mime_type, _, _ = content_type.partition(';')
    #         try:
    #             return typing.cast(
    #                 str,
    #                 pygments.lexers.get_lexer_for_mimetype(mime_type.strip()
    #                                                        ).name  # type: ignore
    #             )
    #         except pygments.util.ClassNotFound:  # pragma: no cover
    #             pass
    #     return ''

    # def _show_response_body(self, response: Response) -> None:
    #     console = rich.console.Console(soft_wrap=True)
    #     lexer_name = self._get_lexer_for_response(response)
    #     if lexer_name:
    #         if lexer_name.lower() == 'json':
    #             try:
    #                 data = response.json()
    #                 text = jsonlib.dumps(data, indent=4)
    #             except ValueError:  # pragma: no cover
    #                 text = response.text
    #         else:
    #             text = response.text

    #         syntax = rich.syntax.Syntax(
    #             text, lexer_name, theme='ansi_dark', word_wrap=False
    #         )
    #         console.print(syntax)
    #     else:
    #         console.print(f'<{len(response.content)} bytes of binary data>')

    _PCTRTT = typing.Tuple[typing.Tuple[str, str], ...]
    _PCTRTTT = typing.Tuple[_PCTRTT, ...]
    _PeerCertRetDictType = typing.Dict[str, typing.Union[str, _PCTRTTT, _PCTRTT]]

    def _format_certificate(self, cert: _PeerCertRetDictType) -> str:  # pragma: no cover
        lines = []
        for key, value in cert.items():
            if isinstance(value, (list, tuple)):
                lines.append(f'*   {key}:')
                for item in value:
                    if key in ('subject', 'issuer'):
                        for sub_item in item:
                            lines.append(f'*     {sub_item[0]}: {sub_item[1]!r}')
                    elif isinstance(item, tuple) and len(item) == 2:
                        lines.append(f'*     {item[0]}: {item[1]!r}')
                    else:
                        lines.append(f'*     {item!r}')
            else:
                lines.append(f'*   {key}: {value!r}')
        return '\n'.join(lines)

    def _request_tracer(
        self,
        config: RequestConfig,
    ) -> Callable[[str, typing.Mapping[str, typing.Any]], typing.Awaitable[None]]:

        async def async_trace(name: str, info: typing.Mapping[str, typing.Any]) -> None:
            console = rich.console.Console()
            if name == 'connection.connect_tcp.started' and config.show_connection_info:
                host = info['host']
                console.print(f'* Connecting to {host!r}')
            elif name == 'connection.connect_tcp.complete' and config.show_connection_info:
                stream = info['return_value']
                server_addr = stream.get_extra_info('server_addr')
                console.print(
                    f'* Connected to {server_addr[0]!r} on port {server_addr[1]}'
                )
            elif name == 'connection.start_tls.complete' and config.show_ssl_info:
                stream = info['return_value']
                ssl_object = stream.get_extra_info('ssl_object')
                version = ssl_object.version()
                cipher = ssl_object.cipher()
                server_cert = ssl_object.getpeercert()
                alpn = ssl_object.selected_alpn_protocol()
                console.print(f'* SSL established using {version!r} / {cipher[0]!r}')
                console.print(f'* Selected ALPN protocol: {alpn!r}')
                if server_cert:
                    console.print('* Server certificate:')
                    console.print(self._format_certificate(server_cert))
            elif name == 'http11.send_request_headers.started':
                request = info['request']
                self._show_request(request, config, http2=False)
            elif name == 'http2.send_request_headers.started':
                request = info['request']
                self._show_request(request, config, http2=True)
            elif name == 'http11.receive_response_headers.complete':
                http_version, status, reason_phrase, headers = info['return_value']
                self._show_response(config, http_version, status, reason_phrase, headers)
            elif name == 'http2.receive_response_headers.complete':
                status, headers = info['return_value']
                http_version = b'HTTP/2'
                reason_phrase = None
                self._show_response(config, http_version, status, reason_phrase, headers)

        return async_trace

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

    def _prepare_auth(
        self,
        state: State,
        config: RequestConfig,
        mask_secrets: bool = False
    ) -> Auth | tuple[str, str] | None:
        auth = evaluate(config.auth, state, mask_secrets=mask_secrets)
        if auth is not None:
            if not isinstance(auth, (Auth, tuple)):
                raise TypeError(
                    f'auth must be httpx.Auth or tuple[str, str], not {type(auth)}'
                )
            return auth

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
        if token is None and None not in {username, password}:
            if scheme is None:
                scheme = 'Basic'
            if scheme.lower() == 'basic':
                return (username, password)
            if scheme.lower() == 'digest':
                return httpx.DigestAuth(username, password)

        def inject_authorization_header(authorization: str, request: Request) -> Request:
            request.headers['Authorization'] = authorization
            return request

        if token is not None:
            if scheme is None or scheme.lower() == 'Bearer':
                # FIXME: is there a way to access FunctionAuth as non-private?
                return httpx._auth.FunctionAuth(
                    functools.partial(inject_authorization_header, f'Bearer {token}')
                )

        return None

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
                raise TypeError(f'authorization must be str, not {type(authorization)}')
            return authorization

        # FIXME: we should move these to _prepare_auth and only provide custom
        # FIXME: authorization header handling here if auth is None
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
            # FIXME: basic auth is as easy as: auth=('user', 'pass')
            # digest: auth = httpx.DigestAuth('user', 'pass')
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
        mask_secrets: bool = False,
        with_authorization: bool = False,
    ) -> dict[str, str] | None:
        headers = evaluate(config.headers or {}, state, mask_secrets=mask_secrets)
        if not isinstance(headers, dict):
            raise TypeError(f'headers must be a dict, not {type(headers)}')
        content_type = evaluate(
            config.content_type, state, mask_secrets=mask_secrets, as_str=True
        )
        if content_type is not None:
            headers['Content-Type'] = content_type
        user_agent = evaluate(
            config.user_agent, state, mask_secrets=mask_secrets, as_str=True
        )
        if user_agent is not None:
            headers['User-Agent'] = user_agent
        if with_authorization:
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
        http2: Node | bool | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        user_agent: Node | str | None = None,
        auth: Node | Auth | tuple[str, str] | None = None,
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
        show_connection_info: bool | None = None,
        show_ssl_info: bool | None = None,
        show_redirects: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
        unmask: bool | None = None,
    ) -> RequestConfig:

        def choose(first: T, second: U) -> T | U:
            return first if first is not None else second

        _show_input_state = None
        _show_output_state = None
        _show_connection_info = None
        _show_ssl_info = None
        _show_redirects = None
        _show_request = None
        _show_request_headers = None
        _show_response = None
        _show_response_headers = None
        _show_response_body = None
        _show_max_body: int | None = None

        # First evaluate instance defaults
        if self.verbose is True:
            _show_input_state = True
            _show_output_state = True
            _show_connection_info = True
            _show_ssl_info = True
            _show_redirects = True
            _show_request = True
            _show_request_headers = True
            _show_response = True
            _show_response_headers = True
        if self.show_headers is not None:
            _show_request_headers = self.show_headers
            _show_response_headers = self.show_headers
        # if self.show_response_body is True:
        #     _show_max_body = None
        if self.quiet is True:
            _show_input_state = False
            _show_output_state = False
            _show_connection_info = False
            _show_ssl_info = False
            _show_redirects = False
            _show_request = False
            _show_request_headers = False
            _show_response = False
            _show_response_headers = False

        # Check if any default have been overridden
        # TODO: add constants for defaults and use them here and in the class definition
        # TODO: we probably need sentinel values
        # if self.show_input_state is True:
        #     _show_input_state = True
        # if self.show_output_state is True:
        #     _show_output_state = True
        # if self.show_connection_info is True:
        #     _show_connection_info = True
        # if self.show_ssl_info is True:
        #     _show_ssl_info = True
        # if self.show_request is False:
        #     _show_request = False
        # if self.show_request_headers is True:
        #     _show_request_headers = True
        # if self.show_response is False:
        #     _show_response = False
        # if self.show_response_headers is True:
        #     _show_response_headers = True
        # if self.show_response_body is not None:
        #     _show_response_body = self.show_response_body
        # if self.show_max_body is not None:
        #     _show_max_body = self.show_max_body

        # Then evaluate function args
        if verbose is True:
            _show_input_state = True
            _show_output_state = True
            _show_connection_info = True
            _show_ssl_info = True
            _show_redirects = True
            _show_request = True
            _show_request_headers = True
            _show_response = True
            _show_response_headers = True
        if show_headers is not None:
            _show_request_headers = show_headers
            _show_response_headers = show_headers
        # if show_response_body is True:
        #     _show_max_body = None
        if quiet is True:
            _show_input_state = False
            _show_output_state = False
            _show_connection_info = False
            _show_ssl_info = False
            _show_redirects = False
            _show_request = False
            _show_request_headers = False
            _show_response = False
            _show_response_headers = False

        if show_input_state is not None:
            _show_input_state = show_input_state
        if show_output_state is not None:
            _show_output_state = show_output_state
        if show_connection_info is not None:
            _show_connection_info = show_connection_info
        if show_ssl_info is not None:
            _show_ssl_info = show_ssl_info
        if show_redirects is not None:
            _show_redirects = show_redirects
        if show_request is not None:
            _show_request = show_request
        if show_request_headers is not None:
            _show_request_headers = show_request_headers
        if show_response is not None:
            _show_response = show_response
        if show_response_headers is not None:
            _show_response_headers = show_response_headers
        if show_response_body is not None:
            _show_response_body = show_response_body
        if _show_response_body is True:
            _show_max_body = None
        if show_max_body is not None:
            _show_max_body = show_max_body

        return RequestConfig(
            url=choose(url, self.url),
            http2=choose(http2, self.http2),
            method=choose(method, self.method),
            params=choose(params, self.params),
            headers=choose(headers, self.headers),
            user_agent=choose(user_agent, self.user_agent),
            auth=choose(auth, self.auth),
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
            # TODO: we should probably not use choose for these but logic commented out above
            show_input_state=choose(_show_input_state, self.show_input_state),
            show_output_state=choose(_show_output_state, self.show_output_state),
            show_connection_info=choose(_show_connection_info, self.show_connection_info),
            show_ssl_info=choose(_show_ssl_info, self.show_ssl_info),
            show_redirects=choose(_show_redirects, self.show_redirects),
            show_request=choose(_show_request, self.show_request),
            show_request_headers=choose(_show_request_headers, self.show_request_headers),
            show_response=choose(_show_response, self.show_response),
            show_response_headers=choose(
                _show_response_headers, self.show_response_headers
            ),
            show_response_body=choose(_show_response_body, self.show_response_body),
            show_headers=choose(show_headers, self.show_headers),
            show_max_body=choose(_show_max_body, self.show_max_body),
            verbose=choose(verbose, self.verbose),
            quiet=choose(quiet, self.quiet),
            unmask=choose(unmask, self.unmask),
        )

    # This is here just for REPL reference and tab completion of argument names
    def __call__(
        self,
        state: StateArg | None,
        /,
        url: Node | str | None = None,
        http2: Node | bool | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        user_agent: Node | str | None = None,
        auth: Node | Auth | tuple[str, str] | None = None,
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
        show_connection_info: bool | None = None,
        show_ssl_info: bool | None = None,
        show_redirects: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
        unmask: bool | None = None,
        **kwargs: Any,
    ) -> State:
        return super().__call__(
            state,
            url=url,
            http2=http2,
            method=method,
            params=params,
            headers=headers,
            user_agent=user_agent,
            auth=auth,
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
            show_connection_info=show_connection_info,
            show_ssl_info=show_ssl_info,
            show_redirects=show_redirects,
            show_request=show_request,
            show_request_headers=show_request_headers,
            show_response=show_response,
            show_response_headers=show_response_headers,
            show_response_body=show_response_body,
            show_headers=show_headers,
            show_max_body=show_max_body,
            verbose=verbose,
            quiet=quiet,
            unmask=unmask,
            **kwargs,
        )

    async def call_async(
        self,
        state: StateArg | None,
        /,
        url: Node | str | None = None,
        http2: Node | bool | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        user_agent: Node | str | None = None,
        auth: Node | Auth | tuple[str, str] | None = None,
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
        show_connection_info: bool | None = None,
        show_ssl_info: bool | None = None,
        show_redirects: bool | None = None,
        show_request: bool | None = None,
        show_request_headers: bool | None = None,
        show_response: bool | None = None,
        show_response_headers: bool | None = None,
        show_response_body: bool | None = None,
        show_headers: bool | None = None,
        show_max_body: int | None = None,
        verbose: bool | None = None,
        quiet: bool | None = None,
        unmask: bool | None = None,
        **kwargs: Any,
    ) -> State:
        state = await super().call_async(State(state), **kwargs)
        config = self._make_request_config(
            url=url,
            http2=http2,
            method=method,
            params=params,
            headers=headers,
            user_agent=user_agent,
            auth=auth,
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
            show_connection_info=show_connection_info,
            show_redirects=show_redirects,
            show_request=show_request,
            show_request_headers=show_request_headers,
            show_response=show_response,
            show_response_headers=show_response_headers,
            show_response_body=show_response_body,
            show_headers=show_headers,
            show_max_body=show_max_body,
            verbose=verbose,
            quiet=quiet,
            unmask=unmask,
        )

        request = self._prepare_call(state, config)
        self._print_input_state(config)
        # TODO: also print prepared config
        self._print_request(config)
        assert self.client is not None
        async with self.client as client:
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
            if config.follow_redirects or res.status_code >= 400:
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
        state = self.on_response(state, res)
        # TODO: json hook
        return state

    def on_response(self, state: State, res: Response) -> State:
        return state

    def _make_request_name(self, config: RequestConfig) -> str:
        if config.name:
            return config.name
        return f'{self.__class__.__name__}<{make_object_id(self)}>'

    def _print_request(self, config: RequestConfig) -> None:
        assert self._prepared_data_masked is not None
        show_request = config.show_request
        show_request_headers = config.show_request_headers
        if not show_request and not show_request_headers:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True, soft_wrap=True)
        syntax_console = get_console(soft_wrap=True)
        if self.request is None:
            console.print(f'[italic]{name} request not prepared[/italic]')
            return
        req = self.request
        now = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
        # console.rule(f'{name} Fetch {now}')
        console.print(f'[bold]*** Fetch {now} [{name}][/bold]')
        return
        method = self._prepared_data_masked.method
        url = self._prepared_data_masked.url
        # status = f'{req.method} {req.url} HTTP/1.1'
        status = f'{method} {url} HTTP/1.1'

        h = {} | (self._prepared_data_masked.headers or {})
        for k, v in req.headers.raw:
            name = k.decode('utf-8')
            if name not in h:
                h[name] = v.decode('utf-8')  # TODO: is it always utf-8?

        headers = [
            # f'{name.decode("ascii")}: {value.decode("ascii")}'
            f'{name}: {value}'
            # TODO: print any header from req.headers.raw which is not in masked headers
            # for name, value in req.headers.raw
            # for name, value in (self._prepared_data_masked.headers or {}).items()
            for name, value in h.items()
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
        # TODO: mask secrets in request content
        # TODO: pretty print json
        content = req.content.decode('utf-8')
        if content:
            # console.rule(characters='–')
            syntax_console.print(content)
            console.print()
        elif req.method in ('POST', 'PUT', 'PATCH'):
            console.print('[italic]Empty request body[/italic]')
        # if show_request_headers or content:
        #     console.print()

    def _print_response(self, config: RequestConfig) -> None:
        show_response = config.show_response
        show_response_headers = config.show_response_headers
        show_response_body = config.show_response_body
        # show_request_headers = config.show_request_headers
        show_max_body = config.show_max_body
        if not show_response and not show_response_headers:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True, soft_wrap=True)
        syntax_console = get_console(soft_wrap=True)
        # if self.response is None:
        #     console.print(f'[italic]{name} no response received[/italic]')
        #     return
        assert self.response is not None
        res = self.response

        # if show_request_headers:
        #     console.print(f'[bold]*** Response [{name}][/bold]')
        downloaded = bytes_to_readable(res.num_bytes_downloaded)
        elapsed = f'{int(res.elapsed.total_seconds()*1000)} ms'
        # # console.rule(f'{name} Response {downloaded} in {elapsed}')
        # status = f'{res.http_version} {res.status_code} {res.reason_phrase or ""}'
        # headers = [
        #     f'{name.decode("ascii")}: {value.decode("ascii")}'
        #     for name, value in res.headers.raw
        # ]
        # hdrs = '\n'.join(headers)
        # if show_response_headers:
        #     s = f'{status}\n{hdrs}\n'
        # else:
        #     s = status
        # syntax_console.print(
        #     rich.syntax.Syntax(f'{s}', 'http', theme='ansi_dark', word_wrap=True)
        # )

        if show_response_body is not False:
            # console.rule(characters='–')
            # console.rule(characters='–', style='dim')
            # console.rule(characters='-', style='dim')
            # console.print('-' * 78, style='dim')
            # console.print('-' * 4, style='dim')
            # console.print('─' * 4, style='dim')
            console.print('─' * 60, style='dim')

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
                        text, lexer_name, theme='ansi_dark', word_wrap=False
                    )
                    syntax_console.print(syntax)
            else:
                console.print(f'<{len(res.content)} bytes of binary data>')
            # console.rule(characters='–', style='dim')
            # console.rule(characters='-', style='dim')
            # console.print('-' * 78, style='dim')
            # console.print('-' * 4, style='dim')
            # console.print('─' * 4, style='dim')
            console.print('─' * 60, style='dim')
        console.print(f'[bold]*** Received {downloaded} in {elapsed} [{name}][/bold]')

    def _print_input_state(self, config: RequestConfig) -> None:
        show_input_state = config.show_input_state
        if not show_input_state:
            return
        name = self._make_request_name(config)
        console = get_console(no_color=True, soft_wrap=True)
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
        console = get_console(no_color=True, soft_wrap=True)
        # console.rule(f'{name} Output State')
        console.print(f'[bold]*** Output State [{name}][/bold]')
        if self.output_state is None:
            console.print('[italic]No output state[/italic]')
            return
        self._print_state(self.output_state, config)

    def _print_state(
        self,
        state: State,
        config: RequestConfig,
    ) -> None:
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
                    console.print('\n[italic]Cookies:[/italic]')
                    for cookie in cookies.jar:
                        console.print(
                            # f'- {repr(cookie)}'
                            f'{cookie.name}="{cookie.value}" for {cookie.domain} {cookie.path}',
                            # no_wrap=True,
                        )
                else:
                    console.print('[italic]Empty cookie Jar[/italic]')
            # console.print(state)
        console.print('')

    def _print_end_request_processing(self, config: RequestConfig) -> None:
        show_output_state = config.show_output_state
        if show_output_state:
            console = get_console(no_color=True, soft_wrap=True)
            # console.rule()
            name = self._make_request_name(config)
            console.print(f'[bold]*** End Request Processing {name}[/bold]')


class RequestConfig(NamedTuple):
    url: Node | str | None = HttpRequest.url
    http2: Node | bool | None = HttpRequest.http2
    method: Node | str | None = HttpRequest.method
    params: Node | dict[str, str] | None = HttpRequest.params
    headers: Node | dict[str, str] | dict[str, list[str]] | None = HttpRequest.headers
    user_agent: Node | str | None = HttpRequest.user_agent
    auth: Node | Auth | tuple[str, str] | None = HttpRequest.auth
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
    show_connection_info: bool | None = HttpRequest.show_connection_info
    show_ssl_info: bool | None = HttpRequest.show_ssl_info
    show_redirects: bool | None = HttpRequest.show_redirects
    show_request: bool | None = HttpRequest.show_request
    show_request_headers: bool | None = HttpRequest.show_request_headers
    show_response: bool | None = HttpRequest.show_response
    show_response_headers: bool | None = HttpRequest.show_request_headers
    show_response_body: bool | None = HttpRequest.show_response_body
    show_headers: bool | None = HttpRequest.show_headers
    show_max_body: int | None = HttpRequest.show_max_body
    verbose: bool | None = HttpRequest.verbose
    quiet: bool | None = HttpRequest.quiet
    unmask: bool | None = HttpRequest.unmask


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

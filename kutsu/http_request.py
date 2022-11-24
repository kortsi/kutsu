"""HTTP Requests"""
from __future__ import annotations

import base64
from typing import IO, Any, Generator

from httpx import AsyncByteStream, AsyncClient, Client, Cookies, Request, Response

from .expressions import Node, subst_data
from .state import Action, AsyncAction, State

Json = dict[str, 'Json'] | list['Json'] | str | int | float | bool | None


class HttpRequest:
    """Base Class for HTTP Requests

    Not very usable on its own. Subclass SyncHttpRequest or AsyncHttpRequest instead.

    """

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
    follow_redirects: bool = False

    # Read cookies from this state variable and add them to the request
    read_cookies_from: str | None = 'cookies'

    # Save cookies from response to this state variable
    save_cookies_to: str | None = 'cookies'

    # Use this as default state. Incoming state overrides these values.
    defaults: State | dict[str, Any] | None = None
    input_state: State | None = None
    output_state: State | None = None

    request: Request | None = None
    response: Response | None = None

    # def __repr__(self) -> str:
    #     TODO: implement
    #     return f'{self.__class__.__name__}()'

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
        yield 'follow_redirects', self.follow_redirects, False
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
    ) -> None:
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
        return self.request

    def _prepare_method(self, state: State) -> str:
        return str(subst_data(self.method, state))

    def _prepare_url(self, state: State) -> str:
        return str(subst_data(self.url, state))

    def _prepare_params(self, state: State) -> dict[str, str] | None:
        params = subst_data(self.params, state)
        if params is None:
            return None
        if not isinstance(params, dict):
            raise TypeError(f'params must be a dict, not {type(params)}')
        return params

    def _prepare_authorization_header(self, state: State) -> str | None:
        authorization = subst_data(self.authorization, state)
        if authorization is not None:
            if not isinstance(authorization, str):
                raise TypeError(f'authorization must be a str, not {type(authorization)}')
            return authorization

        scheme = subst_data(self.auth_scheme, state)
        token = subst_data(self.auth_token, state)
        username = subst_data(self.auth_username, state)
        password = subst_data(self.auth_password, state)

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
        headers = subst_data(self.headers or {}, state)
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
        content = subst_data(self.content, state)
        if content is None:
            return None
        if isinstance(content, (str, bytes)):
            return content
        raise TypeError(f'content must be a str or bytes, not {type(content)}')

    def _prepare_data(self, state: State) -> dict[str, str] | None:
        data = subst_data(self.data, state)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise TypeError(f'data must be a dict, not {type(data)}')
        return data

    def _prepare_files(self, state: State) -> dict[str, bytes | IO[bytes]] | None:
        files = subst_data(self.files, state)
        if files is None:
            return None
        if not isinstance(files, dict):
            raise TypeError(f'files must be a dict, not {type(files)}')
        return files

    def _prepare_json(self, state: State) -> Json | None:
        json = subst_data(self.json, state)
        if json is None:
            return None
        if isinstance(json, (dict, list, str, int, float, bool, type(None))):
            return json
        raise TypeError(
            f'json must be a dict, list, str, int, float, bool, or None, not {type(json)}'
        )

    def _prepare_stream(self, state: State) -> AsyncByteStream | None:
        stream = subst_data(self.stream, state)
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
        self_cookies = subst_data(self.cookies, state)
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

    def _process_response(self, state: State, response: Response) -> State:
        self.response = response
        if self.raise_error:
            response.raise_for_status()
        if self.save_cookies_to:
            state[self.save_cookies_to] = response.cookies
        self.output_state = State(state)
        return state


class SyncHttpRequest(HttpRequest, Action):
    """Synchronous HTTP Request"""

    def __init__(
        self,
        url: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        method: Node | str | None = None,
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
    ) -> None:
        Action.__init__(self)
        HttpRequest.__init__(
            self,
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
        )

    def __call__(
        self,
        state: State | dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> State:
        state = super().__call__(State(state))
        request = self._prepare_call(state, **kwargs)

        with Client() as client:
            response = client.send(request, follow_redirects=self.follow_redirects)

        return self._process_response(state, response)


class AsyncHttpRequest(HttpRequest, AsyncAction):
    """Asynchronous HTTP Request"""

    def __init__(
        self,
        url: Node | str | None = None,
        method: Node | str | None = None,
        params: Node | dict[str, str] | None = None,
        headers: Node | dict[str, str] | dict[str, list[str]] | None = None,
        auth_scheme: Node | str | None = None,
        auth_token: Node | str | None = None,
        auth_username: Node | str | None = None,
        auth_password: Node | str | None = None,
        authorization: Node | str | None = None,
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
    ) -> None:
        AsyncAction.__init__(self)
        HttpRequest.__init__(
            self,
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
        )

    async def __call__(
        self,
        state: State | dict[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> State:
        state = await super().__call__(State(state))
        request = self._prepare_call(state, **kwargs)

        async with AsyncClient() as client:
            response = await client.send(request, follow_redirects=self.follow_redirects)

        return self._process_response(state, response)

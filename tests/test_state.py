# type: ignore
import asyncio
from time import time
from types import FunctionType as function

import pytest

from kutsu.state import Action, AsyncAction, AsyncChain, Chain, Parallel, State


@pytest.mark.parametrize(
    'a',
    [Action, Action(), lambda s: s],
    ids=['class', 'instance', 'function'],
)
@pytest.mark.parametrize(
    'b',
    [Action, Action(), lambda s: s],
    ids=['class', 'instance', 'function'],
)
def test_make_chain(a, b):
    if type(a) == function and type(b) == function:
        pytest.xfail('plain functions cannot be chained')

    chain = a >> b
    assert isinstance(chain, Chain)


async def async_state_transformer(state):
    return state


@pytest.mark.parametrize(
    'a',
    [AsyncAction,
     AsyncAction(), Action,
     Action(), lambda s: s, async_state_transformer],
    ids=['async_class', 'async_instance', 'sync_class', 'sync_instance', 'function', 'coroutine'],
)
@pytest.mark.parametrize(
    'b',
    [AsyncAction, AsyncAction()],
    ids=['class', 'instance'],
)
def test_make_async_chain(a, b):

    chain1 = a >> b
    assert isinstance(chain1, AsyncChain)
    chain2 = b >> a
    assert isinstance(chain2, AsyncChain)


def test_parallel():
    a = AsyncAction()
    b = AsyncAction()

    par = a // b
    assert isinstance(par, Parallel)


def test_many_parallel():
    a = AsyncAction()

    par = a**10
    assert isinstance(par, Parallel)


def test_execute_1000_parallel():

    n = 1000
    SLEEP = 0.01

    class Sleeper(AsyncAction):

        async def __call__(self, state=None):
            state = await super().__call__(state)
            await asyncio.sleep(SLEEP)
            state.results[state.INSTANCE_NUMBER] = True
            return state

    start_time = time()
    s = State(results=[False] * n) | Sleeper()**n
    elapsed_time = time() - start_time

    # If elapsed was less than 1% of the cumulatime time, then it's parallel enough
    assert elapsed_time < SLEEP * n * 0.01

    for i in range(n):
        assert s.results[i] is True


def test_add_state():
    s = State(a=1)
    t = State(b=2)
    u = s + t
    assert u.a == 1
    assert u.b == 2


def test_execute_action():

    class MyTestAction(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = 'a'
            return state

    a = MyTestAction()
    s = a()
    assert s.result == 'a'


def test_pipe_state_into_action():

    class MyTestAction(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = state.input + 1
            return state

    a = MyTestAction()
    s = State(input=1) | a
    assert s.result == 2


def test_pipe_dict_into_action():

    class MyTestAction(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = state.input + 1
            return state

    a = MyTestAction()
    s = {'input': 1} | a
    assert s.result == 2


def test_execute_function():

    def my_test_func(state):
        state.result = 'b'
        return state

    s = State | my_test_func
    assert s.result == 'b'


def test_execute_async_action():

    class AsyncTestAction(AsyncAction):

        async def __call__(self, state=None):
            state = await super().__call__(state)
            state.async_result = 'x'
            return state

    a = AsyncTestAction()
    s = State | a
    assert s.async_result == 'x'


def test_execute_async_function():

    async def async_test_func(state):
        state.async_result = 'y'
        return state

    s = State | async_test_func
    assert s.async_result == 'y'


def test_nested_sync_async():

    class AsyncTestAction(AsyncAction):

        async def __call__(self, state=None):
            state = await super().__call__(state)
            state.async_result = 'z'
            return state

    def sync_test_func(state):
        a = AsyncTestAction()
        s = State | a
        return s

    async def async_test_func(state):
        s = State | sync_test_func
        return s

    s = State | async_test_func
    assert s.async_result == 'z'


def test_execute_parallel():

    class AsyncTestAction(AsyncAction):

        async def __call__(self, state=None, /, **kwargs):
            state = await super().__call__(state, **kwargs)
            state.answer = state.INSTANCE_NUMBER
            return state

    def merge(state, results):
        for result in results:
            state.results[result.INSTANCE_NUMBER] = result.answer
        return state

    async def run_test():
        par = AsyncTestAction**10
        state = State(results=[None] * 10)
        state = await par(state, merge)
        return state

    state = asyncio.run(run_test())
    assert len(state.results) == 10
    assert state.results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# TODO: test execute Chain
# TODO: test execute AsyncChain

# type: ignore
import asyncio
from time import time

import pytest

from kutsu.state import (
    Action,
    AsyncAction,
    Chain,
    Default,
    Eval,
    Identity,
    Override,
    Parallel,
    Slice,
    State,
)


def test_make_state():

    class MyState(State):
        a = 1  # noqa

    s = MyState(b=2)
    s.c = 3
    s['d'] = 4

    assert s.a == s['a'] == 1
    assert s.b == s['b'] == 2
    assert s.c == s['c'] == 3
    assert s.d == s['d'] == 4


def test_compare_states():
    s1 = State(a=1)
    s2 = State(a=1)
    s3 = State(a=2)
    s4 = State(a=1, b=2)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4

    del s4.b

    assert s1 == s4


def test_add_state():
    s = State(a=1)
    t = State(b=2)
    u = s + t
    assert u.a == 1
    assert u.b == 2


def test_action():
    a = Action()
    b = Action(lambda state: State(state, value=1))

    s1 = State(value=2)
    s2 = a(s1)
    assert s2 == s1

    s3 = State(value=2)
    s4 = b(s3)
    assert s4 != s3
    assert s4.value == 1


def test_async_action():

    async def async_transformer(state):
        return State(state, value=1)

    a = AsyncAction()
    b = AsyncAction(async_transformer)

    s1 = State(value=2)
    s2 = s1 | a
    assert s2.value == 2

    s3 = State(value=2)
    s4 = s3 | b
    assert (s4.value == 1)


def test_invalid_pipe():

    a = Action()

    with pytest.raises(TypeError):
        () | a

    with pytest.raises(TypeError):
        '' | a

    with pytest.raises(TypeError):
        set() | a


def test_identity():
    s = State(x=1)
    t = s | Identity
    assert t.x == 1
    assert t == s


def test_default():
    default = Default(a=1)

    s = State() | default
    assert s.a == 1

    s = State(a=2) | default
    assert s.a == 2


def test_override():
    override = Override(a=2)

    s = State() | override
    assert s.a == 2

    s = State(a=1) | override
    assert s.a == 2


def test_slice():
    s1 = State(a=1, b=2, c=3)
    s2 = s1 | Slice('a', 'c')
    assert s2 == State(a=1, c=3)


def test_eval():
    s = State(a='value', b='${a}')
    t = s | Eval
    assert t.b == 'value'


def test_parallel():
    a = AsyncAction()
    b = AsyncAction()

    par = a // b
    assert isinstance(par, Parallel)


def test_many_parallel():
    a = AsyncAction()

    par = a**10
    assert isinstance(par, Parallel)


def sync_transformer(state):
    return state


@pytest.mark.parametrize(
    'a',
    [Action, Action(), lambda s: s, sync_transformer],
    ids=['class', 'instance', 'lambda', 'function'],
)
@pytest.mark.parametrize(
    'b',
    [Action, Action()],
    ids=['class', 'instance'],
)
def test_make_chain(a, b):
    chain1 = a >> b
    assert isinstance(chain1, Chain)
    chain2 = b >> a
    assert isinstance(chain2, Chain)


async def async_transformer(state):
    return state


@pytest.mark.parametrize(
    'a',
    [
        AsyncAction,
        AsyncAction(), Action,
        Action(), lambda s: s, sync_transformer, async_transformer
    ],
    ids=[
        'async_class', 'async_instance', 'sync_class', 'sync_instance', 'lambda',
        'function', 'coroutine'
    ],
)
@pytest.mark.parametrize(
    'b',
    [AsyncAction, AsyncAction()],
    ids=['class', 'instance'],
)
def test_make_async_chain(a, b):
    chain1 = a >> b
    assert isinstance(chain1, Chain)
    chain2 = b >> a
    assert isinstance(chain2, Chain)


def test_execute_chain():

    class MyTestAction1(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = 1
            return state

    class MyTestAction2(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = 2
            return state

    s1 = State(result=0) | MyTestAction1()
    s2 = State(result=0) | MyTestAction2()
    s3 = State(result=0) | MyTestAction1() >> MyTestAction2()
    s4 = State(result=0) | MyTestAction2() >> MyTestAction1()

    assert s1.result == 1
    assert s2.result == 2
    assert s3.result == 2
    assert s4.result == 1


def test_execute_async_chain():

    class MyTestAction1(AsyncAction):

        async def call_async(self, state=None):
            state = await super().call_async(state)
            state.result = 1
            return state

    class MyTestAction2(AsyncAction):

        async def call_async(self, state=None):
            state = await super().call_async(state)
            state.result = 2
            return state

    s1 = State(result=0) | MyTestAction1()
    s2 = State(result=0) | MyTestAction2()
    s3 = State(result=0) | MyTestAction1() >> MyTestAction2()
    s4 = State(result=0) | MyTestAction2() >> MyTestAction1()

    assert s1.result == 1
    assert s2.result == 2
    assert s3.result == 2
    assert s4.result == 1


def test_override_action():

    class MyTestAction1(Action):

        def __call__(self, state=None):
            state = super().__call__(state)
            state.result = 1
            return state

    s1 = State(result=0) | MyTestAction1() >> Override(result=2)
    s2 = State(result=0) | MyTestAction1() >> Override(State(result=2))
    s3 = {} | MyTestAction1() >> Override(result=2)
    s4 = {} | MyTestAction1() >> Override(State(result=2))

    assert s1.result == 2
    assert s2.result == 2
    assert s3.result == 2
    assert s4.result == 2


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

        async def call_async(self, state=None):
            state = await super().call_async(state)
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

        async def call_async(self, state=None):
            state = await super().call_async(state)
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

        async def call_async(self, state=None, /, **kwargs):
            state = await super().call_async(state, **kwargs)
            state.answer = state.INSTANCE_NUMBER
            return state

    def merge(state):
        state.answers = [result.answer for result in state.results]
        del state.results
        return state

    async def run_test():
        # par = Parallel([AsyncTestAction] * 10, merge)
        par = Parallel([AsyncTestAction] * 10)
        state = State(results=[None] * 10)
        state = await par.call_async(state)
        # return state
        return state | merge

    state = asyncio.run(run_test())
    assert len(state.answers) == 10
    assert state.answers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_execute_1000_parallel():

    n = 1000
    SLEEP = 0.05

    class Sleeper(AsyncAction):

        async def call_async(self, state=None):
            state = await super().call_async(state)
            await asyncio.sleep(SLEEP)
            # state.results[state.INSTANCE_NUMBER] = True
            state.result = True
            return state

    start_time = time()
    s = {} | Sleeper()**n
    elapsed_time = time() - start_time

    # If elapsed was less than 1% of the cumulatime time, then it's parallel enough
    assert elapsed_time < SLEEP * n * 0.01

    for i in range(n):
        assert s.results[i].result is True

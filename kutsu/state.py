"""State which can be passed from action to action and mutated by them"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Iterator, Optional, Protocol, Type, TypeGuard, Union


def short_repr(thing: Any) -> str:
    if isinstance(thing, list):
        things = ', '.join([short_repr(t) for t in thing])
        return f'[{things}]'

    if isinstance(thing, State):
        name = thing.__class__.__name__
        items = {
            k: v
            for k, v in thing.__dict__.items() if k not in {'COOKIES'} and not k.startswith('_')
        }
        if len(items) == 0:
            return '<Empty State>'
        if name == 'State':
            return f'State<{len(items)} items>'
        return f'State<{name}>'

    return repr(thing)


class SyncStateTransformer(Protocol):

    def __call__(self, state: State) -> State:
        ...


class AsyncStateTransformer(Protocol):

    async def __call__(self, state: State) -> State:
        ...


SyncStateTransformerArg = Union[SyncStateTransformer, Type[SyncStateTransformer]]
AsyncStateTransformerArg = Union[AsyncStateTransformer, Type[AsyncStateTransformer]]

StateTransformer = Union[SyncStateTransformer, AsyncStateTransformer]
StateTransformerArg = Union[SyncStateTransformerArg, AsyncStateTransformerArg]


def async_callable(x: Any) -> TypeGuard[AsyncStateTransformer]:
    """True if x is async callable"""
    return (
        inspect.iscoroutinefunction(x) or inspect.iscoroutinefunction(getattr(x, '__call__', None))
    )


def sync_callable(x: Any) -> TypeGuard[SyncStateTransformer]:
    """True if x is callable but not async"""
    return callable(x) and not async_callable(x)


class MetaChainable(type):
    """Operator overloading for Chainable class. We mostly instantiate classes for convenience.
    Please don't use this as a metaclass for anything else than a Chainable subclass.
    """

    def __repr__(cls) -> str:
        return f'Chainable<{cls.__name__}>'

    def __rshift__(
        cls,
        other: Chainable | Type[Chainable] | State | Type[State] | StateTransformer | dict[str, Any]
    ) -> SyncOrAsyncChain:
        self: Chainable = cls()
        return self.__rshift__(other)

    def __rrshift__(
        cls, other: State | Type[State] | StateTransformer | dict[str, Any]
    ) -> SyncOrAsyncChain:
        self: Chainable = cls()
        return self.__rrshift__(other)

    def __ror__(cls, other: dict[str, Any]) -> State:  # type: ignore
        self: Chainable = cls()
        return self.__ror__(other)


class Chainable(metaclass=MetaChainable):
    """Chainable action. Not very usable by itself - subclass Action or AsyncAction instead.

    This simply defines the right shift (>>) operator overloading to make chaining possible,
    as well as allowing dicts to be piped into actions using the or (|) operator overload.

    It is missing the definition of __call__, which is synchronous for Action and asynchronous
    for AsyncAction, and we cannot define both in this class.
    """

    @staticmethod
    def from_func(
        func: Callable[[State], State] | Callable[[State], Awaitable[State]]
    ) -> Action | AsyncAction:
        if async_callable(func):
            return AsyncAction(func)
        return Action(func)  # type: ignore

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __rshift__(
        self,
        other: Chainable | Type[Chainable] | State | Type[State] | StateTransformer
        | dict[str, Any],
    ) -> SyncOrAsyncChain:
        """Combine chainables into a new chain"""
        if inspect.isclass(other):
            other = other()

        if isinstance(self, (Chain, AsyncChain)):
            # We must append actions to this existing chain
            a = self.actions
        else:
            # We must create a new chain
            a = [self]  # type: ignore

        if isinstance(other, (Chain, AsyncChain)):
            # We combine the actions
            b = other.actions
        elif isinstance(other, (State, dict)):
            # We create an action that overrides state with this state
            if isinstance(other, dict):
                override = State(other)
            else:
                override = other

            # TODO: Perhaps make a proper Action subclass out of this?
            def override_state(state: State) -> State:
                return override(state)

            b = [Action(override_state)]  # type: ignore
        elif isinstance(other, Chainable):
            # We just append the chainable
            b = [other]  # type: ignore
        elif sync_callable(other):
            # We create an action that transforms state with this function
            b = [Action(other)]  # type: ignore
        elif async_callable(other):
            # We create an action that transforms state with this coroutine
            b = [AsyncAction(other)]  # type: ignore
        else:
            return NotImplemented

        new_chain = a + b

        if all(isinstance(x, Action) for x in new_chain):
            return Chain(new_chain)  # type: ignore

        return AsyncChain(new_chain)  # type: ignore

    def __rrshift__(
        self, other: State | Type[State] | StateTransformer | dict[str, Any]
    ) -> SyncOrAsyncChain:
        """Chain starting with a State or a dict"""
        if inspect.isclass(other):
            override = other()
        else:
            override = other

        if isinstance(override, (State, dict)):
            if isinstance(override, dict):
                override = State(override)

            # TODO: Perhaps make a proper Action subclass out of this?
            def override_state(state: State) -> State:
                return override(state)  # type: ignore

            # Return an Action that overrides given state with this state
            return Action(override_state) >> self

        if sync_callable(override):
            # We were given a function or coroutine
            return Action(override) >> self

        if async_callable(override):
            # We were given a coroutine
            return AsyncAction(override) >> self

        return NotImplemented

    def __ror__(self, other: dict[str, Any]) -> State:
        """We must be able to: {'key': 'value'} | Action()"""
        state = State(other)
        return state.__or__(self)  # type: ignore


class MetaAction(MetaChainable):
    """Operator overloading for Action"""

    def __repr__(cls) -> str:
        return f'Action<{cls.__name__}>'


class Action(Chainable, metaclass=MetaAction):
    """Synchronous action"""
    func: Callable[[State], State] | None

    def __init__(self, func: Callable[[State], State] | None = None) -> None:
        if func is not None and not sync_callable(func):
            raise TypeError('Expected non-coroutine function')
        self.func = func

    def __call__(
        self,  # pylint:disable=unused-variable
        state: State | None = None,
    ) -> State:
        if self.func is not None:
            return self.func(state or State())
        return state or State()

    def __repr__(self) -> str:
        if self.func is not None:
            return f'{self.__class__.__name__}<function {self.func.__name__}>()'
        return f'{self.__class__.__name__}()'


class MetaAsyncAction(MetaChainable):
    """Operator overloading for AsyncAction"""

    def __repr__(cls) -> str:
        return f'AsyncAction<{cls.__name__}>'

    def __floordiv__(cls, other: AsyncAction) -> Parallel:
        # Return type would be Parallel[Intersection[S, U], Intersection[T,V]] if
        # only intersection types were supported.
        # We instantiate here for convenience
        self: AsyncAction = cls()
        return self.__floordiv__(other)

    def __pow__(cls, n: int) -> Parallel:
        # We instantiate here for convenience
        self: AsyncAction = cls()
        return self.__pow__(n)


class AsyncAction(Chainable, metaclass=MetaAsyncAction):
    """An asynchronous action"""
    coro: Callable[[State], Awaitable[State]] | None

    def __init__(self, coro: Callable[[State], Awaitable[State]] | None = None) -> None:
        if coro is not None and not async_callable(coro):
            raise TypeError('Expected coroutine function')
        self.coro = coro

    def __floordiv__(self, other: AsyncAction) -> Parallel:
        """Parallelize actions"""
        # Return type would be Parallel[Intersection[S, U], Intersection[T,V]] if
        # only intersection types were supported.
        if inspect.isclass(other):
            if not issubclass(other, AsyncAction):
                return NotImplemented
            # We instantiate here for convenience
            other = other()  # type: ignore
        if not isinstance(other, AsyncAction):
            return NotImplemented  # type: ignore
        if isinstance(self, Parallel):
            a = self.actions  # pylint:disable=no-member
        else:
            a = [self]
        if isinstance(other, Parallel):
            b = other.actions
        else:
            b = [other]
        return Parallel(a + b)

    def __pow__(self, n: int) -> Parallel:
        """Run n instances of self in parallel"""
        if not isinstance(n, int):
            return NotImplemented  # type: ignore
        return Parallel([self] * n)

    async def __call__(
        self,  # pylint:disable=unused-variable
        state: State | None = None,
    ) -> State:
        if self.coro is not None:
            return await self.coro(state or State())
        return state or State()

    def __repr__(self) -> str:
        if self.coro is not None:
            return f'{self.__class__.__name__}<coroutine {self.coro.__name__}>()'
        return f'{self.__class__.__name__}()'


def simple_state_merge(state: State, results: list[ParallelState]) -> State:
    """Merges a list of result states into a source state"""
    for r in results:
        state = r(state)
    return state


class Parallel(AsyncAction):
    """A set of actions that can be executed in parallel"""

    def __init__(self, actions: list[AsyncAction], merge: Optional[MergeFn] = None) -> None:
        super().__init__()
        for action in actions:
            if not async_callable(action):
                raise ValueError('All parallel actions must be async callable')

        self.actions = actions
        self.merge = merge

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({short_repr(self.actions)})"

    def __len__(self) -> int:
        return len(self.actions)

    async def __call__(  # pylint:disable=no-self-argument
        self,
        state: State | None = None,
        merge: MergeFn | None = None,
    ) -> State:
        state_ = state or State()
        queue: list[Awaitable[State]] = []

        for instance_number, action in enumerate(self.actions):
            queue.append(
                (state_(INSTANCE_NUMBER=instance_number) >> action)(state_)  # type: ignore
            )

        results: list[ParallelState] = await asyncio.gather(*queue)

        merge_fn = merge or self.merge or simple_state_merge
        state_ = merge_fn(state_, results)

        if hasattr(state_, 'INSTANCE_NUMBER'):
            # Delete instance number after all have been executed
            delattr(state_, 'INSTANCE_NUMBER')

        return state_


class Chain(Action):
    """A sequence of synchronous actions"""
    actions: list[SyncStateTransformer]

    def __init__(self, actions: list[SyncStateTransformer]) -> None:
        super().__init__()
        for action in actions:
            if not callable(action):
                raise ValueError('All actions must be callable')
            if async_callable(action):
                raise ValueError('Chain cannot accept async callables. '
                                 'Use AsyncChain instead.')

        self.actions = actions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({short_repr(self.actions)})"

    def __len__(self) -> int:
        return len(self.actions)

    def __call__(self, state: State | None = None) -> State:
        s = state or State()
        for action in self.actions:
            s = action(s)
        return s


class AsyncChain(AsyncAction):
    """An sequence of asynchronous and possibly synchronous actions"""
    actions: list[StateTransformer]

    def __init__(self, actions: list[StateTransformer]) -> None:
        super().__init__()
        for action in actions:
            if not callable(action):
                raise ValueError('All actions must be callable')

        self.actions = actions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({short_repr(self.actions)})"

    def __len__(self) -> int:
        return len(self.actions)

    async def __call__(self, state: State | None = None) -> State:
        s = state or State()
        for action in self.actions:
            if async_callable(action):
                s = await action(s)
            else:
                s = action(s)  # type: ignore
        return s


SyncOrAsyncChain = Union[Chain, AsyncChain]


def execute_sync(action: SyncStateTransformer, state: State) -> State:
    """Execute synchronous action"""
    if not sync_callable(action):
        raise RuntimeError('Action must be synchronous')
    retval = action(state)
    return retval


def execute_async(action: AsyncStateTransformer, state: State) -> State:
    """Execute asynchronous action

    This function is our gateway from synchronous to asynchronous actions.
    If an existing event loop is found, the action is executed in that loop.
    Otherwise a new event loop is created and the action is executed in it."""

    # FIXME: make sure this works in python REPL, ipython, jupyter

    if not async_callable(action):
        raise RuntimeError('Action must be asynchronous')

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop
        return asyncio.run(action(state))

    try:
        import nest_asyncio

        # TODO: this should be optional, configurable and probably only done once
        nest_asyncio.apply()
    except ImportError:
        import warnings
        warnings.warn(
            'Package nest_asyncio not found. Unable to patch asyncio to be re-entrant. '
            'This may cause failure of asynchronous actions.',
            category=RuntimeWarning,
        )

    return loop.run_until_complete(action(state))


# def execute(action: Action[T, U], state: T) -> U:
def execute(action: StateTransformer, state: State) -> State:
    """Execute action using state as input"""
    if sync_callable(action):
        return execute_sync(action, state)
    if async_callable(action):
        return execute_async(action, state)

    raise TypeError(f"Action {action} is not callable")


class MetaState(type):
    """Metaclass for State - convenience methods for adding and oring classes"""

    def __add__(cls, other: State | Type[State]) -> State:
        self: State = cls()
        return self.__add__(other)

    def __or__(cls, other: StateTransformerArg) -> State:  # type: ignore
        self: State = cls()
        return self.__or__(other)


class State(metaclass=MetaState):
    """Object to hold key-value pairs as attributes"""

    # def __add__(self: T: other: U) -> T & U:  # https://github.com/python/typing/issues/213
    def __add__(self, other: State | Type[State]) -> State:
        """Combine two states by overriding this state with the other one

            self + other = other(self)
            [State + State -> State]
        """
        if inspect.isclass(other):
            other = other()
        return other(self)

    # def __or__(self: T, other: Action[T, U]) -> U:
    def __or__(self, action: StateTransformerArg) -> State:
        """Execute by piping contents of this state to an action"""
        if inspect.isclass(action):
            action = action()
        return execute(action, self)

    # def __call__(self: T, state: U, /, **kwargs: V) -> T & U & V:
    def __call__(
        self,
        state: Optional[State | Type[State] | dict[str, Any]] = None,
        /,
        **kwargs: Any
    ) -> State:
        """Create a new state by applying self to state, overriding given state with values
        from self. If state is None, a new state is created. Values from kwargs override
        values from both state and self."""
        new_state = State()
        if state is not None:
            if isinstance(state, dict):
                for var in state:
                    setattr(new_state, var, state[var])
            else:
                for var in vars(state):
                    setattr(new_state, var, getattr(state, var))
        for var in vars(self):
            setattr(new_state, var, getattr(self, var))
        for key, value in kwargs.items():
            setattr(new_state, key, value)
        return new_state

    def __init__(
        self,
        state: Optional[State | Type[State] | dict[str, Any]] = None,
        /,
        **kwargs: Any
    ) -> None:
        """Create a new state, optionally copying values from given state and kwargs"""
        # We make sure every class variable is available in self.__dict__
        d = {}
        for var in dir(self.__class__):
            if not var.startswith('__'):
                d[var] = getattr(self.__class__, var)
        for key, value in d.items():
            setattr(self, key, value)
        if state is not None:
            if isinstance(state, dict):
                for var in state:
                    setattr(self, var, state[var])
            else:
                for var in vars(state):
                    setattr(self, var, getattr(state, var))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(f'Key {key} not found')
        return getattr(self, key)

    # This is here to make mypy happy
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        delattr(self, key)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if name == 'State':
            title = 'State'
        else:
            title = f'State<{name}>'
        visible_items = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        hidden_items = {k: v for k, v in self.__dict__.items() if k.startswith('_')}
        if len(visible_items) + len(hidden_items) == 0:
            return '<Empty State>'
        hidden = f'({len(hidden_items)} hidden)' if len(hidden_items) > 0 else ''
        name_lengths = [len(name) for name in visible_items]
        width = min(30, max(name_lengths))
        return f'<{title} {hidden}\n' + '\n'.join(
            [
                f'  {k.ljust(width)} = ({type(v).__name__}) {repr(v)}'
                for k, v in visible_items.items()
            ]
        ) + '\n>'


class StateProtocol(Protocol):
    """Protocol for State"""

    def __add__(self, other: State | Type[State]) -> State:
        ...

    def __or__(self, other: Chainable | Type[Chainable]) -> Any:
        ...

    def __call__(
        self, state: State | Type[State] | dict[str, Any] | None = None, /, **kwargs: Any
    ) -> State:
        ...

    def __init__(
        self, state: State | Type[State] | dict[str, Any] | None = None, /, **kwargs: Any
    ) -> None:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[str]:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...


class ParallelState(StateProtocol):
    """State for Parallel instances"""
    INSTANCE_NUMBER: int


MergeFn = Callable[[State, list[ParallelState]], State]

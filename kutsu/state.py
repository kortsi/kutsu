"""State which can be passed from action to action and mutated by them"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from inspect import iscoroutinefunction as iscoro
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Protocol,
    Sequence,
    Union,
)

# This is what is typically accepted as argument when State is passed
StateArg = Union['State', type['State'], Mapping[str, Any]]


class StateTransformerFunc(Protocol):

    def __call__(self, state: StateArg | None) -> State:
        ...


class StateTransformerCoro(Protocol):

    async def __call__(self, state: StateArg | None) -> State:
        ...


# StateTransformer can be a function, a coroutine, or an Action instance
# An Action subclass is typically accepted as argument as well
StateTransformer = Union[StateTransformerFunc, StateTransformerCoro, 'Action',
                         type['Action']]


class MetaAction(type):
    """Operator overloading for Action class.

    We mostly instantiate classes forconvenience. Please don't use this as a
    metaclass for anything else than an Action subclass.
    """

    def __repr__(cls) -> str:
        return cls.__name__

    def __rshift__(cls, other: StateTransformer | StateArg) -> Chain:
        self: Action = cls()
        return self.__rshift__(other)

    def __rrshift__(cls, other: StateArg) -> Chain:
        self: Action = cls()
        return self.__rrshift__(other)

    def __ror__(cls, other: Mapping[str, Any] | None) -> State:  # type: ignore
        self: Action = cls()
        return self.__ror__(other)

    def __or__(cls, other: StateTransformer) -> State:
        self: Action = cls()
        return self.__or__(other)


class Action(metaclass=MetaAction):
    """Action"""
    _func: Callable[[State], State] | None = None

    def __init__(self, func: Callable[[StateArg], State] | None = None) -> None:
        if func is not None and (isinstance(func, AsyncAction) or iscoro(func)):
            raise TypeError('Expected non-coroutine function')
        self._func = func

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        if self._func is not None:
            return self._func(State(state))
        return State(state)

    def __repr__(self) -> str:
        if self._func is not None:
            return f'{self.__class__.__name__}(function {self._func.__name__})()'
        return f'{self.__class__.__name__}()'

    def __rshift__(self, other: StateTransformer | StateArg) -> Chain:
        """Combine chainables into a new chain"""
        if inspect.isclass(other):
            other = other()

        if isinstance(self, Chain):
            # We must append actions to this existing chain
            a = self.actions
        else:
            # We must create a new chain
            a = [self]

        if isinstance(other, Chain):
            # We combine the actions
            b = other.actions
        elif isinstance(other, (State, Mapping)):
            # We create an action that overrides state with this state
            b = [Override(other)]
        elif isinstance(other, Action):
            # We just append the chainable
            b = [other]
        elif iscoro(other):
            # We create an action that transforms state with this coroutine
            b = [AsyncAction(other)]
        elif callable(other):
            # We create an action that transforms state with this function
            b = [Action(other)]  # type: ignore
        else:
            return NotImplemented  # type: ignore

        return Chain(a + b)

    def __rrshift__(self, other: StateArg) -> Chain:
        """Chain starting with a State or a Mapping"""
        if inspect.isclass(other):
            override = other()
        else:
            override = other

        if isinstance(override, (State, Mapping)):
            return Override(override) >> self

        if iscoro(override):
            # We were given a coroutine
            return AsyncAction(override) >> self

        if callable(override):
            # We were given a function or coroutine
            return Action(override) >> self

        return NotImplemented

    def __ror__(self, other: Mapping[str, Any] | None) -> State:
        """This is here to support piping dicts to Actions

        Eg. {'key': 'value'} | Action()
        Also: None | Action()
        """
        if isinstance(other, Mapping) or other is None:
            state = State(other)
            return state.__or__(self)
        return NotImplemented  # type: ignore

    def __or__(self, other: StateTransformer) -> State:
        """This is here to support piping empty state to a pair of Actions

        Eg. Action() | Action()
        """
        return State() | self | other


class MetaAsyncAction(MetaAction):
    """Metaclass for AsyncAction"""

    def __repr__(cls) -> str:
        return cls.__name__

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


class AsyncAction(Action, metaclass=MetaAsyncAction):
    """An asynchronous action"""
    _coro: Callable[[State], Awaitable[State]] | None = None

    def __init__(  # pylint: disable=super-init-not-called
        self,
        coro: Callable[[StateArg], Awaitable[State]] | None = None,
    ) -> None:
        if coro is not None and not iscoro(coro):
            raise TypeError('Expected coroutine function')
        self._coro = coro

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
        # TODO: find a way to instantiate n actions instead of using n refs to one action
        return Parallel([self] * n)

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **kwargs: Any,
    ) -> State:
        # import trio
        # return trio.run(self.call_async, State(state), **kwargs)
        try:
            loop = asyncio.get_running_loop()
            # raise RuntimeError('Cannot call async action when a loop is already running')
        except RuntimeError:
            # No running loop
            loop = None

        if loop is None:
            return asyncio.run(self.call_async(State(state), **kwargs))

        # TODO: replace with ThreadPooExecutor?
        try:
            import nest_asyncio

            # TODO: this should be optional, configurable and probably only done once
            # FIXME: I think we should not do this. If someone needs to do it, they
            # FIXME: can just: await action.call_async(...)
            nest_asyncio.apply()
        except ImportError:
            import warnings
            warnings.warn(
                'Package nest_asyncio not found. '
                'Unable to patch asyncio to be re-entrant. '
                'This may cause failure of asynchronous actions.',
                category=RuntimeWarning,
            )

        return loop.run_until_complete(self.call_async(State(state), **kwargs))

    async def call_async(
        self,
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        if self._coro is not None:
            return await self._coro(State(state))
        return State(state)

    def __repr__(self) -> str:
        if self._coro is not None:
            return f'{self.__class__.__name__}(coroutine {self._coro.__name__})()'
        return f'{self.__class__.__name__}()'


class Parallel(AsyncAction):
    """A set of async actions that are executed in parallel

    The input state is fed into each action with an added extra attribute
    `INSTANCE_NUMBER`, which is the index of the action in the list of actions.

    The resulting state has a single attribute, `results`, which is a list of
    the resulting states of each action.

    If `return_exceptions` is True, the `results` list may contain exceptions
    raised by actions that failed. For successful actions, the list will contain
    a State instance.

    If `return_exceptions` is False (the default), the `results` list will only
    contain State instances, and the first failure in any actions will raise an
    exception. The rest of the actions are not cancelled in this case.
    """
    actions: list[AsyncAction]
    return_exceptions: bool

    def __init__(
        self,
        actions: Sequence[AsyncAction | type[AsyncAction] | StateTransformerCoro],
        return_exceptions: bool = False,
    ) -> None:
        super().__init__()
        actions_ = []
        for action in actions:
            if isinstance(action, AsyncAction):
                actions_.append(action)
            elif inspect.isclass(action) and issubclass(action, AsyncAction):
                actions_.append(action())
            elif iscoro(action):
                actions_.append(AsyncAction(action))
            else:
                raise ValueError('All parallel actions must be async actions')

        self.actions = actions_
        self.return_exceptions = return_exceptions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.actions})"

    def __len__(self) -> int:
        return len(self.actions)

    async def call_async(self, state: StateArg | None, /, **__: Any) -> State:
        state_ = State(state)
        queue: list[Awaitable[State]] = []

        for instance_number, action in enumerate(self.actions):
            action = state_(INSTANCE_NUMBER=instance_number) >> action
            queue.append(action.call_async(state_))

        return State(
            # TODO: allow customizing the name of the results attribute
            # TODO: add an action to merge results into a single state
            results=await
            asyncio.gather(*queue, return_exceptions=self.return_exceptions)
        )


class Identity(Action):
    """An action that returns the state it receives"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        return State(state)


class Default(Action):
    """Uses given state as default. Incoming state will override defaults.

    Eg:

        Default(a=0) >> Action

    Will send a=0 to Action, while:

        State(a=1) | Default(a=0) >> Action

    Will send a=1 to Action.
    """

    def __init__(self, state: StateArg | None = None, /, **kwargs: Any) -> None:
        super().__init__()
        self.bound_state = State(state, **kwargs)

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        return (State(state))(self.bound_state)


class Slice(Action):
    """Get a subset of state"""

    def __init__(self, *keys: str) -> None:
        super().__init__()
        self.keys = keys

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        state = State(state)
        return State(**{k: state[k] for k in self.keys})


class Eval(Action):
    """Evaluate state. Substitute all variables with their evaluated expressions."""

    def __init__(self, mask_secrets: bool = False) -> None:
        super().__init__()
        self.mask_secrets = mask_secrets

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        from .expressions import evaluate
        state = State(state)
        for key in state:
            state[key] = evaluate(state[key], state, mask_secrets=self.mask_secrets)
        return state


class Override(Action):
    """Override state with given state"""

    def __init__(self, state: StateArg | None = None, /, **kwargs: Any) -> None:
        super().__init__()
        self.bound_state = State(state, **kwargs)

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **__: Any,
    ) -> State:
        return self.bound_state(state)


class Chain(AsyncAction):
    """A set of actions that are executed in sequence"""
    actions: list[Action]

    def __init__(self, actions: Sequence[StateTransformer]) -> None:
        super().__init__()
        self.actions = []
        for action in actions:
            if inspect.isclass(action) and issubclass(action, Action):
                self.actions.append(action())
            elif isinstance(action, Action):
                self.actions.append(action)
            elif isinstance(action, (State, Mapping)):
                self.actions.append(Override(action))
            elif iscoro(action):
                self.actions.append(AsyncAction(action))
            elif callable(action):
                self.actions.append(Action(action))  # type: ignore
            else:
                raise TypeError(f'Not chainable: {action}')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.actions})"

    def __len__(self) -> int:
        return len(self.actions)

    def __call__(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **kwargs: Any,
    ) -> State:
        if any(isinstance(a, AsyncAction) or iscoro(a) for a in self.actions):
            # We have async actions, so we use Action.__call__ to get an ioloop
            return super().__call__(state)

        # All actions are synchronous so we just call them directly
        s = State(state)
        for action in self.actions:
            s = action(s)
        return s

    async def call_async(
        self,  # pylint:disable=unused-variable
        state: StateArg | None,
        /,
        **kwargs: Any,
    ) -> State:
        s = State(state)
        for action in self.actions:
            if isinstance(action, AsyncAction):
                s = await action.call_async(s, **kwargs)
            elif iscoro(action):
                s = await action(s, **kwargs)
            else:
                s = action(s, **kwargs)
        return s


class MetaState(type):
    """Metaclass for State - convenience methods for adding and oring classes"""

    def __add__(cls, other: StateArg) -> State:
        self: State = cls()
        return self.__add__(other)

    def __or__(  # type: ignore
        cls, other: StateTransformer
    ) -> State:
        self: State = cls()
        return self.__or__(other)

    def __eq__(cls, other: object) -> bool:
        self: State = cls()

        if isinstance(other, State):
            return self.__eq__(other)

        if inspect.isclass(other) and issubclass(other, State):
            return self.__eq__(other())

        return NotImplemented

    def __hash__(cls) -> int:
        return hash((cls.__module__, cls.__name__))


class State(metaclass=MetaState):
    """Object to hold key-value pairs as attributes"""

    # def __add__(self: T: other: U) -> T & U:
    # # https://github.com/python/typing/issues/213
    def __add__(self, other: StateArg) -> State:
        """Combine two states by overriding this state with the other one

            self + other = other(self)
            [State + State -> State]
        """
        other = State(other)
        return other(self)

    def __or__(self, action: StateTransformer) -> State:
        """Execute by piping contents of this state to an action

        If the action is asynchronous, and an existing event loop is found,
        the action is executed in that loop. Otherwise a new event loop is
        created and the action is executed in it.
        """
        if inspect.isclass(action) and issubclass(action, Action):
            action = action()
        elif isinstance(action, Action):
            pass
        elif isinstance(action, (State, Mapping)):
            action = Override(action)
        elif iscoro(action):
            action = AsyncAction(action)
        elif callable(action):
            action = Action(action)  # type: ignore
        else:
            return NotImplemented  # type: ignore

        return action(self)

    def __call__(self, state: StateArg | None = None, /, **kwargs: Any) -> State:
        """Create a new state by applying self to state, overriding given state
        with values from self. If state is None, a new state is created. Values
        from kwargs override values from both state and self."""
        new_state = State()
        if state is not None:
            if isinstance(state, Mapping):
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

    def __init__(self, state: StateArg | None = None, /, **kwargs: Any) -> None:
        """Create a new state, optionally copying values from given state and
        kwargs"""
        if inspect.isclass(state) and issubclass(state, State):
            state = state()
        if not isinstance(state, (State, Mapping)) and state is not None:
            raise TypeError(f'Expected State. Mapping or None, got {type(state)}')

        # We make sure every class variable is available in self.__dict__
        d = {}
        for var in dir(self.__class__):
            if not var.startswith('__'):
                d[var] = getattr(self.__class__, var)
        for key, value in d.items():
            setattr(self, key, value)
        if state is not None:
            if isinstance(state, Mapping):
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

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        delattr(self, key)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return (
                all(key in other for key in self) and all(key in self for key in other)
                and all(self[key] == other[key] for key in self)
                and all(self[key] == other[key] for key in other)
            )

        if inspect.isclass(other) and issubclass(other, State):
            other = other()
            return (
                all(key in other for key in self) and all(key in self for key in other)
                and all(self[key] == other[key] for key in self)
                and all(self[key] == other[key] for key in other)
            )

        return NotImplemented

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

    def __rich_repr__(self) -> Generator[Any, None, None]:
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                yield k, v


class StateProtocol(Protocol):
    """Protocol for State"""

    def __add__(self, other: StateArg) -> State:
        ...

    def __or__(self, other: StateTransformer) -> Any:
        ...

    def __call__(self, state: StateArg | None = None, /, **kwargs: Any) -> State:
        ...

    def __init__(self, state: StateArg | None = None, /, **kwargs: Any) -> None:
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

    def __eq__(self, other: object) -> bool:
        ...

"""Expressions"""
from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping
from string import Template
from typing import Any

from .state import State


class Masked:
    """Masked value"""

    def __init__(self, type_: type) -> None:
        self.type = type_

    def __repr__(self) -> str:
        return f'Masked[{self.type.__name__}]'

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Masked) and self.type == other.type


class _MaskedStr(Masked):
    """Temporary placeholder for masked strings. Stripped from output of 'evaluate'."""

    def __init__(self, value: str) -> None:
        super().__init__(str)
        self.type = str
        self.value = value

    def __repr__(self) -> str:
        return f'_MaskedStr[{repr(self.value)}]'

    def __str__(self) -> str:
        return self.value


def remove_masked_str(data: Any) -> Any:
    """Remove _MaskedStr from data structure"""

    if isinstance(data, _MaskedStr):
        return data.value

    if isinstance(data, list):
        return [remove_masked_str(d) for d in data]

    if isinstance(data, set):
        return {remove_masked_str(d) for d in data}

    if isinstance(data, tuple):
        return tuple(remove_masked_str(d) for d in data)

    if isinstance(data, dict):
        return {remove_masked_str(k): remove_masked_str(v) for k, v in data.items()}

    return data


def evaluate(
    data: Any, state: State, mask_secrets: bool = False, as_str: bool = False
) -> Any:
    """Evaluate data structure using state"""
    data = _evaluate(data, state, mask_secrets=mask_secrets)

    data = remove_masked_str(data)

    if as_str:
        if isinstance(data, Masked):
            return '****MASKED****'

    return data


def _evaluate(data: Any, state: State, mask_secrets: bool = False) -> Any:
    """Evaluate data structure using state"""

    if isinstance(data, str):
        tpl = Template(data)

        # Find out which variables are used in the template
        variable_names: list[str] = []
        for _, a, b, _ in re.findall(
            tpl.pattern,
            tpl.template,
        ):
            # For each match, either a or b will be None, or both of them will be None
            if a or b:
                variable_names.append(a or b)

        variables = {}
        for v in variable_names:
            variables[v] = _evaluate(
                getattr(state, v, None), state, mask_secrets=mask_secrets
            )

        masked = False

        for v in variables:  # pylint: disable=consider-using-dict-items
            if variables[v] is None:
                # Replace occurrences of None with the empty string
                variables[v] = ''
            elif isinstance(variables[v], _MaskedStr):
                variables[v] = variables[v].value
                masked = True
            elif isinstance(variables[v], Masked):
                # Masked secret
                variables[v] = '****MASKED****'
                masked = True

        string = tpl.safe_substitute(**variables)
        if masked:
            return _MaskedStr(string)

        return string

    if isinstance(data, Secret):
        secret_value = _evaluate(data.arg0, state, mask_secrets=mask_secrets)
        if mask_secrets:
            if isinstance(secret_value, Masked):
                return secret_value
            return Masked(type(secret_value))
        return secret_value

    if isinstance(data, Del) or data is Del:
        # We let Del pass through because it will have to be handled
        # at the collection level
        return data

    if isinstance(data, list):
        L = []
        for d in data:
            result = _evaluate(d, state, mask_secrets=mask_secrets)
            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                L.append(result)
        return L

    if isinstance(data, set):
        S = set()
        for d in data:
            result = _evaluate(d, state, mask_secrets=mask_secrets)
            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                S.add(result)
        return S

    if isinstance(data, tuple):
        T = []
        for d in data:
            result = _evaluate(d, state, mask_secrets=mask_secrets)
            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                T.append(result)
        return tuple(T)

    if isinstance(data, dict):
        D = {}
        for k, v in data.items():
            result_key = _evaluate(k, state, mask_secrets=mask_secrets)
            result_value = _evaluate(v, state, mask_secrets=mask_secrets)
            # Exclude keys marked with value Del()
            if not isinstance(result_value, Del) and result_value is not Del:
                D[result_key] = result_value
        return D

    if isinstance(data, Node):
        return _evaluate(
            data(state, mask_secrets=mask_secrets), state, mask_secrets=mask_secrets
        )

    return data


class MetaNode(type):
    """Node type"""

    def __repr__(cls) -> str:
        return cls.__name__


class Node(metaclass=MetaNode):
    """Base class for all expression nodes"""

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        raise NotImplementedError(f'You must subclass {self.__class__.__name__}')

    # TODO: not sure if we want to enable these (tests also disabled)
    # Technically these do work, but they make it very easy to introduce errors and
    # can make things very hard to reason about. All sorts of things expect these
    # to return a boolean, while these would return a Node instead. Node will be
    # evaluated to True. This can lead to unexpected behavior.
    # You should use Eq() and Ne() directly instead.

    # def __eq__(self, other: Any) -> 'Node':  # type: ignore
    #     return Eq(self, other)

    # def __ne__(self, other: Any) -> 'Node':  # type: ignore
    #     return Ne(self, other)

    def __lt__(self, other: Any) -> 'Node':
        return Lt(self, other)

    def __le__(self, other: Any) -> 'Node':
        return Lte(self, other)

    def __gt__(self, other: Any) -> 'Node':
        return Gt(self, other)

    def __ge__(self, other: Any) -> 'Node':
        return Gte(self, other)

    def __add__(self, other: Any) -> 'Node':
        return Add(self, other)

    def __radd__(self, other: Any) -> 'Node':
        return Add(other, self)

    def __sub__(self, other: Any) -> 'Node':
        return Sub(self, other)

    def __rsub__(self, other: Any) -> 'Node':
        return Sub(other, self)

    def __mul__(self, other: Any) -> 'Node':
        return Mul(self, other)

    def __rmul__(self, other: Any) -> 'Node':
        return Mul(other, self)

    def __truediv__(self, other: Any) -> 'Node':
        return Div(self, other)

    def __rtruediv__(self, other: Any) -> 'Node':
        return Div(other, self)

    def __mod__(self, other: Any) -> 'Node':
        return Mod(self, other)

    def __rmod__(self, other: Any) -> 'Node':
        return Mod(other, self)

    def __floordiv__(self, other: Any) -> 'Node':
        return FloorDiv(self, other)

    def __rfloordiv__(self, other: Any) -> 'Node':
        return FloorDiv(other, self)

    def __pow__(self, other: Any) -> 'Node':
        return Pow(self, other)

    def __rpow__(self, other: Any) -> 'Node':
        return Pow(other, self)

    def __and__(self, other: Any) -> 'Node':
        return And(self, other)

    def __rand__(self, other: Any) -> 'Node':
        return And(other, self)

    def __or__(self, other: Any) -> 'Node':
        return Or(self, other)

    def __ror__(self, other: Any) -> 'Node':
        return Or(other, self)

    def __invert__(self) -> 'Node':
        return DelIfNone(self)

    # TODO: __neg__ and __pos__?


class MetaUnaryNode(MetaNode):
    """UnaryNode type"""

    def __repr__(cls) -> str:
        return cls.__name__


# TODO: make generic
# class UnaryNode(Node, Generic[T], metaclass=MetaUnaryNode):
#    def __init__(self, arg0: T) -> None:
#        self.arg0 = arg0


class UnaryNode(Node, metaclass=MetaUnaryNode):  # pylint: disable=abstract-method
    """Unary expression"""

    def __init__(self, arg0: Any) -> None:
        self.arg0 = arg0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.arg0)})'

    # def __rich_repr__(self) -> Generator[str, None, None]:
    #     yield self.arg0


class MetaBinaryNode(MetaNode):
    """BinaryNode type"""

    def __repr__(cls) -> str:
        return cls.__name__


class BinaryNode(Node, metaclass=MetaBinaryNode):  # pylint: disable=abstract-method
    """Binary expression"""

    def __init__(self, arg0: Any, arg1: Any) -> None:
        self.arg0 = arg0
        self.arg1 = arg1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.arg0)}, {repr(self.arg1)})'

    # def __rich_repr__(self) -> Generator[str, None, None]:
    #     yield self.arg0
    #     yield self.arg1


class MetaTernaryNode(MetaNode):
    """TernaryNode type"""

    def __repr__(cls) -> str:
        return cls.__name__


class TernaryNode(Node, metaclass=MetaTernaryNode):  # pylint: disable=abstract-method
    """Ternary expression"""

    def __init__(self, arg0: Any, arg1: Any, arg2: Any) -> None:
        self.arg0 = arg0
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'({repr(self.arg0)}, {repr(self.arg1)}, {repr(self.arg2)})'
        )

    # def __rich_repr__(self) -> Generator[str, None, None]:
    #     yield self.arg0
    #     yield self.arg1
    #     yield self.arg2


class UnaryBoolNode(UnaryNode):
    """Unary boolean expression"""

    def __call__(self, state: State, mask_secrets: bool = False) -> bool:
        return self.eval(_evaluate(self.arg0, state, mask_secrets=mask_secrets))

    def eval(self, value: Any) -> bool:
        """Evaluate"""
        raise NotImplementedError(f'You must subclass {self.__class__.__name__}')


class IsNone(UnaryBoolNode):
    """arg0 is None"""

    def eval(self, value: Any) -> bool:
        return value is None


class IsNotNone(UnaryBoolNode):
    """arg0 is not None"""

    def eval(self, value: Any) -> bool:
        return value is not None


class IsTrue(UnaryBoolNode):
    """arg0 is True"""

    def eval(self, value: Any) -> bool:
        return value is True


class IsNotTrue(UnaryBoolNode):
    """arg0 is not True"""

    def eval(self, value: Any) -> bool:
        return value is not True


class IsTruthy(UnaryBoolNode):
    """arg0 is truthy"""

    def eval(self, value: Any) -> bool:
        # print('IsTruthy', value, bool(value))
        return bool(value)


class IsNotTruthy(UnaryBoolNode):
    """arg0 is not truthy"""

    def eval(self, value: Any) -> bool:
        return not bool(value)


class IsFalse(UnaryBoolNode):
    """arg0 is False"""

    def eval(self, value: Any) -> bool:
        return value is False


class IsNotFalse(UnaryBoolNode):
    """arg0 is not False"""

    def eval(self, value: Any) -> bool:
        return value is not False


class IsFalsy(UnaryBoolNode):
    """arg0 is falsy"""

    def eval(self, value: Any) -> bool:
        return not bool(value)


class IsNotFalsy(UnaryBoolNode):
    """arg0 is not falsy"""

    def eval(self, value: Any) -> bool:
        return bool(value)


class Not(UnaryBoolNode):
    """not arg0"""

    def eval(self, value: Any) -> bool:
        return not value


# TODO: IsEmpty, IsNotEmpty


class BinaryBoolNode(BinaryNode):
    """Binary boolean expression"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> bool | Masked:
        first = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        second = _evaluate(self.arg1, state, mask_secrets=mask_secrets)
        if isinstance(first, Masked) or isinstance(second, Masked):
            return Masked(bool)
        return self.eval(first, second)

    def eval(self, value1: Any, value2: Any) -> bool:
        """Evaluate"""
        raise NotImplementedError(f'You must subclass {self.__class__.__name__}')


class Eq(BinaryBoolNode):
    """arg0 == arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 == value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} == {repr(self.arg1)})'


class Ne(BinaryBoolNode):
    """arg0 != arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 != value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} != {repr(self.arg1)})'


class Gt(BinaryBoolNode):
    """arg0 > arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 > value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} > {repr(self.arg1)})'


class Gte(BinaryBoolNode):
    """arg0 >= arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 >= value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} >= {repr(self.arg1)})'


class Lt(BinaryBoolNode):
    """arg0 < arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 < value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} < {repr(self.arg1)})'


class Lte(BinaryBoolNode):
    """arg0 <= arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 <= value2)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} <= {repr(self.arg1)})'


class In(BinaryBoolNode):
    """arg0 in arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 in value2)


class NotIn(BinaryBoolNode):
    """arg0 not in arg1"""

    def eval(self, value1: Any, value2: Any) -> bool:
        return bool(value1 not in value2)


# TODO: Is, IsNot, Any, All


class BinaryAlgebraNode(BinaryNode):
    """Binary algebra expression"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        # Evaluate data
        first = _evaluate(self.arg0, state, mask_secrets=False)
        second = _evaluate(self.arg1, state, mask_secrets=False)
        data = _evaluate(self.eval(first, second), state, mask_secrets=False)

        if not mask_secrets:
            return data

        # See if we must return a secret
        first_secret = _evaluate(self.arg0, state, mask_secrets=True)
        second_secret = _evaluate(self.arg1, state, mask_secrets=True)
        if isinstance(first_secret, Masked) or isinstance(second_secret, Masked):
            # We should hide the result
            return Masked(type(data))

        # Plain value
        return data

    def eval(self, value1: Any, value2: Any) -> Any:
        """Evaluate"""
        raise NotImplementedError(f'You must subclass {self.__class__.__name__}')


class Add(BinaryAlgebraNode):
    """arg0 + arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 + value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} + {repr(self.arg1)})'


class Sub(BinaryAlgebraNode):
    """arg0 - arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 - value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} - {repr(self.arg1)})'


class Mul(BinaryAlgebraNode):
    """arg0 * arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 * value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} * {repr(self.arg1)})'


class Div(BinaryAlgebraNode):
    """arg0 / arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 / value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} / {repr(self.arg1)})'


class FloorDiv(BinaryAlgebraNode):
    """arg0 // arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 // value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} // {repr(self.arg1)})'


class Mod(BinaryAlgebraNode):
    """arg0 % arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1 % value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} % {repr(self.arg1)})'


class Pow(BinaryAlgebraNode):
    """arg0 ** arg1"""

    def eval(self, value1: Any, value2: Any) -> Any:
        return value1**value2

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} ** {repr(self.arg1)})'


class Del(Node):
    """Delete dict key or list item"""

    def __repr__(self) -> str:
        return 'Del()'

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        return RuntimeError('Del instance should not be called')


class DelIfNone(UnaryNode):
    """If arg0 is None then Del() else arg0"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        result = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        return Del() if result is None else result


class Var(UnaryNode):
    """Get variable arg0 from state and evaluate it"""

    def __call__(self, state: State, mask_secrets: bool = False) -> Any:
        # key = get_key(_subst_data(self.arg0, state))
        key = _evaluate(self.arg0, state)
        value = getattr(state, key, None)

        value = _evaluate(value, state, mask_secrets=mask_secrets)
        new_value = _evaluate(value, state, mask_secrets=mask_secrets)
        while new_value != value:
            value = new_value
            new_value = _evaluate(value, state, mask_secrets=mask_secrets)

        return value

    # TODO: This does not work well, investigate why
    # TODO: I think it could be made to work when restricted to State only
    # TODO: Check __getattribute__
    # def __getattr__(self, name: str) -> Any:
    #     return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)


class Env(UnaryNode):
    """Get variable arg0 from the environment. No evaluation is done."""

    def __call__(self, state: State, mask_secrets: bool = False) -> str | None:
        return os.getenv(self.arg0)


class GetItem(BinaryNode):
    """Subscription operation"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        obj = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        key = _evaluate(self.arg1, state, mask_secrets=False)
        value = obj[key]
        return _evaluate(value, state, mask_secrets=mask_secrets)

    # def __getattr__(self, name: str) -> Any:
    #     return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)

    def __repr__(self) -> str:
        return f'{repr(self.arg0)}[{repr(self.arg1)}]'


class GetAttr(BinaryNode):
    """Attribute get operation"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        obj = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        key = _evaluate(self.arg1, state, mask_secrets=False)
        value = getattr(obj, key)
        return _evaluate(value, state, mask_secrets=mask_secrets)

    # def __getattr__(self, name: str) -> Any:
    #    return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)

    def __repr__(self) -> str:
        return f'{repr(self.arg0)}.{repr(self.arg1)}'


class Secret(UnaryNode):
    """Make value secret

    This is used as a special placeholder which can be used to
    conceal the value when eg. logging"""

    def __call__(self, state: State, mask_secrets: bool = False) -> Any:
        return _evaluate(self.arg0, state, mask_secrets=mask_secrets)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({type(self.arg0).__name__})"

    @property
    def value(self) -> Any:
        """Get secret value"""
        return self.arg0


class Raise(UnaryNode):
    """Raise exception"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        raise RuntimeError(self.arg0)


class If(TernaryNode):
    """If arg0 is truthy then arg1 else arg2

        If('x', 'yes it is true', 'no it is not')
        If(Gt('y', 0), 'greater than zero', 'not positive')
    """

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        if isinstance(self.arg0, str):
            # We have a variable name
            p = Var(self.arg0)
        else:
            # We expect an expression of some kind
            p = self.arg0

        # FIXME: this will reveal Secret(bool) value of self.arg0
        truthy: bool = _evaluate(IsTruthy(p), state, mask_secrets=False)
        result = self.arg1 if truthy else self.arg2
        return _evaluate(result, state, mask_secrets=mask_secrets)


class Select(TernaryNode):
    """Select from arg1 using arg0 as key, default to arg2

        Select(
            'x',
            {
                'A': 'You chose A',
                'B': 'You chose B'
            },
            'Unknown selection'
        )
    """

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        if isinstance(self.arg0, str):
            # We have a variable name
            e = Var(self.arg0)
        else:
            # We expect an expression of some kind
            e = self.arg0

        key = _evaluate(e, state, mask_secrets=False)

        result = self.arg1[key] if key in self.arg1 else self.arg2

        # return _subst_data(result, state)
        return _evaluate(result, state, mask_secrets=mask_secrets)


class Json(UnaryNode):
    """Render json - prepare for json.dumps"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        data = _evaluate(self.arg0, state, mask_secrets=mask_secrets)

        def to_json(data: Any) -> Any:
            if isinstance(data, Mapping):
                return {k: to_json(v) for k, v in data.items()}

            if isinstance(data, Collection):
                return [to_json(d) for d in data]

            return data

        return to_json(data)


class Or(BinaryNode):
    """arg0 if arg0 is truthy else arg1"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        p = _evaluate(IsTruthy(self.arg0), state, mask_secrets=False)
        first = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        second = _evaluate(self.arg1, state, mask_secrets=mask_secrets)
        if isinstance(first, Masked) or isinstance(second, Masked):
            return Masked(bool)
        return first if p else second

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} | {repr(self.arg1)})'


class And(BinaryNode):
    """arg1 if arg0 is truthy else arg0"""

    def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
        p = _evaluate(IsTruthy(self.arg0), state, mask_secrets=False)
        first = _evaluate(self.arg0, state, mask_secrets=mask_secrets)
        second = _evaluate(self.arg1, state, mask_secrets=mask_secrets)
        if isinstance(first, Masked) or isinstance(second, Masked):
            return Masked(bool)
        return second if p else first

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} & {repr(self.arg1)})'


# class VarOr(BinaryNode):
#     """Var(arg0) if Var(arg0) is truthy else arg1"""
#
#     def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
#         if _evaluate(IsTruthy(Var(self.arg0)), state, mask_secrets=False):
#             result = Var(self.arg0)
#         else:
#             result = self.arg1
#
#         return _evaluate(result, state, mask_secrets=mask_secrets)
#
#
# class VarAnd(BinaryNode):
#     """arg1 if Var(arg0) is truthy else None"""
#
#     def __call__(self, state: 'State', mask_secrets: bool = False) -> Any:
#         if _evaluate(IsTruthy(Var(self.arg0)), state, mask_secrets=False):
#             result = self.arg1
#         else:
#             result = None
#
#         return _evaluate(result, state, mask_secrets=mask_secrets)

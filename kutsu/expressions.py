"""Expressions"""
from __future__ import annotations

import os
import re
from string import Template
from typing import Any

from .state import State


class StringValue:

    def __init__(self, template: Template, variables: dict[str, Any]) -> None:
        self.template = template
        self.variables = variables


class SecretValue:

    def __init__(self, value: Any) -> None:
        self.value = value


def get_key(key: str | StringValue) -> str:
    """Get key from str or StringValue"""
    if isinstance(key, StringValue):
        return reveal_str(key, mask_secrets=False)
    return key


def subst_str(s: str, state: State, mask_secrets: bool = False) -> str:
    """Substitute template variables in string using state"""
    s2 = _subst_str(s, state)
    return reveal_str(s2, mask_secrets=mask_secrets)

    # TODO: render until there are no changes
    # tpl = Template(s)
    # # Find out all variables requiring substitution and substitute them
    # variable_names: List[str] = []
    # for _, a, b, _ in re.findall(tpl.pattern, tpl.template):
    #     # For each match a or b or neither could be non-empty, but not both
    #     if a or b:
    #         variable_names.append(a or b)

    # variables = {}
    # for v in variable_names:
    #     variables[v] = subst_data(getattr(state, v, None), state, mask_secrets=mask_secrets)

    # # for _, a, b, _ in re.findall(tpl.pattern, tpl.template):
    # #     if a or b:
    # #         variables[a or b] = subst_data(getattr(state, a or b), state,
    # #             mask_secrets=mask_secrets)

    # return tpl.safe_substitute(**variables)
    # # return StringValue(tpl, variables)
    # # subst = {k: v if v is not None else '' for k, v in vars(state).items()}
    # # return Template(s).safe_substitute(**subst)


def reveal_str(sv: StringValue, mask_secrets: bool = False) -> str:
    """Render string"""
    if isinstance(sv, str):
        # TODO: why do we sometimes receive an str?
        return sv
    values = reveal_data(sv.variables, mask_secrets=mask_secrets)
    return sv.template.safe_substitute(**values)


def _subst_str(s: str, state: State) -> StringValue:
    assert isinstance(s, str)

    tpl = Template(s)

    # Find out all variables requiring substitution and substitute them
    variable_names: list[str] = []
    for _, a, b, _ in re.findall(
        tpl.pattern,  # pylint:disable=no-member
        tpl.template
    ):
        # For each match a or b or neither could be non-empty, but not both
        if a or b:
            variable_names.append(a or b)

    variables = {}
    for v in variable_names:
        # variables[v] = getattr(state, v)
        variables[v] = _subst_data(getattr(state, v, None), state)

    # Replace occurrences of None with the empty string
    # Use string "None" if you need to print "None"
    for v in variables:  # pylint:disable=consider-using-dict-items
        if variables[v] is None:
            variables[v] = ''

    return StringValue(tpl, variables)


def reveal_data(data: Any, mask_secrets: bool = False) -> Any:
    """Reveal or mask Secret() objects"""
    if isinstance(data, SecretValue):
        secret_value = data.value
        if mask_secrets:
            # print('mask Secret', secret_value)
            return f'*MASKED[{getattr(type(secret_value), "__name__", str(secret_value))}]*'
        # print('reveal Secret', secret_value)
        return reveal_data(secret_value)

    if isinstance(data, StringValue):
        # print('reveal StringValue', data.template.template, data.variables)
        return reveal_str(data, mask_secrets=mask_secrets)

    if isinstance(data, list):
        # print('reveal list', data)
        return [reveal_data(d, mask_secrets) for d in data]

    if isinstance(data, set):
        # print('reveal set', data)
        return {reveal_data(d, mask_secrets) for d in data}

    if isinstance(data, tuple):
        # print('reveal tuple', data)
        return (reveal_data(d, mask_secrets) for d in data)

    if isinstance(data, dict):
        # print('reveal dict', data)
        return {
            (reveal_str(k, mask_secrets)): reveal_data(v, mask_secrets)
            for k, v in data.items()
        }

    # print('reveal data', data)
    return data


# TODO: rename to "evaluate"
def subst_data(data: Any, state: State, mask_secrets: bool = False) -> Any:
    """Substitute template variable in data structure using state"""
    a = _subst_data(data, state)
    return reveal_data(a, mask_secrets)


def _subst_data(data: Any, state: State) -> Any:
    """Substitute template variable in data structure using state"""

    if isinstance(data, str):
        # print('str', data)
        return _subst_str(data, state)

    if isinstance(data, Secret):
        # print('secret', data)
        secret_value = _subst_data(data.arg0, state)
        return SecretValue(secret_value)

    if isinstance(data, Del) or data is Del:
        # We let Del pass through because it will have to be handled
        # at the collection level
        # print('Del', data)
        return data

    if isinstance(data, list):
        # print('list', data)
        L = []
        for d in data:
            result = _subst_data(d, state)
            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                # print('keep', result)
                L.append(result)
            # else:
            # print('del', result)

        return L

    if isinstance(data, set):
        # print('set', data)
        S = set()
        for d in data:
            result = _subst_data(d, state)
            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                # print('keep', result)
                S.add(result)
            # else:
            #     # print('del', result)

        return S

    if isinstance(data, tuple):
        # print('tuple', data)
        T = []
        for d in data:
            result = _subst_data(d, state)

            # Exclude values marked with Del()
            if not isinstance(result, Del) and result is not Del:
                # print('keep', result)
                T.append(result)
            # else:
            #     # print('del', result)

        return tuple(T)

    if isinstance(data, dict):
        # print('dict', data)
        D = {}
        for k, v in data.items():
            result_key = _subst_data(k, state)
            result_value = _subst_data(v, state)

            # Exclude keys marked with value Del()
            if not isinstance(result_value, Del) and result_value is not Del:
                # print('keep', result_key, result_value)
                D[result_key] = result_value
            # else:
            #     # print('del', result_key, result_value)

        return D

    if isinstance(data, State):
        # We do not call State because we want it to remain the same
        # TODO: why is this? we could convert it into a dict, for instance
        return data

    if callable(data):
        # print('callable', data, isinstance(data, Del))
        # We call the callable giving state as the only arg
        # Node instances work as well as functions or anything that can take
        # a state argument and return something
        return _subst_data(data(state), state)

    # Anything else we return as is
    # TODO: render until there are no changes
    # print('other', data)
    return data


class MetaNode(type):
    """Node type"""

    def __repr__(cls) -> str:
        return cls.__name__


class Node(metaclass=MetaNode):
    """Base class for all expression nodes"""

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __call__(self, state: 'State') -> Any:
        raise NotImplementedError(f'You must subclass {self.__class__.__name__}')

    def __eq__(self, other: Any) -> 'Node':  # type: ignore
        return Eq(self, other)

    def __ne__(self, other: Any) -> 'Node':  # type: ignore
        return Ne(self, other)

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

    # FIXME: neg?


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
        return f"{self.__class__.__name__}({repr(self.arg0)})"

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
        return f"{self.__class__.__name__}({repr(self.arg0)}, {repr(self.arg1)})"

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
        return f"{self.__class__.__name__}({repr(self.arg0)}, {repr(self.arg1)}, {repr(self.arg2)})"

    # def __rich_repr__(self) -> Generator[str, None, None]:
    #     yield self.arg0
    #     yield self.arg1
    #     yield self.arg2


class UnaryBoolNode(UnaryNode):
    """Unary boolean expression"""

    def __call__(self, state: State) -> bool:
        return self.eval(subst_data(self.arg0, state))

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

    def __call__(self, state: 'State') -> bool:
        return self.eval(subst_data(self.arg0, state), subst_data(self.arg1, state))

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

    def __call__(self, state: 'State') -> Any:
        # Evaluate data
        first = subst_data(self.arg0, state)
        second = subst_data(self.arg1, state)
        data = _subst_data(self.eval(first, second), state)

        # See if it should be secret
        first_secret = _subst_data(self.arg0, state)
        second_secret = _subst_data(self.arg1, state)
        if isinstance(first_secret, SecretValue) or isinstance(second_secret, SecretValue):
            # We should hide the result
            return SecretValue(data)

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

    def __call__(self, state: 'State') -> Any:
        return RuntimeError('Del instance should not be called')


class DelIfNone(UnaryNode):
    """If arg0 is None then Del() else arg0"""

    def __call__(self, state: 'State') -> Any:
        result = subst_data(self.arg0, state)
        return Del() if result is None else result


class Var(UnaryNode):
    """Get variable arg0 from state and evaluate it"""

    def __call__(self, state: 'State') -> Any:
        key = get_key(_subst_data(self.arg0, state))
        # print('key', key)
        if not hasattr(state, key):
            return None

        return _subst_data(state[key], state)

    # TODO: This does not work well, investigate why
    # TODO: I think it could be made to work when restricted to State only
    # TODO: Check __getattribute__
    # def __getattr__(self, name: str) -> Any:
    #     return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)


class Env(UnaryNode):
    """Get variable arg0 from the environment. No evaluation is done."""

    def __call__(self, state: State) -> str | None:
        return os.getenv(self.arg0)


class GetItem(BinaryNode):
    """List subscription operation"""

    def __call__(self, state: 'State') -> Any:
        l = _subst_data(self.arg0, state)  # noqa
        L = subst_data(self.arg0, state)
        k = get_key(_subst_data(self.arg1, state))
        v = L[k]
        if isinstance(l, SecretValue):
            return SecretValue(v)
        return v

    # def __getattr__(self, name: str) -> Any:
    #     return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)

    def __repr__(self) -> str:
        return f'{repr(self.arg0)}[{repr(self.arg1)}]'


class GetAttr(BinaryNode):
    """Attribute get operation"""

    def __call__(self, state: 'State') -> Any:
        l = _subst_data(self.arg0, state)  # noqa
        L = subst_data(self.arg0, state)
        k = get_key(_subst_data(self.arg1, state))
        v = getattr(L, k)
        if isinstance(l, SecretValue):
            return SecretValue(v)
        return v

    # def __getattr__(self, name: str) -> Any:
    #    return GetAttr(self, name)

    def __getitem__(self, name: Any) -> Any:
        return GetItem(self, name)

    def __repr__(self) -> str:
        return f'{repr(self.arg0)}.{repr(self.arg1)}'


class Secret(Var):
    """Get variable arg0 from state and evaluate it
    This is used as a special placeholder which can be used to
    conceal the value when eg. logging"""

    # TODO: I think it is wrong to subclass from Var?
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(****)"

    @property
    def value(self) -> Any:
        """Get secret value"""
        return self.arg0


class Raise(UnaryNode):
    """Raise exception"""

    def __call__(self, state: 'State') -> Any:
        raise RuntimeError(self.arg0)


class If(TernaryNode):
    """If arg0 is truthy then arg1 else arg2

        If('x', 'yes it is true', 'no it is not')
        If(Gt('y', 0), 'greater than zero', 'not positive')
    """

    def __call__(self, state: 'State') -> Any:
        if isinstance(self.arg0, str):
            # We have a variable name
            p = Var(self.arg0)
        else:
            # We expect an expression of some kind
            p = self.arg0

        truthy: bool = subst_data(IsTruthy(p), state)

        result = self.arg1 if truthy else self.arg2

        return _subst_data(result, state)


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

    def __call__(self, state: 'State') -> Any:
        if isinstance(self.arg0, str):
            # We have a variable name
            e = Var(self.arg0)
        else:
            # We expect an expression of some kind
            e = self.arg0

        # print('select e', e)
        key = get_key(_subst_data(e, state))
        # print('select key', key)

        result = self.arg1[key] if key in self.arg1 else self.arg2

        return _subst_data(result, state)


class Json(UnaryNode):
    """Render json - prepare for json.dumps"""

    def __call__(self, state: 'State') -> Any:
        data = _subst_data(self.arg0, state)

        # print('jsonifying', data)

        def to_json(data: Any) -> Any:
            if isinstance(data, list):
                return [to_json(d) for d in data]

            if isinstance(data, set):
                return list({to_json(d) for d in data})

            if isinstance(data, tuple):
                return list((to_json(d) for d in data))

            if isinstance(data, dict):
                return {k: to_json(v) for k, v in data.items()}

            return data

        return to_json(data)


class Or(BinaryNode):
    """arg0 if arg0 is truthy else arg1"""

    def __call__(self, state: 'State') -> Any:
        if subst_data(IsTruthy(self.arg0), state):
            result = self.arg0
        else:
            result = self.arg1

        return _subst_data(result, state)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} | {repr(self.arg1)})'


class And(BinaryNode):
    """arg1 if arg0 is truthy else None"""

    def __call__(self, state: 'State') -> Any:
        if subst_data(IsTruthy(self.arg0), state):
            result = self.arg1
        else:
            result = None

        return _subst_data(result, state)

    def __repr__(self) -> str:
        return f'({repr(self.arg0)} & {repr(self.arg1)})'


# class VarOr(BinaryNode):
#     """Var(arg0) if Var(arg0) is truthy else arg1"""
#
#     def __call__(self, state: 'State') -> Any:
#         if subst_data(IsTruthy(Var(self.arg0)), state):
#             result = Var(self.arg0)
#         else:
#             result = self.arg1
#
#         return _subst_data(result, state)
#
#
# class VarAnd(BinaryNode):
#     """arg1 if Var(arg0) is truthy else None"""
#
#     def __call__(self, state: 'State') -> Any:
#         if subst_data(IsTruthy(Var(self.arg0)), state):
#             result = self.arg1
#         else:
#             result = None
#
#         return _subst_data(result, state)
